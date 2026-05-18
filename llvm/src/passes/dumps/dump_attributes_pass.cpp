#include "docc/passes/dumps/dump_attributes_pass.h"

#include <sdfg/data_flow/library_nodes/stdlib/stdlib.h>
#include <sdfg/serializer/json_serializer.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "docc/analysis/attributes.h"
#include "docc/analysis/sdfg_registry.h"
#include "docc/target/tenstorrent/tenstorrent_offloading_node.h"
#include "docc/utils.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/alloca.h"
#include "sdfg/data_flow/library_nodes/stdlib/calloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"
#include "sdfg/data_flow/library_nodes/stdlib/memmove.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/types/type.h"
#include "symengine/symengine_rcp.h"

namespace docc {
namespace passes {

llvm::PreservedAnalyses DumpAttributesPass::
    run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    auto& registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    std::unordered_map<std::string, analysis::Attributes> sdfg_attributes_map;
    registry.for_each_sdfg_modifiable(Module, [&](sdfg::StructuredSDFG& sdfg) {
        sdfg::analysis::AnalysisManager analysis_manager(sdfg);
        auto& analysis = analysis_manager.get<AttributesAnalysis>();
        sdfg_attributes_map[sdfg.name()] = analysis.get();
    });

    // Read current index JSON
    std::filesystem::path build_path = analysis::SDFGRegistry::docc_extract_dir(Module);
    std::ifstream index_stream(build_path / "JSON");
    if (!index_stream.is_open()) {
        throw std::runtime_error("Failed to open SDFG index JSON");
    }
    json index_json;
    index_stream >> index_json;

    // Update attributes in JSON
    for (auto& sdfg_json : index_json["sdfgs"]) {
        auto& attributes = sdfg_attributes_map[sdfg_json["name"]];
        sdfg_json["attributes"] = attributes.to_json();
    }

    // Write back updated JSON
    std::ofstream output_stream(build_path / "JSON", std::ios::trunc);
    if (!output_stream.is_open()) {
        throw std::runtime_error("Failed to open SDFG index JSON for writing");
    }
    output_stream << index_json.dump(4);
    output_stream.close();

    return llvm::PreservedAnalyses::all();
}

AttributesAnalysis::AttributesAnalysis(sdfg::StructuredSDFG& sdfg) : sdfg::analysis::Analysis(sdfg) {}

std::string AttributesAnalysis::name() const { return "AttributesAnalysis"; }

void AttributesAnalysis::run(sdfg::analysis::AnalysisManager& analysis_manager) {
    this->attributes_ = analysis::Attributes();

    // Currently, we only support argument attributes for void-returning SDFGs
    if (this->sdfg_.return_type() != sdfg::types::Scalar(sdfg::types::PrimitiveType::Void) ||
        this->sdfg_.name() == "main") {
        return;
    }

    auto& root = this->sdfg_.root();

    // Create empty attributes and arguments map
    std::unordered_map<std::string, analysis::ArgumentAttributes> arguments_map;
    for (auto& argument : this->sdfg_.arguments()) {
        arguments_map.insert({argument, AttributesAnalysis::empty()});
    }

    // Buffer for allocations and frees
    std::unordered_map<std::string, sdfg::symbolic::Expression> buffer_alloc;
    std::unordered_set<std::string> buffer_free;

    // Collect offloading nodes at beginning of SDFG
    for (size_t i = 0; i < root.size(); i++) {
        // Assignments are not allowed
        if (!root.at(i).second.empty()) {
            break;
        }

        // Child must be a block
        auto* block = dynamic_cast<sdfg::structured_control_flow::Block*>(&root.at(i).first);
        if (!block) {
            break;
        }

        // Block must contain exactly one library node
        auto& dfg = block->dataflow();
        if (dfg.library_nodes().size() != 1 || dfg.tasklets().size() != 0) {
            break;
        }

        // Disallow containers with managed types
        bool managed = false;
        for (auto* access_node : dfg.data_nodes()) {
            if (dynamic_cast<sdfg::data_flow::ConstantNode*>(access_node)) {
                continue;
            }
            if (this->sdfg_.type(access_node->data()).storage_type().allocation() ==
                    sdfg::types::StorageType::Managed ||
                this->sdfg_.type(access_node->data()).storage_type().deallocation() ==
                    sdfg::types::StorageType::Managed) {
                managed = true;
                break;
            }
        }
        if (managed) {
            break;
        }

        // Library node must be a copy-in or allocation
        auto* libnode = *dfg.library_nodes().begin();
        if (auto* alloca_node = dynamic_cast<sdfg::stdlib::AllocaNode*>(libnode)) {
            auto* ret = this->get_out_access(alloca_node, "_ret");
            if (ret) {
                buffer_alloc.insert({ret->data(), alloca_node->size()});
            } else {
                break;
            }
        } else if (auto* calloc_node = dynamic_cast<sdfg::stdlib::CallocNode*>(libnode)) {
            auto* ret = this->get_out_access(calloc_node, "_ret");
            if (ret) {
                buffer_alloc.insert({ret->data(), sdfg::symbolic::mul(calloc_node->num(), calloc_node->size())});
            } else {
                break;
            }
        } else if (auto* malloc_node = dynamic_cast<sdfg::stdlib::MallocNode*>(libnode)) {
            auto* ret = this->get_out_access(malloc_node, "_ret");
            if (ret) {
                buffer_alloc.insert({ret->data(), malloc_node->size()});
            } else {
                break;
            }
        } else if (auto* memcpy_node = dynamic_cast<sdfg::stdlib::MemcpyNode*>(libnode)) {
            auto* src = this->get_in_access(memcpy_node, "_src");
            auto* dst = this->get_out_access(memcpy_node, "_dst");
            if (src && dst && arguments_map.contains(src->data())) {
                this->set_argument_attributes(
                    arguments_map.at(src->data()), "HOST", dst->data(), memcpy_node->count(), true, false, false, false
                );
            } else {
                break;
            }
        } else if (auto* memmove_node = dynamic_cast<sdfg::stdlib::MemmoveNode*>(libnode)) {
            auto* src = this->get_in_access(memmove_node, "_src");
            auto* dst = this->get_out_access(memmove_node, "_dst");
            if (src && dst && arguments_map.contains(src->data())) {
                this->set_argument_attributes(
                    arguments_map.at(src->data()), "HOST", dst->data(), memmove_node->count(), true, false, false, false
                );
            } else {
                break;
            }
        } else if (auto* cuda_offloading_node = dynamic_cast<sdfg::cuda::CUDADataOffloadingNode*>(libnode)) {
            if (cuda_offloading_node->is_h2d()) {
                auto* src = this->get_in_access(cuda_offloading_node, "_src");
                auto* dst = this->get_out_access(cuda_offloading_node, "_dst");
                if (src && dst && arguments_map.contains(src->data())) {
                    this->set_argument_attributes(
                        arguments_map.at(src->data()),
                        "CUDA",
                        dst->data(),
                        cuda_offloading_node->alloc_size(),
                        true,
                        false,
                        cuda_offloading_node->is_alloc(),
                        false
                    );
                } else {
                    break;
                }
            } else if (cuda_offloading_node->is_alloc()) {
                auto* ret = this->get_out_access(cuda_offloading_node, "_ret");
                if (ret) {
                    buffer_alloc.insert({ret->data(), cuda_offloading_node->alloc_size()});
                } else {
                    break;
                }
            } else {
                break;
            }
        } else if (auto* tt_offloading_node = dynamic_cast<sdfg::tenstorrent::TTDataOffloadingNode*>(libnode)) {
            if (tt_offloading_node->is_h2d()) {
                auto* src = this->get_in_access(tt_offloading_node, "_src");
                auto* dst = this->get_out_access(tt_offloading_node, "_dst");
                if (src && dst && arguments_map.contains(src->data())) {
                    this->set_argument_attributes(
                        arguments_map.at(src->data()),
                        "TENSTORRENT",
                        dst->data(),
                        tt_offloading_node->alloc_size(),
                        true,
                        false,
                        tt_offloading_node->is_alloc(),
                        false
                    );
                } else {
                    break;
                }
            } else if (tt_offloading_node->is_alloc()) {
                auto* ret = this->get_out_access(tt_offloading_node, "_ret");
                if (ret) {
                    buffer_alloc.insert({ret->data(), tt_offloading_node->alloc_size()});
                } else {
                    break;
                }
            } else {
                break;
            }
        } else if (auto* external_offloading_node = dynamic_cast<sdfg::offloading::ExternalDataOffloadingNode*>(libnode
                   )) {
            if (external_offloading_node->is_h2d()) {
                auto* src = this->get_in_access(external_offloading_node, external_offloading_node->inputs().back());
                auto* dst = this->get_out_access(
                    external_offloading_node,
                    external_offloading_node->input(external_offloading_node->transfer_index())
                );
                if (src && dst && arguments_map.contains(src->data())) {
                    this->set_argument_attributes(
                        arguments_map.at(src->data()),
                        "EXTERNAL",
                        dst->data(),
                        external_offloading_node->alloc_size(),
                        true,
                        false,
                        external_offloading_node->is_alloc(),
                        false
                    );
                } else {
                    break;
                }
            } else if (external_offloading_node->is_alloc()) {
                auto* ret = this->get_out_access(external_offloading_node, "_ret");
                if (ret) {
                    buffer_alloc.insert({ret->data(), external_offloading_node->alloc_size()});
                } else {
                    break;
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Find last node before return
    long long last_node = root.size() - 1;
    for (size_t i = 0; i < root.size(); i++) {
        if (dynamic_cast<sdfg::structured_control_flow::Return*>(&root.at(i).first)) {
            last_node = i - 1;
            break;
        }
    }

    // Collect offloading nodes at end of SDFG
    for (long long i = last_node; i >= 0; i--) {
        // Assignments are not allowed
        if (!root.at(i).second.empty()) {
            break;
        }

        // Child must be a block
        auto* block = dynamic_cast<sdfg::structured_control_flow::Block*>(&root.at(i).first);
        if (!block) {
            break;
        }

        // Block must contain exactly one library node
        auto& dfg = block->dataflow();
        if (dfg.library_nodes().size() != 1 || dfg.tasklets().size() != 0) {
            break;
        }

        // Library node symbols must only depend on arguments
        auto* libnode = *dfg.library_nodes().begin();
        bool symbols_invalid = false;
        for (auto& symbol : libnode->symbols()) {
            if (!this->sdfg_.is_argument(symbol->get_name())) {
                symbols_invalid = true;
                break;
            }
        }
        if (symbols_invalid) {
            break;
        }

        // Disallow containers with managed types
        bool managed = false;
        for (auto* access_node : dfg.data_nodes()) {
            if (dynamic_cast<sdfg::data_flow::ConstantNode*>(access_node)) {
                continue;
            }
            if (this->sdfg_.type(access_node->data()).storage_type().allocation() ==
                    sdfg::types::StorageType::Managed ||
                this->sdfg_.type(access_node->data()).storage_type().deallocation() ==
                    sdfg::types::StorageType::Managed) {
                managed = true;
                break;
            }
        }
        if (managed) {
            break;
        }

        // Library node must be a copy-out or free
        if (auto* free_node = dynamic_cast<sdfg::stdlib::FreeNode*>(libnode)) {
            auto* ptr_in = this->get_in_access(free_node, "_ptr");
            auto* ptr_out = this->get_out_access(free_node, "_ptr");
            if (ptr_in && ptr_out && ptr_in->data() == ptr_out->data()) {
                buffer_free.insert(ptr_in->data());
            } else {
                break;
            }
        } else if (auto* memcpy_node = dynamic_cast<sdfg::stdlib::MemcpyNode*>(libnode)) {
            auto* src = this->get_in_access(memcpy_node, "_src");
            auto* dst = this->get_out_access(memcpy_node, "_dst");
            if (src && dst && arguments_map.contains(dst->data())) {
                this->set_argument_attributes(
                    arguments_map.at(dst->data()), "HOST", src->data(), memcpy_node->count(), false, true, false, false
                );
            } else {
                break;
            }
        } else if (auto* cuda_offloading_node = dynamic_cast<sdfg::cuda::CUDADataOffloadingNode*>(libnode)) {
            if (cuda_offloading_node->is_d2h()) {
                auto* src = this->get_in_access(cuda_offloading_node, "_src");
                auto* dst = this->get_out_access(cuda_offloading_node, "_dst");
                if (src && dst && arguments_map.contains(dst->data())) {
                    this->set_argument_attributes(
                        arguments_map.at(dst->data()),
                        "CUDA",
                        src->data(),
                        cuda_offloading_node->alloc_size(),
                        false,
                        true,
                        false,
                        cuda_offloading_node->is_free()
                    );
                } else {
                    break;
                }
            } else if (cuda_offloading_node->is_free()) {
                auto* ptr_in = this->get_in_access(cuda_offloading_node, "_ptr");
                auto* ptr_out = this->get_out_access(cuda_offloading_node, "_ptr");
                if (ptr_in && ptr_out && ptr_in->data() == ptr_out->data()) {
                    buffer_free.insert(ptr_in->data());
                } else {
                    break;
                }
            } else {
                break;
            }
        } else if (auto* tt_offloading_node = dynamic_cast<sdfg::tenstorrent::TTDataOffloadingNode*>(libnode)) {
            if (tt_offloading_node->is_d2h()) {
                auto* src = this->get_in_access(tt_offloading_node, "_src");
                auto* dst = this->get_out_access(tt_offloading_node, "_dst");
                if (src && dst && arguments_map.contains(dst->data())) {
                    this->set_argument_attributes(
                        arguments_map.at(dst->data()),
                        "TENSTORRENT",
                        src->data(),
                        tt_offloading_node->alloc_size(),
                        false,
                        true,
                        false,
                        tt_offloading_node->is_free()
                    );
                } else {
                    break;
                }
            } else if (tt_offloading_node->is_free()) {
                // TT free nodes do not generate code and are basically useless, but for completeness they can be
                // detected here
                auto* ptr_in = this->get_in_access(tt_offloading_node, "_ptr");
                auto* ptr_out = this->get_out_access(tt_offloading_node, "_ptr");
                if (ptr_in && ptr_out && ptr_in->data() == ptr_out->data()) {
                    buffer_free.insert(ptr_in->data());
                } else {
                    break;
                }
            } else {
                break;
            }
        } else if (auto* external_offloading_node = dynamic_cast<sdfg::offloading::ExternalDataOffloadingNode*>(libnode
                   )) {
            if (external_offloading_node->is_d2h()) {
                auto* src = this->get_in_access(
                    external_offloading_node,
                    external_offloading_node->input(external_offloading_node->transfer_index())
                );
                auto* dst = this->get_out_access(external_offloading_node, "_ret");
                if (src && dst && arguments_map.contains(dst->data())) {
                    this->set_argument_attributes(
                        arguments_map.at(dst->data()),
                        "EXTERNAL",
                        src->data(),
                        external_offloading_node->alloc_size(),
                        false,
                        true,
                        false,
                        external_offloading_node->is_free()
                    );
                } else {
                    break;
                }
            } else if (external_offloading_node->is_free()) {
                auto* ptr_in = this->get_in_access(external_offloading_node, "_ptr");
                auto* ptr_out = this->get_out_access(external_offloading_node, "_ptr");
                if (ptr_in && ptr_out && ptr_in->data() == ptr_out->data()) {
                    buffer_free.insert(ptr_in->data());
                } else {
                    break;
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Check buffers for allocations and frees
    for (auto& argument : this->sdfg_.arguments()) {
        if (!arguments_map.contains(argument)) {
            return;
        }
        auto& attrs = arguments_map.at(argument);
        if (buffer_alloc.contains(attrs.copy_buffer)) {
            if (!sdfg::symbolic::null_safe_eq(buffer_alloc.at(attrs.copy_buffer), attrs.copy_size)) {
                return;
            }
            attrs.alloc = true;
        }
        if (buffer_free.contains(attrs.copy_buffer)) {
            attrs.free = true;
        }

        // Remove if only allocation or only free exists
        if (attrs.alloc != attrs.free) {
            arguments_map.at(argument) = AttributesAnalysis::empty();
        }

        // Remove if copy buffer is an argument or external
        if (this->sdfg_.is_argument(attrs.copy_buffer) || this->sdfg_.is_external(attrs.copy_buffer)) {
            arguments_map.at(argument) = AttributesAnalysis::empty();
        }
    }

    // Create vector
    for (auto& argument : this->sdfg_.arguments()) {
#ifndef NDEBUG
        this->debug_print(argument, arguments_map.at(argument));
#endif
        this->attributes_.arguments.push_back(arguments_map.at(argument));
    }
}

const analysis::Attributes& AttributesAnalysis::get() { return attributes_; }

analysis::ArgumentAttributes AttributesAnalysis::empty() {
    return {
        .copy_target = "",
        .copy_buffer = "",
        .copy_size = SymEngine::null,
        .copy_in = false,
        .copy_out = false,
        .alloc = false,
        .free = false
    };
}

sdfg::data_flow::AccessNode* AttributesAnalysis::get_in_access(sdfg::data_flow::CodeNode* node, const std::string& conn) {
    auto& dfg = node->get_parent();
    for (auto& iedge : dfg.in_edges(*node)) {
        if (iedge.dst_conn() == conn) {
            return dynamic_cast<sdfg::data_flow::AccessNode*>(&iedge.src());
        }
    }
    return nullptr;
}

sdfg::data_flow::AccessNode* AttributesAnalysis::get_out_access(sdfg::data_flow::CodeNode* node, const std::string& conn) {
    auto& dfg = node->get_parent();
    for (auto& oedge : dfg.out_edges(*node)) {
        if (oedge.src_conn() == conn) {
            return static_cast<sdfg::data_flow::AccessNode*>(&oedge.dst());
        }
    }
    return nullptr;
}

void AttributesAnalysis::debug_print(const std::string& argument, const analysis::ArgumentAttributes& attrs) {
    if (attrs.copy_buffer.empty()) {
        return;
    }
    LLVM_DEBUG_PRINTLN(
        "Attributes of argument '" << argument << "': '" << attrs.copy_buffer << "' " << attrs.copy_target
                                   << (attrs.alloc ? " ALLOC" : "      ") << (attrs.copy_in ? " COPY-IN" : "        ")
                                   << (attrs.copy_out ? " COPY-OUT" : "         ") << (attrs.free ? " FREE" : "     ")
                                   << (attrs.copy_size.is_null() ? "" : " " + attrs.copy_size->__str__())
    );
}

void AttributesAnalysis::set_argument_attributes(
    analysis::ArgumentAttributes& attrs,
    const std::string& copy_target,
    const std::string& copy_buffer,
    const sdfg::symbolic::Expression& copy_size,
    bool copy_in,
    bool copy_out,
    bool alloc,
    bool free
) {
    if (attrs.copy_target.empty()) {
        attrs.copy_target = copy_target;
        attrs.copy_buffer = copy_buffer;
        attrs.copy_size = copy_size;
        attrs.copy_in = copy_in;
        attrs.copy_out = copy_out;
        attrs.alloc = alloc;
        attrs.free = free;
    } else if (attrs.copy_target == copy_target && attrs.copy_buffer == copy_buffer &&
               sdfg::symbolic::null_safe_eq(attrs.copy_size, copy_size)) {
        attrs.copy_in = attrs.copy_in || copy_in;
        attrs.copy_out = attrs.copy_out || copy_out;
        attrs.alloc = attrs.alloc || alloc;
        attrs.free = attrs.free || free;
    } else {
        attrs = AttributesAnalysis::empty();
    }
}

} // namespace passes
} // namespace docc
