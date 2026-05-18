#include "docc/passes/inlining/argument_expansion_pass.h"

#include <cstddef>
#include <string>
#include <vector>

#include "docc/utils.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/call_node.h"
#include "sdfg/element.h"
#include "sdfg/passes/code_motion/extended_block_hoisting.h"
#include "sdfg/passes/code_motion/extended_block_sorting.h"
#include "sdfg/passes/extended_data_transfer_minimization_pass.h"
#include "sdfg/passes/offloading/data_transfer_minimization_pass.h"
#include "sdfg/passes/offloading/remove_redundant_transfers_pass.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/passes/readonly_transfer_hoisting_pass.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/types/function.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace docc {
namespace passes {

bool ArgumentExpansionPass::expand_arguments(
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    const std::string& callee_name,
    const docc::analysis::Attributes& attributes,
    const std::string& target
) {
    auto& users_analysis = analysis_manager.get<sdfg::analysis::Users>();
    auto& scope_analysis = analysis_manager.get<sdfg::analysis::ScopeAnalysis>();

    // find all call nodes that reference the candidate SDFG
    for (auto& user : users_analysis.uses(callee_name)) {
        auto element = dynamic_cast<sdfg::data_flow::CallNode*>(user->element());
        if (!element) {
            continue;
        }
        auto& graph = element->get_parent();
        auto& block = static_cast<sdfg::structured_control_flow::Block&>(*graph.get_parent());

        // Limitation: isolated call nodes only
        for (auto& iedge : graph.in_edges(*element)) {
            if (graph.in_degree(iedge.src()) != 0) {
                return false;
            }
        }
        for (auto& oedge : graph.out_edges(*element)) {
            if (graph.out_degree(oedge.dst()) != 0) {
                return false;
            }
        }

        // Collect inputs
        std::vector<std::string> inputs = element->inputs();
        std::unordered_map<std::string, sdfg::data_flow::Memlet*> iedges;
        for (auto& iedge : graph.in_edges(*element)) {
            iedges[iedge.dst_conn()] = &iedge;
        }

        // Define relevant types
        sdfg::types::Scalar void_type(sdfg::types::PrimitiveType::Void);
        sdfg::types::Pointer opaque_ptr;
        sdfg::types::Function alloc_function(opaque_ptr, false);
        sdfg::types::Function copy_in_function(opaque_ptr, false);
        sdfg::types::Function copy_out_function(void_type, false);
        sdfg::types::Function free_function(opaque_ptr, false);
        sdfg::types::Function kernel_function(void_type, false);
        for (const auto& inp : inputs) {
            auto iedge = iedges.at(inp);
            alloc_function.add_param(iedge->base_type());
            copy_in_function.add_param(iedge->base_type());
            copy_out_function.add_param(iedge->base_type());
            free_function.add_param(iedge->base_type());
            kernel_function.add_param(iedge->base_type());
        }
        copy_in_function.add_param(opaque_ptr);
        copy_out_function.add_param(opaque_ptr);
        free_function.add_param(opaque_ptr);
        for (const auto& arg_attrs : attributes.arguments) {
            if (arg_attrs.copy_target.empty()) {
                continue;
            }
            kernel_function.add_param(opaque_ptr);
        }

        std::vector<std::string> arg_to_buffer_map;
        for (const auto& arg_attrs : attributes.arguments) {
            if (arg_attrs.copy_target.empty()) {
                arg_to_buffer_map.push_back("");
                continue;
            }

            std::string buffer_name = builder.find_new_name(arg_attrs.copy_buffer);
            arg_to_buffer_map.push_back(buffer_name);

            sdfg::types::Pointer buffer_type;
            if (target == "CUDA") {
                buffer_type.storage_type().value("NV_Generic");
            } else {
                buffer_type.storage_type().value("CPU_Stack");
            }
            buffer_type.storage_type().allocation(sdfg::types::StorageType::AllocationType::Unmanaged);
            buffer_type.storage_type().deallocation(sdfg::types::StorageType::AllocationType::Unmanaged);

            builder.add_container(buffer_name, buffer_type, false, false);
        }

        auto& parent = static_cast<sdfg::structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
        auto& temporary_parent = builder.add_sequence_after(parent, block);

        // Define copy-in and allocation operations
        for (size_t i = 0; i < attributes.arguments.size(); i++) {
            const auto& arg_attrs = attributes.arguments.at(i);
            if (arg_attrs.copy_target.empty()) {
                continue;
            }
            if (!arg_attrs.alloc && !arg_attrs.copy_in) {
                continue;
            }

            if (arg_attrs.alloc) {
                const std::string callee_alloc = callee_name + "_alloc_" + std::to_string(i);
                if (builder.subject().exists(callee_alloc)) {
                    assert(builder.subject().type(callee_alloc) == alloc_function);
                } else {
                    builder.add_container(callee_alloc, alloc_function, false, true);
                }
            }

            if (arg_attrs.copy_in) {
                const std::string callee_copy_in = callee_name + "_in_" + std::to_string(i);
                if (builder.subject().exists(callee_copy_in)) {
                    assert(builder.subject().type(callee_copy_in) == copy_in_function);
                } else {
                    builder.add_container(callee_copy_in, copy_in_function, false, true);
                }
            }

            auto& block = builder.add_block(temporary_parent);
            auto& lib_node = builder.add_library_node<sdfg::offloading::ExternalDataOffloadingNode>(
                block,
                sdfg::DebugInfo(),
                inputs,
                callee_name,
                i,
                (arg_attrs.copy_in ? sdfg::offloading::DataTransferDirection::H2D
                                   : sdfg::offloading::DataTransferDirection::NONE),
                (arg_attrs.alloc ? sdfg::offloading::BufferLifecycle::ALLOC
                                 : sdfg::offloading::BufferLifecycle::NO_CHANGE)
            );

            std::string buffer_name = arg_to_buffer_map.at(i);
            auto& out_node = builder.add_access(block, buffer_name);
            builder.add_computational_memlet(block, lib_node, "_ret", out_node, {}, opaque_ptr, sdfg::DebugInfo());

            for (size_t j = 0; j < inputs.size(); j++) {
                auto& inp = inputs.at(j);
                auto iedge = iedges.at(inp);
                auto& iedge_src = static_cast<sdfg::data_flow::AccessNode&>(iedge->src());
                sdfg::data_flow::AccessNode* in_node;
                if (i != j && !attributes.arguments.at(j).copy_buffer.empty()) {
                    in_node = &builder.add_constant(block, sdfg::symbolic::__nullptr__()->__str__(), opaque_ptr);
                } else if (auto* iedge_src_const = dynamic_cast<sdfg::data_flow::ConstantNode*>(&iedge_src)) {
                    in_node = &builder.add_constant(block, iedge_src.data(), iedge_src_const->type());
                } else {
                    in_node = &builder.add_access(block, iedge_src.data());
                }
                builder.add_memlet(
                    block, *in_node, "void", lib_node, inp, iedge->subset(), iedge->base_type(), iedge->debug_info()
                );
            }

            if (arg_attrs.copy_in) {
                auto& out_in_node = builder.add_access(block, buffer_name);
                builder
                    .add_computational_memlet(block, out_in_node, lib_node, "_ret", {}, opaque_ptr, sdfg::DebugInfo());
            }
        }

        // Define kernel call
        {
            std::string callee_kernel = callee_name + "_kernel";
            if (builder.subject().exists(callee_kernel)) {
                assert(builder.subject().type(callee_kernel) == kernel_function);
            } else {
                builder.add_container(callee_kernel, kernel_function, false, true);
            }

            std::vector<std::string> kernel_outputs;
            for (size_t i = 0; i < inputs.size(); i++) {
                auto& iedge = iedges.at(inputs.at(i));
                if (iedge->base_type().type_id() != sdfg::types::TypeID::Pointer ||
                    (!attributes.arguments.at(i).copy_buffer.empty() && !attributes.arguments.at(i).copy_out)) {
                    continue;
                }
                kernel_outputs.push_back(iedge->dst_conn());
            }
            std::vector<std::string> kernel_inputs = inputs;
            for (const auto& attrs : attributes.arguments) {
                if (attrs.copy_target == "") {
                    continue;
                }
                kernel_outputs.push_back("_arg" + std::to_string(kernel_inputs.size()));
                kernel_inputs.push_back("_arg" + std::to_string(kernel_inputs.size()));
            }

            auto& kernel_block = builder.add_block(temporary_parent);
            auto& lib_node = builder.add_library_node<
                sdfg::data_flow::CallNode>(kernel_block, sdfg::DebugInfo(), callee_kernel, kernel_outputs, kernel_inputs);

            for (size_t i = 0; i < inputs.size(); i++) {
                auto& inp = inputs.at(i);
                auto iedge = iedges.at(inp);
                if (iedge->base_type().type_id() == sdfg::types::TypeID::Pointer &&
                    !attributes.arguments.at(i).copy_buffer.empty() && !attributes.arguments.at(i).copy_out) {
                    auto& in_node =
                        builder.add_constant(kernel_block, sdfg::symbolic::__nullptr__()->__str__(), opaque_ptr);
                    builder.add_memlet(
                        kernel_block, in_node, "void", lib_node, inp, {}, iedge->base_type(), iedge->debug_info()
                    );
                    continue;
                }

                auto& iedge_src = static_cast<sdfg::data_flow::AccessNode&>(iedge->src());
                if (auto* iedge_src_const = dynamic_cast<sdfg::data_flow::ConstantNode*>(&iedge_src)) {
                    auto& in_node = builder.add_constant(kernel_block, iedge_src.data(), iedge_src_const->type());
                    builder.add_memlet(
                        kernel_block,
                        in_node,
                        "void",
                        lib_node,
                        inp,
                        iedge->subset(),
                        iedge->base_type(),
                        iedge->debug_info()
                    );
                } else {
                    auto& in_node = builder.add_access(kernel_block, iedge_src.data());
                    builder.add_memlet(
                        kernel_block,
                        in_node,
                        "void",
                        lib_node,
                        inp,
                        iedge->subset(),
                        iedge->base_type(),
                        iedge->debug_info()
                    );
                }
                if (iedge->base_type().type_id() != sdfg::types::TypeID::Pointer) {
                    continue;
                }

                auto& out_node = builder.add_access(kernel_block, iedge_src.data());
                builder.add_memlet(
                    kernel_block,
                    lib_node,
                    inp,
                    out_node,
                    "void",
                    iedge->subset(),
                    iedge->base_type(),
                    iedge->debug_info()
                );
            }
            // Connect argument buffers
            size_t arg_index = inputs.size();
            for (size_t i = 0; i < attributes.arguments.size(); i++) {
                const auto& arg_attrs = attributes.arguments.at(i);
                if (arg_attrs.copy_target.empty()) {
                    continue;
                }

                std::string buffer_name = arg_to_buffer_map.at(i);
                auto& in_node = builder.add_access(kernel_block, buffer_name);
                builder.add_computational_memlet(
                    kernel_block,
                    in_node,
                    lib_node,
                    "_arg" + std::to_string(arg_index),
                    {},
                    opaque_ptr,
                    sdfg::DebugInfo()
                );
                auto& out_node = builder.add_access(kernel_block, buffer_name);
                builder.add_computational_memlet(
                    kernel_block,
                    lib_node,
                    "_arg" + std::to_string(arg_index),
                    out_node,
                    {},
                    opaque_ptr,
                    sdfg::DebugInfo()
                );

                arg_index++;
            }
        }

        // Define copy-out operations
        for (size_t i = 0; i < attributes.arguments.size(); i++) {
            const auto& arg_attrs = attributes.arguments.at(i);
            if (arg_attrs.copy_target.empty()) {
                continue;
            }
            if (!arg_attrs.copy_out && !arg_attrs.free) {
                continue;
            }

            if (arg_attrs.copy_out) {
                const std::string callee_copy_out = callee_name + "_out_" + std::to_string(i);
                if (builder.subject().exists(callee_copy_out)) {
                    assert(builder.subject().type(callee_copy_out) == copy_out_function);
                } else {
                    builder.add_container(callee_copy_out, copy_out_function, false, true);
                }
            }

            if (arg_attrs.free) {
                const std::string callee_free = callee_name + "_free_" + std::to_string(i);
                if (builder.subject().exists(callee_free)) {
                    assert(builder.subject().type(callee_free) == free_function);
                } else {
                    builder.add_container(callee_free, free_function, false, true);
                }
            }

            auto& block = builder.add_block(temporary_parent);
            auto& lib_node = builder.add_library_node<sdfg::offloading::ExternalDataOffloadingNode>(
                block,
                sdfg::DebugInfo(),
                inputs,
                callee_name,
                i,
                (arg_attrs.copy_out ? sdfg::offloading::DataTransferDirection::D2H
                                    : sdfg::offloading::DataTransferDirection::NONE),
                (arg_attrs.free ? sdfg::offloading::BufferLifecycle::FREE : sdfg::offloading::BufferLifecycle::NO_CHANGE
                )
            );

            std::string buffer_name = arg_to_buffer_map.at(i);
            auto& buffer_in = builder.add_access(block, buffer_name);
            builder.add_computational_memlet(
                block, buffer_in, lib_node, lib_node.inputs().back(), {}, opaque_ptr, sdfg::DebugInfo()
            );

            for (size_t j = 0; j < inputs.size(); j++) {
                auto& inp = inputs.at(j);
                auto iedge = iedges.at(inp);
                auto& iedge_src = static_cast<sdfg::data_flow::AccessNode&>(iedge->src());
                sdfg::data_flow::AccessNode* in_node;
                if (i != j && !attributes.arguments.at(j).copy_buffer.empty()) {
                    in_node = &builder.add_constant(block, sdfg::symbolic::__nullptr__()->__str__(), opaque_ptr);
                } else if (auto* iedge_src_const = dynamic_cast<sdfg::data_flow::ConstantNode*>(&iedge_src)) {
                    in_node = &builder.add_constant(block, iedge_src.data(), iedge_src_const->type());
                } else {
                    in_node = &builder.add_access(block, iedge_src.data());
                }
                builder.add_memlet(
                    block, *in_node, "void", lib_node, inp, iedge->subset(), iedge->base_type(), iedge->debug_info()
                );

                if (arg_attrs.copy_out && inp == inputs.at(i)) {
                    auto& out_node = builder.add_access(block, iedge_src.data());
                    builder.add_computational_memlet(
                        block, lib_node, inp, out_node, iedge->subset(), iedge->base_type(), sdfg::DebugInfo()
                    );
                }
            }

            if (!arg_attrs.copy_out) {
                auto& buffer_out = builder.add_access(block, buffer_name);
                builder.add_computational_memlet(block, lib_node, "_ptr", buffer_out, {}, opaque_ptr, sdfg::DebugInfo());
            }
        }

        size_t block_index = parent.index(block);
        builder.remove_child(parent, block_index);
        builder.move_children(temporary_parent, parent, block_index);
    }
    LLVM_DEBUG_PRINTLN("Inlining SDFG '" << callee_name << "' at call node in SDFG '" << builder.subject().name() << "'");
    analysis_manager.invalidate_all();

    return true;
}

llvm::PreservedAnalyses ArgumentExpansionPass::
    run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    auto& registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    registry.for_each_sdfg_modifiable(Module, [&](sdfg::StructuredSDFG& sdfg) {
        sdfg::builder::StructuredSDFGBuilder builder(sdfg);
        sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

        // Find other SDFGs that can be expanded into this one
        bool applied = false;
        auto externals = builder.subject().externals();
        for (auto& external : externals) {
            if (registry.has_function(external)) {
                auto& candidate_attrs = registry.attributes(external);
                if (candidate_attrs.arguments.empty()) {
                    continue;
                }
                bool supported = true;
                std::string target;
                for (const auto& arg_attrs : candidate_attrs.arguments) {
                    if (arg_attrs.copy_target.empty()) {
                        continue;
                    }
                    if (target.empty()) {
                        target = arg_attrs.copy_target;
                    } else if (arg_attrs.copy_target != target) {
                        supported = false;
                        break;
                    }
                }
                if (supported && !target.empty() && target != "TENSTORRENT") {
                    LLVM_DEBUG_PRINTLN(
                        "Expanding arguments of SDFG '" << external << "' into SDFG '" << builder.subject().name()
                                                        << "'"
                    );
                    applied |= this->expand_arguments(builder, analysis_manager, external, candidate_attrs, target);
                }
            }
            if (registry.has_external_function(external)) {
                auto& candidate_attrs = registry.external_attributes(external);
                if (candidate_attrs.arguments.empty()) {
                    continue;
                }
                bool supported = true;
                std::string target;
                for (const auto& arg_attrs : candidate_attrs.arguments) {
                    if (arg_attrs.copy_target.empty()) {
                        continue;
                    }
                    if (target.empty()) {
                        target = arg_attrs.copy_target;
                    } else if (arg_attrs.copy_target != target) {
                        supported = false;
                        break;
                    }
                }
                if (supported && target == "EXTERNAL") {
                    LLVM_DEBUG_PRINTLN(
                        "Expanding arguments of shared library SDFG '" << external << "' into SDFG '"
                                                                       << builder.subject().name() << "'"
                    );
                    applied |= this->expand_arguments(builder, analysis_manager, external, candidate_attrs);
                }
            }
        }

        if (applied) {
            analysis_manager.invalidate_all();
            sdfg::passes::Pipeline code_motion("Code Motion");
            code_motion.register_pass<sdfg::passes::DeadCFGElimination>();
            code_motion.register_pass<sdfg::passes::ExtendedBlockHoistingPass>();
            code_motion.register_pass<sdfg::passes::ExtendedBlockSortingPass>();
            code_motion.run(builder, analysis_manager);

            analysis_manager.invalidate_all();
            sdfg::passes::ExtendedDataTransferMinimizationPass data_transfer_minimization_pass;
            data_transfer_minimization_pass.run(builder, analysis_manager);

            analysis_manager.invalidate_all();
            sdfg::passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
            readonly_transfer_hoisting_pass.run(builder, analysis_manager);

            analysis_manager.invalidate_all();
            sdfg::passes::RemoveRedundantTransfersPass remove_redundant_transfers_pass;
            remove_redundant_transfers_pass.run(builder, analysis_manager);
        }
    });

    return llvm::PreservedAnalyses::all();
}

} // namespace passes
} // namespace docc
