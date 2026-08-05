#include "sdfg/passes/statistics.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sdfg/helpers/helpers.h"
#include "sdfg/passes/expansion/lib_node_expander.h"

namespace sdfg {
namespace passes {

constexpr bool LOG_SCOPE_ENTRY_EXIT = false;


HierarchicalStatistics::Node* HierarchicalStatistics::enter_scope(const std::string& name, const std::string& scope_type) {
    auto node = std::make_unique<Node>();
    node->scope_type = scope_type;
    node->name = name;
    node->parent = current_;
    node->enter_time = std::chrono::high_resolution_clock::now();

    if (LOG_SCOPE_ENTRY_EXIT) {
        DEBUG_PRINTLN("Started " << scope_type << " '" << name);
    }

    Node* raw = node.get();
    if (current_ == nullptr) {
        roots_.push_back(std::move(node));
    } else {
        current_->children.push_back(std::move(node));
    }
    current_ = raw;
    return raw;
}

void HierarchicalStatistics::exit_scope() {
    if (current_ == nullptr) {
        return;
    }
    current_->exit_time = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_->exit_time - current_->enter_time).count();

    if (LOG_SCOPE_ENTRY_EXIT) {
        DEBUG_PRINTLN("Finished " << current_->scope_type << " '" << current_->name << "' in " << duration << " ms");
    }

    current_ = current_->parent;
}

void HierarchicalStatistics::set_metric(const std::string& key, uint64_t value) {
    if (current_ == nullptr) {
        return;
    }
    current_->metrics[key] = value;
}

void HierarchicalStatistics::add_metric(const std::string& key, uint64_t value) {
    if (current_ == nullptr) {
        return;
    }
    current_->metrics[key] += value;
}

void HierarchicalStatistics::clear() {
    roots_.clear();
    current_ = nullptr;
}

int statistics_mode_env() {
    const char* val = std::getenv(DOCC_STATISTICS_ENV);
    if (val == nullptr) {
        return 0;
    }
    return std::stoi(val);
}

uint64_t HierarchicalStatistics::Node::duration_ms() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(this->exit_time - this->enter_time).count();
}

void HierarchicalStatistics::print_node_self(std::ostream& stream, const HierarchicalStatistics::Node& node, int depth) {
    std::string indent(2 * (depth + 1), ' ');
    stream << indent << node.scope_type << " " << node.name << "  " << node.duration_ms() << " ms  ";
    if (!node.metrics.empty()) {
        std::vector<std::pair<std::string, uint64_t>> metrics(node.metrics.begin(), node.metrics.end());
        std::sort(metrics.begin(), metrics.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        stream << "  [";
        for (size_t i = 0; i < metrics.size(); ++i) {
            if (i > 0) {
                stream << ", ";
            }
            stream << metrics[i].first << "=" << metrics[i].second;
        }
        stream << "]";
    }
    stream << std::endl;
}

HierarchicalStatistics::RemapAction HierarchicalStatistics::custom_print_node(std::ostream& out, const Node&, int depth) {
    return RemapAction::Proceed;
}

void HierarchicalStatistics::print_node(std::ostream& stream, const HierarchicalStatistics::Node& node, int depth) {
    bool descend_children = true;
    RemapAction action = custom_print_node(stream, node, depth);
    if (action == RemapAction::Skip) {
        return;
    } else if (action == RemapAction::NoDescend) {
        descend_children = false;
    }

    print_node_self(stream, node, depth);
    if (descend_children) {
        for (const auto& child : node.children) {
            print_node(stream, *child, depth + 1);
        }
    }
}

std::string HierarchicalStatistics::report(const std::string& title) {
    if (roots_.empty()) {
        return "";
    }
    std::stringstream stream;
    stream << title << std::endl;
    for (const auto& root : roots_) {
        print_node(stream, *root, 0);
    }
    return stream.str();
}

bool CompileStatistics::enabled_ = false;

std::string CompileStatistics::summary() { return report(ReportLevel::SummarizePipelineContentsAndAnalysis); }

HierarchicalStatistics::RemapAction CompileStatistics::custom_print_node(std::ostream& out, const Node& node, int depth) {
    auto level = report_detail_;
    if (node.scope_type == PIPELINE_SCOPE) {
        if (level == ReportLevel::SummarizePipelineContentsAndAnalysis) {
            print_node_self(out, node, depth);
            summarize_pipeline_passes(out, node, depth);
            return RemapAction::Skip;
        } else if (level == ReportLevel::PipelineTotalsOnly) {
            return RemapAction::NoDescend;
        } else {
            return RemapAction::Proceed;
        }
    } else if (node.scope_type == ANALYSIS_SCOPE || node.scope_type == PASS_SCOPE) {
        if (level != ReportLevel::All) {
            return RemapAction::NoDescend;
        } else {
            return RemapAction::Proceed;
        }
    } else if (node.scope_type == ANALYSIS_MGR_SCOPE && depth < 2 && level != ReportLevel::All) {
        return RemapAction::Skip;
    }
    return RemapAction::Proceed;
}

void CompileStatistics::summarize_pipeline_passes(std::ostream& out, const HierarchicalStatistics::Node& node, int depth) {
    // Aggregate direct children only: passes by name, everything else under "other" by scope type.
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> passes;
    std::vector<std::string> pass_order;
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> others;

    for (const auto& child : node.children) {
        if (child->scope_type == PASS_SCOPE) {
            auto [it, inserted] = passes.try_emplace(child->name, 0, 0);
            if (inserted) {
                pass_order.push_back(child->name);
            }
            it->second.first += 1;
            it->second.second += child->duration_ms();
        } else {
            auto& bucket = others[child->scope_type];
            bucket.first += 1;
            bucket.second += child->duration_ms();
        }
    }

    auto compare_data_fn = [](const std::tuple<std::string, uint64_t, uint64_t>& a,
                              const std::tuple<std::string, uint64_t, uint64_t>& b) {
        auto [a_name, a_count, a_milliseconds] = a;
        auto [b_name, b_count, b_milliseconds] = b;
        return a_milliseconds > b_milliseconds || (a_milliseconds == b_milliseconds && a_count > b_count) ||
               (a_milliseconds == b_milliseconds && a_count == b_count && a_name < b_name);
    };

    std::string indent(2 * (depth + 2), ' ');

    // Passes are reported in first-encounter order, matching the pipeline execution order.
    for (const auto& name : pass_order) {
        const auto& entry = passes[name];
        out << indent << entry.first << " x " << name << "  " << entry.second << " ms" << std::endl;
    }

    std::vector<std::tuple<std::string, uint64_t, uint64_t>> other_data;
    for (const auto& [scope_type, entry] : others) {
        other_data.push_back({scope_type, entry.first, entry.second});
    }
    std::sort(other_data.begin(), other_data.end(), compare_data_fn);
    for (const auto& [scope_type, count, milliseconds] : other_data) {
        out << indent << "Other: " << milliseconds << " ms  " << count << "  " << scope_type << std::endl;
    }
}

std::string CompileStatistics::report(ReportLevel detail) {
    this->report_detail_ = detail;

    return HierarchicalStatistics::report("DOCC Statistics:");
}


void CodegenStatistics::add_codegen(const std::string& name, uint64_t milliseconds) {
    if (!count_.contains(name)) {
        count_.insert({name, 1});
    } else {
        count_[name]++;
    }
    if (!time_.contains(name)) {
        time_.insert({name, milliseconds});
    } else {
        time_[name] += milliseconds;
    }
}

std::string CodegenStatistics::summary() {
    if (count_.empty()) {
        return "";
    }

    auto compare_data_fn = [](const std::tuple<std::string, uint64_t, uint64_t>& a,
                              const std::tuple<std::string, uint64_t, uint64_t>& b) {
        auto [a_name, a_count, a_milliseconds] = a;
        auto [b_name, b_count, b_milliseconds] = b;
        return a_milliseconds > b_milliseconds || (a_milliseconds == b_milliseconds && a_count > b_count) ||
               (a_milliseconds == b_milliseconds && a_count == b_count && a_name < b_name);
    };
    std::stringstream stream;
    stream << "Codegen Statistics:" << std::endl;

    std::vector<std::tuple<std::string, uint64_t, uint64_t>> data;
    uint64_t time_sum = 0;
    for (auto [name, count] : count_) {
        if (time_.contains(name)) {
            auto milliseconds = time_[name];
            data.push_back({name, count, milliseconds});
            time_sum += milliseconds;
        }
    }
    std::sort(data.begin(), data.end(), compare_data_fn);

    if (!data.empty()) {
        stream << "  Codegen: " << time_sum << " ms" << std::endl;
        for (auto [name, count, milliseconds] : data) {
            stream << "    " << milliseconds << " ms  " << count << "  " << name << std::endl;
        }
    }

    return stream.str();
}

} // namespace passes
} // namespace sdfg
