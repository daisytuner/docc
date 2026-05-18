#pragma once

#include <sdfg/serializer/json_serializer.h>

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace docc {
namespace analysis {

struct ArgumentAttributes {
    std::string copy_target;
    std::string copy_buffer;
    sdfg::symbolic::Expression copy_size;
    bool copy_in;
    bool copy_out;
    bool alloc;
    bool free;
};

struct Attributes {
    std::vector<ArgumentAttributes> arguments;

    static Attributes from_json(const json& j) {
        Attributes attrs;
        if (!j.is_null()) {
            for (const auto& arg_attrs : j["arguments"]) {
                ArgumentAttributes argument;
                argument.copy_target = arg_attrs.value("copy_target", "");
                argument.copy_buffer = arg_attrs.value("copy_buffer", "");
                argument.copy_in = arg_attrs.value("copy_in", false);
                argument.copy_out = arg_attrs.value("copy_out", false);
                argument.alloc = arg_attrs.value("alloc", false);
                argument.free = arg_attrs.value("free", false);
                argument.copy_size = SymEngine::null;
                if (arg_attrs["copy_size"].is_string() && !arg_attrs["copy_size"].get<std::string>().empty()) {
                    argument.copy_size = sdfg::symbolic::parse(arg_attrs["copy_size"].get<std::string>());
                }
                attrs.arguments.push_back(argument);
            }
        }
        return attrs;
    }

    json to_json() const {
        json j;
        j["arguments"] = json::array();
        for (const auto& arg_attrs : arguments) {
            std::string copy_size_str;
            if (!arg_attrs.copy_size.is_null()) {
                sdfg::serializer::JSONSymbolicPrinter printer;
                copy_size_str = printer.apply(arg_attrs.copy_size);
            }

            nlohmann::json attr_json;
            attr_json["copy_target"] = arg_attrs.copy_target;
            attr_json["copy_buffer"] = arg_attrs.copy_buffer;
            attr_json["copy_in"] = arg_attrs.copy_in;
            attr_json["copy_out"] = arg_attrs.copy_out;
            attr_json["alloc"] = arg_attrs.alloc;
            attr_json["free"] = arg_attrs.free;
            attr_json["copy_size"] = copy_size_str;

            j["arguments"].push_back(attr_json);
        }
        return j;
    }
};

} // namespace analysis
} // namespace docc
