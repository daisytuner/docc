#include <algorithm>
#include <boost/program_options.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/cutouts/cutouts.h"
#include "sdfg/dcov/pack.h"
#include "sdfg/dcov/region_builder.h"
#include "sdfg/dcov/region_differ.h"
#include "sdfg/dcov/region_serializer.h"
#include "sdfg/dcov/trace_loader.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_sdfg.h"

namespace po = boost::program_options;
using namespace sdfg;

namespace {

/// Read an SDFG JSON file and build its static region model.
/// Throws std::runtime_error with a user-facing message on failure.
dcov::Module load_module(
    const std::filesystem::path& input_path, const std::vector<std::pair<std::string, std::string>>& build_config
) {
    std::ifstream in(input_path);
    if (!in.is_open()) throw std::runtime_error("cannot open input '" + input_path.string() + "'");

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to parse JSON '" + input_path.string() + "': " + e.what());
    }

    std::unique_ptr<StructuredSDFG> sdfg;
    try {
        serializer::JSONSerializer json_serializer;
        sdfg = json_serializer.deserialize(j);
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to deserialize SDFG '" + input_path.string() + "': " + e.what());
    }

    dcov::RegionBuilder builder;
    return builder.build(*sdfg, build_config);
}

/// Read a whole file into a byte vector.
std::vector<uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open '" + path.string() + "'");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// Build the static region model AND retain the deserialized SDFG (for cutouts).
dcov::Module load_module_with_sdfg(const std::filesystem::path& input_path, std::unique_ptr<StructuredSDFG>& out_sdfg) {
    std::ifstream in(input_path);
    if (!in.is_open()) throw std::runtime_error("cannot open input '" + input_path.string() + "'");

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to parse JSON '" + input_path.string() + "': " + e.what());
    }

    try {
        serializer::JSONSerializer json_serializer;
        out_sdfg = json_serializer.deserialize(j);
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to deserialize SDFG '" + input_path.string() + "': " + e.what());
    }

    dcov::RegionBuilder builder;
    return builder.build(*out_sdfg, {});
}

/// Create a cutout SDFG (serialized to JSON) for each region with a control-flow
/// element id. Regions whose element is not a control-flow node (e.g. library
/// nodes) or whose cutout fails are skipped and counted in @p skipped.
std::vector<dcov::CutoutInput> make_cutouts(StructuredSDFG& sdfg, const dcov::Module& module, size_t& skipped) {
    std::vector<dcov::CutoutInput> out;
    skipped = 0;
    builder::StructuredSDFGBuilder sb(sdfg);
    analysis::AnalysisManager am(sb.subject());
    serializer::JSONSerializer js;
    for (const auto& r : module.regions) {
        if (r.element_id == 0) continue;
        auto* el = sb.find_element_by_id(r.element_id);
        auto* node = dynamic_cast<structured_control_flow::ControlFlowNode*>(el);
        if (node == nullptr) {
            ++skipped;
            continue;
        }
        try {
            auto cut = util::cutout(sb.subject(), am, *node);
            std::string text = js.serialize(*cut).dump();
            dcov::CutoutInput ci;
            ci.region_key = r.region_key;
            ci.display_key = r.display_key;
            ci.element_type = r.element_type;
            ci.element_id = r.element_id;
            ci.format = "json";
            ci.bytes.assign(text.begin(), text.end());
            out.push_back(std::move(ci));
        } catch (const std::exception&) {
            ++skipped;
        }
    }
    return out;
}

void write_output(const std::filesystem::path& output_path, const std::string& text) {
    if (output_path.empty()) {
        std::cout << text;
        return;
    }
    std::ofstream out(output_path);
    if (!out.is_open()) throw std::runtime_error("cannot open output '" + output_path.string() + "'");
    out << text;
}

int run_emit(int argc, char* argv[]) {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    std::string format;
    std::string target;
    std::string frontend;
    std::filesystem::path trace_path;
    std::filesystem::path arg_capture_dir;

    po::options_description desc("dcov - emit a static region model from an SDFG");
    desc.add_options()
        ("help,h", "produce help message")
        ("input,i", po::value<std::filesystem::path>(&input_path), "path to SDFG JSON (e.g. py4.norm.json)")
        ("output,o", po::value<std::filesystem::path>(&output_path)->default_value(""), "output file (default: stdout)")
        ("format,f", po::value<std::string>(&format)->default_value("dcov"), "output format: dcov | json")
        ("target", po::value<std::string>(&target)->default_value(""), "build config label: target device")
        ("frontend", po::value<std::string>(&frontend)->default_value(""), "build config label: frontend")
        ("trace", po::value<std::filesystem::path>(&trace_path)->default_value(""), "daisy_trace.json: annotate regions with runtime + metrics")
        ("arg-captures", po::value<std::filesystem::path>(&arg_capture_dir)->default_value(""), "arg-capture dir (default: read from --trace)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "dcov: " << e.what() << "\n";
        return 2;
    }

    if (vm.count("help") || input_path.empty()) {
        std::cout << desc << "\n";
        return vm.count("help") ? 0 : 2;
    }

    if (format != "dcov" && format != "json") {
        std::cerr << "dcov: unknown format '" << format << "' (expected 'dcov' or 'json')\n";
        return 2;
    }

    std::vector<std::pair<std::string, std::string>> build_config;
    if (!target.empty()) build_config.emplace_back("target", target);
    if (!frontend.empty()) build_config.emplace_back("frontend", frontend);

    dcov::Module module;
    try {
        module = load_module(input_path, build_config);
    } catch (const std::exception& e) {
        std::cerr << "dcov: " << e.what() << "\n";
        return 1;
    }

    if (!trace_path.empty()) {
        try {
            dcov::annotate_with_trace(module, trace_path);
            if (arg_capture_dir.empty()) arg_capture_dir = dcov::arg_capture_path_from_trace(trace_path);
        } catch (const std::exception& e) {
            std::cerr << "dcov: " << e.what() << "\n";
            return 1;
        }
    }
    if (!arg_capture_dir.empty()) dcov::annotate_with_arg_captures(module, arg_capture_dir);

    std::unique_ptr<dcov::RegionSerializer> region_serializer;
    if (format == "json")
        region_serializer = std::make_unique<dcov::DcovJsonSerializer>();
    else
        region_serializer = std::make_unique<dcov::DcovRecordSerializer>();

    try {
        std::ostringstream buf;
        region_serializer->serialize(module, buf);
        write_output(output_path, buf.str());
    } catch (const std::exception& e) {
        std::cerr << "dcov: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

int run_diff(int argc, char* argv[]) {
    std::filesystem::path path_a;
    std::filesystem::path path_b;
    std::filesystem::path output_path;
    std::string format;

    po::options_description desc("dcov diff - semantic diff between two SDFG region models");
    desc.add_options()
        ("help,h", "produce help message")
        ("a", po::value<std::filesystem::path>(&path_a), "path to baseline SDFG JSON (A)")
        ("b", po::value<std::filesystem::path>(&path_b), "path to changed SDFG JSON (B)")
        ("output,o", po::value<std::filesystem::path>(&output_path)->default_value(""), "output file (default: stdout)")
        ("format,f", po::value<std::string>(&format)->default_value("text"), "output format: text | json");

    po::positional_options_description pos;
    pos.add("a", 1).add("b", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "dcov diff: " << e.what() << "\n";
        return 2;
    }

    if (vm.count("help") || path_a.empty() || path_b.empty()) {
        std::cout << "usage: dcov diff <A.json> <B.json> [-f text|json] [-o out]\n\n" << desc << "\n";
        return vm.count("help") ? 0 : 2;
    }

    if (format != "text" && format != "json") {
        std::cerr << "dcov diff: unknown format '" << format << "' (expected 'text' or 'json')\n";
        return 2;
    }

    dcov::Module a, b;
    try {
        a = load_module(path_a, {});
        b = load_module(path_b, {});
    } catch (const std::exception& e) {
        std::cerr << "dcov diff: " << e.what() << "\n";
        return 1;
    }

    dcov::RegionDiffer differ;
    dcov::ModuleDiff diff = differ.diff(a, b);

    std::ostringstream buf;
    if (format == "json")
        dcov::DiffJsonSerializer().serialize(diff, buf);
    else
        dcov::DiffReportSerializer().serialize(diff, buf);

    try {
        write_output(output_path, buf.str());
    } catch (const std::exception& e) {
        std::cerr << "dcov diff: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

int run_pack(int argc, char* argv[]) {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    std::filesystem::path trace_path;
    std::filesystem::path arg_capture_dir;

    po::options_description desc("dcov pack - bundle SDFG + region model + metrics + arg captures into a .dcovpack");
    desc.add_options()
        ("help,h", "produce help message")
        ("input,i", po::value<std::filesystem::path>(&input_path), "path to final SDFG JSON (the runtime matcher, e.g. py5.post_sched.json)")
        ("output,o", po::value<std::filesystem::path>(&output_path), "output .dcovpack file")
        ("trace", po::value<std::filesystem::path>(&trace_path)->default_value(""), "daisy_trace.json (runtime + metrics)")
        ("arg-captures", po::value<std::filesystem::path>(&arg_capture_dir)->default_value(""), "arg-capture dir (default: read from --trace)")
        ("no-cutouts", po::bool_switch(), "do not generate per-region cutout SDFGs");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "dcov pack: " << e.what() << "\n";
        return 2;
    }

    if (vm.count("help") || input_path.empty() || output_path.empty()) {
        std::cout
            << "usage: dcov pack -i <final.json> [--trace t] [--arg-captures dir] [--no-cutouts] -o out.dcovpack\n\n"
            << desc << "\n";
        return vm.count("help") ? 0 : 2;
    }

    std::unique_ptr<StructuredSDFG> sdfg;
    dcov::Module module;
    std::vector<uint8_t> sdfg_bytes;
    try {
        module = load_module_with_sdfg(input_path, sdfg);
        sdfg_bytes = read_file_bytes(input_path);
    } catch (const std::exception& e) {
        std::cerr << "dcov pack: " << e.what() << "\n";
        return 1;
    }

    dcov::PackMeta meta;
    meta.tool_version = "dcov";
    meta.kernel = module.name;
    meta.module_id = module.module_id;
    meta.sdfg_file = input_path.string();
    meta.trace_file = trace_path.string();
    meta.sdfg_format = "json";

    try {
        if (!trace_path.empty()) {
            dcov::annotate_with_trace(module, trace_path);
            if (arg_capture_dir.empty()) arg_capture_dir = dcov::arg_capture_path_from_trace(trace_path);
        }
        if (!arg_capture_dir.empty()) dcov::annotate_with_arg_captures(module, arg_capture_dir);

        std::vector<dcov::CutoutInput> cutouts;
        if (!vm["no-cutouts"].as<bool>()) {
            size_t skipped = 0;
            cutouts = make_cutouts(*sdfg, module, skipped);
            std::cerr << "dcov: generated " << cutouts.size() << " cutouts (" << skipped << " skipped)\n";
        }

        dcov::write_pack(module, sdfg_bytes, cutouts, arg_capture_dir, meta, output_path);
    } catch (const std::exception& e) {
        std::cerr << "dcov pack: " << e.what() << "\n";
        return 1;
    }

    std::cerr << "dcov: wrote " << output_path.string() << "\n";
    return 0;
}

int run_inspect(int argc, char* argv[]) {
    std::filesystem::path pack_path;

    po::options_description desc("dcov inspect - summarize a .dcovpack");
    desc.add_options()("help,h", "produce help message")(
        "pack", po::value<std::filesystem::path>(&pack_path), "path to .dcovpack"
    );
    po::positional_options_description pos;
    pos.add("pack", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "dcov inspect: " << e.what() << "\n";
        return 2;
    }
    if (vm.count("help") || pack_path.empty()) {
        std::cout << "usage: dcov inspect <pack.dcovpack>\n\n" << desc << "\n";
        return vm.count("help") ? 0 : 2;
    }

    dcov::Pack pack;
    try {
        pack = dcov::read_pack(pack_path);
    } catch (const std::exception& e) {
        std::cerr << "dcov inspect: " << e.what() << "\n";
        return 1;
    }

    nlohmann::json model;
    try {
        model = nlohmann::json::parse(pack.region_model_json);
    } catch (const std::exception& e) {
        std::cerr << "dcov inspect: bad region model: " << e.what() << "\n";
        return 1;
    }

    const auto& regions = model.value("regions", nlohmann::json::array());
    size_t profiled = 0, captured = 0;
    std::vector<std::pair<double, std::string>> hot;
    for (const auto& r : regions) {
        if (r.value("has_arg_capture", false)) ++captured;
        if (!r.contains("profile") || r["profile"].is_null()) continue;
        ++profiled;
        const double rt = r["profile"].value("runtime_us", 0.0);
        std::string label = r.value("element_type", std::string());
        if (!r.value("op_class", std::string()).empty()) label += " " + r.value("op_class", std::string());
        label += " [eid=" + std::to_string(r.value("element_id", static_cast<size_t>(0))) + "]";
        hot.emplace_back(rt, label);
    }
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "PACK " << pack_path.filename().string() << "\n";
    std::cout << "  tool=" << pack.meta.tool_version << "  created=" << pack.meta.created_utc << "\n";
    std::cout << "  kernel=" << pack.meta.kernel << "  module_id=" << pack.meta.module_id << "\n";
    std::cout << "  sdfg=" << pack.meta.sdfg_file << "\n";
    std::cout << "  trace=" << (pack.meta.trace_file.empty() ? "-" : pack.meta.trace_file) << "\n";
    std::cout << "  regions=" << regions.size() << "  profiled=" << profiled << "  arg_captured=" << captured << "\n";
    std::cout << "  captures=" << pack.captures.size() << "  blob_bytes=" << pack.blob_store.size() << "\n";
    std::cout << "  sdfg_format=" << pack.meta.sdfg_format << "  sdfg_bytes=" << pack.sdfg_bytes.size() << "\n";
    std::cout << "  cutouts=" << pack.cutouts.size() << "  cutout_bytes=" << pack.cutout_store.size() << "\n";

    const size_t topn = std::min<size_t>(hot.size(), 8);
    if (topn) {
        std::cout << "  hottest regions:\n";
        for (size_t i = 0; i < topn; ++i)
            std::cout << "    " << std::fixed << std::setprecision(3) << hot[i].first << "us  " << hot[i].second
                      << "\n";
    }
    return 0;
}

int run_unpack(int argc, char* argv[]) {
    std::filesystem::path pack_path;
    std::filesystem::path out_dir;

    po::options_description desc("dcov unpack - restore region model + arg captures from a .dcovpack");
    desc.add_options()
        ("help,h", "produce help message")
        ("pack", po::value<std::filesystem::path>(&pack_path), "path to .dcovpack")
        ("output,o", po::value<std::filesystem::path>(&out_dir), "output directory");
    po::positional_options_description pos;
    pos.add("pack", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "dcov unpack: " << e.what() << "\n";
        return 2;
    }
    if (vm.count("help") || pack_path.empty() || out_dir.empty()) {
        std::cout << "usage: dcov unpack <pack.dcovpack> -o <dir>\n\n" << desc << "\n";
        return vm.count("help") ? 0 : 2;
    }

    dcov::Pack pack;
    try {
        pack = dcov::read_pack(pack_path);
    } catch (const std::exception& e) {
        std::cerr << "dcov unpack: " << e.what() << "\n";
        return 1;
    }

    try {
        std::filesystem::create_directories(out_dir);

        std::ofstream rm(out_dir / "region_model.json");
        rm << pack.region_model_json;

        // Embedded SDFG (extension follows the recorded format).
        const std::string sdfg_ext = pack.meta.sdfg_format.empty() ? "bin" : pack.meta.sdfg_format;
        if (!pack.sdfg_bytes.empty()) {
            std::ofstream sf(out_dir / ("sdfg." + sdfg_ext), std::ios::binary);
            sf.write(
                reinterpret_cast<const char*>(pack.sdfg_bytes.data()),
                static_cast<std::streamsize>(pack.sdfg_bytes.size())
            );
        }

        // Per-region cutout SDFGs.
        if (!pack.cutouts.empty()) {
            const std::filesystem::path cut_dir = out_dir / "cutouts";
            std::filesystem::create_directories(cut_dir);
            for (const auto& c : pack.cutouts) {
                if (c.offset + c.size > pack.cutout_store.size()) throw std::runtime_error("cutout blob out of range");
                const std::string ext = c.format.empty() ? "bin" : c.format;
                std::ofstream cf(cut_dir / ("region_" + std::to_string(c.element_id) + "." + ext), std::ios::binary);
                cf.write(
                    reinterpret_cast<const char*>(pack.cutout_store.data() + c.offset),
                    static_cast<std::streamsize>(c.size)
                );
            }
        }

        const std::filesystem::path caps_dir = out_dir / "arg_captures";
        std::filesystem::create_directories(caps_dir);

        std::map<size_t, nlohmann::json> indexes;
        for (const auto& c : pack.captures) {
            if (c.offset + c.size > pack.blob_store.size()) throw std::runtime_error("capture blob out of range");

            if (!c.ext_file.empty()) {
                std::ofstream bin(caps_dir / c.ext_file, std::ios::binary);
                bin.write(
                    reinterpret_cast<const char*>(pack.blob_store.data() + c.offset),
                    static_cast<std::streamsize>(c.size)
                );
            }

            auto& idx = indexes[c.element_id];
            if (idx.is_null()) {
                idx["element_id"] = std::to_string(c.element_id);
                idx["invocation"] = c.invocation;
                idx["target"] = c.target;
                idx["format"] = c.format;
                idx["captures"] = nlohmann::json::array();
            }
            idx["captures"].push_back({
                {"after", c.after},
                {"arg_idx", c.arg_idx},
                {"dims", c.dims},
                {"ext_file", c.ext_file},
                {"primitive_type", c.primitive_type},
            });
        }

        std::string stem = pack.meta.kernel.empty() ? "kernel" : pack.meta.kernel;
        for (const auto& [eid, idx] : indexes) {
            std::ofstream
                ij(caps_dir / (stem + "_inv" + std::to_string(idx.value("invocation", 0)) + "_" + std::to_string(eid) +
                               ".index.json"));
            ij << idx.dump();
        }
    } catch (const std::exception& e) {
        std::cerr << "dcov unpack: " << e.what() << "\n";
        return 1;
    }

    std::cerr << "dcov: unpacked to " << out_dir.string() << " (" << pack.captures.size() << " captures, "
              << pack.cutouts.size() << " cutouts)\n";
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    // Register library-node (de)serializers so deserialize() can rebuild lib nodes.
    serializer::register_default_serializers();

    // Subcommand dispatch.
    if (argc >= 2) {
        const std::string cmd = argv[1];
        if (cmd == "diff") return run_diff(argc - 1, argv + 1);
        if (cmd == "pack") return run_pack(argc - 1, argv + 1);
        if (cmd == "inspect") return run_inspect(argc - 1, argv + 1);
        if (cmd == "unpack") return run_unpack(argc - 1, argv + 1);
    }

    return run_emit(argc, argv);
}
