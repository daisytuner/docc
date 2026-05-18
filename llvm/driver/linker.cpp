#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <list>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/bit.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Object/ELFTypes.h>
#include <llvm/Support/Error.h>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "utils.h"

const std::string BASE_LINKER = "ld.lld-19";

std::vector<std::filesystem::path> find_modules(const std::vector<std::string>& cmd) {
    const std::regex object_file("[a-z,A-Z,0-9,\\+,\\-,\\_,\\/,\\.]+.o");

    std::vector<std::filesystem::path> modules;
    for (auto& part : cmd) {
        if (std::regex_match(part, object_file)) {
            modules.push_back({part});
        }
    }

    return modules;
}

std::vector<std::filesystem::path> find_shared_libraries(const std::vector<std::string>& cmd) {
    const std::regex shared_object_file("[a-z,A-Z,0-9,\\+,\\-,\\_,\\/,\\.]+.so[0-9,\\.]*");
    std::vector<std::filesystem::path> libraries;

    std::unordered_set<std::string> library_paths_set, library_names;
    std::list<std::string> library_paths;
    for (size_t i = 0; i < cmd.size(); i++) {
        if (cmd[i] == "-o") {
            i++;
            continue;
        } else if (std::regex_match(cmd[i], shared_object_file)) {
            libraries.push_back({cmd[i]});
        } else if (cmd[i] == "-L") {
            i++;
            if (!library_paths_set.contains(cmd[i])) {
                library_paths_set.insert(cmd[i]);
                library_paths.push_back(cmd[i]);
            }
        } else if (cmd[i].starts_with("-L")) {
            const std::string path = cmd[i].substr(2);
            if (!library_paths_set.contains(path)) {
                library_paths_set.insert(path);
                library_paths.push_back(path);
            }
        } else if (cmd[i].starts_with("--library-path=")) {
            const std::string path = cmd[i].substr(15);
            if ((path.starts_with('"') && path.ends_with('"')) || (path.starts_with('\'') && path.ends_with('\''))) {
                const std::string extracted_path = path.substr(1, path.length() - 1);
                if (!library_paths_set.contains(extracted_path)) {
                    library_paths_set.insert(extracted_path);
                    library_paths.push_back(extracted_path);
                }
            } else {
                if (!library_paths_set.contains(path)) {
                    library_paths_set.insert(path);
                    library_paths.push_back(path);
                }
            }
        } else if (cmd[i] == "-l") {
            i++;
            library_names.insert(cmd[i]);
        } else if (cmd[i].starts_with("-l")) {
            library_names.insert(cmd[i].substr(2));
        } else if (cmd[i].starts_with("--library")) {
            const std::string name = cmd[i].substr(9);
            if ((name.starts_with('"') && name.ends_with('"')) || (name.starts_with('\'') && name.ends_with('\''))) {
                library_names.insert(name.substr(1, name.length() - 1));
            } else {
                library_names.insert(name);
            }
        }
    }

    auto ld_library_path = docc::split_env(docc::getEnv("LD_LIBRARY_PATH"), ':');
    for (auto& library_name : library_names) {
        auto shared_object_name = "lib" + library_name + ".so";
        std::string library_file = docc::find_file(library_paths, shared_object_name.c_str());
        if (!library_file.empty()) {
            libraries.push_back({library_file});
            continue;
        }
        library_file = docc::find_file(ld_library_path, shared_object_name.c_str());
        if (library_file.empty()) {
#ifndef NDEBUG
            std::cerr << "[linker] Could not find shared object file for library: " << library_name << std::endl;
#endif
        } else {
            libraries.push_back({library_file});
        }
    }

    return libraries;
}

template<typename llvm::endianness E, bool is64>
void dump_dynamic_symbols(std::ofstream& stream, const std::string& library, llvm::MemoryBufferRef mem_buffer_ref) {
    llvm::Expected<llvm::object::ELFObjectFile<llvm::object::ELFType<E, is64>>> elf_or_err =
        llvm::object::ELFObjectFile<llvm::object::ELFType<E, is64>>::create(mem_buffer_ref);
    if (!elf_or_err) {
#ifndef NDEBUG
        std::cerr << "[linker] Could not read ELF object file: " << library << std::endl;
#endif
        return;
    }
    llvm::object::ELFObjectFile<llvm::object::ELFType<E, is64>> elf = std::move(elf_or_err.get());

    const std::regex copy_in("\\_in\\_[0-9]+$");
    const std::regex copy_out("\\_out\\_[0-9]+$");
    const std::regex alloc("\\_alloc\\_[0-9]+$");
    const std::regex free("\\_free\\_[0-9]+$");

    std::unordered_set<std::string> symbols, symbols_kernel;
    std::unordered_map<std::string, std::unordered_set<size_t>> symbols_in, symbols_out, symbols_alloc, symbols_free;
    std::unordered_map<std::string, size_t> max_copy_symbol;
    for (llvm::object::SymbolRef symbol : elf.getDynamicSymbolIterators()) {
        llvm::Expected<llvm::StringRef> name_or_err = symbol.getName();
        if (!name_or_err) {
            continue;
        }
        const std::string name = name_or_err->str();
        if (name.ends_with("_kernel")) {
            symbols_kernel.insert(name);
        } else if (std::regex_search(name, copy_in)) {
            size_t in_pos = name.find("_in_");
            const std::string orig_name = name.substr(0, in_pos);
            const std::string num_str = name.substr(in_pos + 4);
            char* endptr;
            size_t num = std::strtoull(num_str.c_str(), &endptr, 10);
            if (!symbols_in.contains(orig_name)) {
                symbols_in.insert({orig_name, {}});
            }
            symbols_in[orig_name].insert(num);
            if (!max_copy_symbol.contains(orig_name)) {
                max_copy_symbol.insert({orig_name, num});
            } else if (num > max_copy_symbol[orig_name]) {
                max_copy_symbol[orig_name] = num;
            }
        } else if (std::regex_search(name, copy_out)) {
            size_t out_pos = name.find("_out_");
            const std::string orig_name = name.substr(0, out_pos);
            const std::string num_str = name.substr(out_pos + 5);
            char* endptr;
            size_t num = std::strtoull(num_str.c_str(), &endptr, 10);
            if (!symbols_out.contains(orig_name)) {
                symbols_out.insert({orig_name, {}});
            }
            symbols_out[orig_name].insert(num);
            if (!max_copy_symbol.contains(orig_name)) {
                max_copy_symbol.insert({orig_name, num});
            } else if (num > max_copy_symbol[orig_name]) {
                max_copy_symbol[orig_name] = num;
            }
        } else if (std::regex_search(name, alloc)) {
            size_t alloc_pos = name.find("_alloc_");
            const std::string orig_name = name.substr(0, alloc_pos);
            const std::string num_str = name.substr(alloc_pos + 7);
            char* endptr;
            size_t num = std::strtoull(num_str.c_str(), &endptr, 10);
            if (!symbols_alloc.contains(orig_name)) {
                symbols_alloc.insert({orig_name, {}});
            }
            symbols_alloc[orig_name].insert(num);
            if (!max_copy_symbol.contains(orig_name)) {
                max_copy_symbol.insert({orig_name, num});
            } else if (num > max_copy_symbol[orig_name]) {
                max_copy_symbol[orig_name] = num;
            }
        } else if (std::regex_search(name, free)) {
            size_t free_pos = name.find("_free_");
            const std::string orig_name = name.substr(0, free_pos);
            const std::string num_str = name.substr(free_pos + 6);
            char* endptr;
            size_t num = std::strtoull(num_str.c_str(), &endptr, 10);
            if (!symbols_free.contains(orig_name)) {
                symbols_free.insert({orig_name, {}});
            }
            symbols_free[orig_name].insert(num);
            if (!max_copy_symbol.contains(orig_name)) {
                max_copy_symbol.insert({orig_name, num});
            } else if (num > max_copy_symbol[orig_name]) {
                max_copy_symbol[orig_name] = num;
            }
        } else {
            symbols.insert(name);
        }
    }

    bool first = true;
    for (auto& symbol_kernel : symbols_kernel) {
        const std::string symbol = symbol_kernel.substr(0, symbol_kernel.length() - 7);
        if (!symbols.contains(symbol)) {
            continue;
        }
        if (first) {
            first = false;
        } else {
            stream << ",";
        }
        stream << "{";
        stream << "\"attributes\": {";
        stream << "\"arguments\": [";
        if (max_copy_symbol.contains(symbol)) {
            for (size_t i = 0; i <= max_copy_symbol[symbol]; i++) {
                if (i > 0) {
                    stream << ",";
                }
                bool copy_in = symbols_in.contains(symbol) && symbols_in[symbol].contains(i);
                bool copy_out = symbols_out.contains(symbol) && symbols_out[symbol].contains(i);
                bool alloc = symbols_alloc.contains(symbol) && symbols_alloc[symbol].contains(i);
                bool free = symbols_free.contains(symbol) && symbols_free[symbol].contains(i);
                std::string copy_buffer = "", copy_target = "";
                if (copy_in || copy_out || alloc || free) {
                    copy_buffer = "__daisy_inline_" + std::to_string(i);
                    copy_target = "EXTERNAL";
                }
                stream << "{";
                stream << "\"alloc\": " << std::boolalpha << alloc << ",";
                stream << "\"copy_buffer\": \"" << copy_buffer << "\",";
                stream << "\"copy_in\": " << std::boolalpha << copy_in << ",";
                stream << "\"copy_out\": " << std::boolalpha << copy_out << ",";
                stream << "\"copy_size\": null,";
                stream << "\"copy_target\": \"" << copy_target << "\",";
                stream << "\"free\": " << std::boolalpha << free;
                stream << "}";
            }
        }
        stream << "]";
        stream << "},";
        stream << "\"file\": \"" << library << "\",";
        stream << "\"name\": \"" << symbol << "\"";
        stream << "}";
    }
}

void dump_libraries(const std::string& dump_file, const std::vector<std::string>& cmd) {
    std::ofstream stream;
    stream.open(dump_file, std::ios_base::out);
    if (!stream.good()) {
#ifndef NDEBUG
        std::cerr << "[linker] Could not open library externals file: " << dump_file << std::endl;
#endif
        return;
    }

    stream << "{";
    stream << "\"sdfgs\": [";
    auto libraries = find_shared_libraries(cmd);
    for (auto& library : libraries) {
        llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>> bin_or_err =
            llvm::object::createBinary(library.generic_string());
        if (!bin_or_err) {
#ifndef NDEBUG
            std::cerr << "[linker] Could not read binary: " << library.generic_string() << std::endl;
#endif
            continue;
        }
        llvm::object::Binary& bin = *bin_or_err.get().getBinary();
        switch (bin.getType()) {
            case 13: // ID_ELF32L = ELF 32-bit, little endian
                dump_dynamic_symbols<
                    llvm::endianness::little,
                    true>(stream, library.generic_string(), bin.getMemoryBufferRef());
                break;
            case 14: // ID_ELF32B = ELF 32-bit, big endian
                dump_dynamic_symbols<
                    llvm::endianness::big,
                    false>(stream, library.generic_string(), bin.getMemoryBufferRef());
                break;
            case 15: // ID_ELF64L = ELF 64-bit, little endian
                dump_dynamic_symbols<
                    llvm::endianness::little,
                    true>(stream, library.generic_string(), bin.getMemoryBufferRef());
                break;
            case 16: // ID_ELF64B = ELF 64-bit, big endian
                dump_dynamic_symbols<
                    llvm::endianness::big,
                    true>(stream, library.generic_string(), bin.getMemoryBufferRef());
                break;
            default: // No ELF binary
                continue;
        }
    }
    stream << "]";
    stream << "}";
    stream.close();
}

std::string hash_module_name(const llvm::Module& Module) {
    std::string srcName = Module.getSourceFileName();
    std::string absPath = std::filesystem::absolute(srcName).string();

    std::hash<std::string> hash_fn;
    size_t hash_value = hash_fn(absPath);
    std::string stable_hash = std::to_string(hash_value);

    return stable_hash;
}

std::string get_extract_dir(const std::filesystem::path& Obj) {
    llvm::LLVMContext Ctx;

    // 1) try the fast path: IR in .llvmbc
    if (auto MB = llvm::MemoryBuffer::getFile(Obj.string(), /*IsText=*/false)) {
        if (llvm::identify_magic((*MB)->getBuffer()) == llvm::file_magic::bitcode) {
            if (auto MOrErr = llvm::getOwningLazyBitcodeModule(std::move(*MB), Ctx)) {
                std::unique_ptr<llvm::Module> Mod = std::move(*MOrErr);
                if (auto* S = llvm::dyn_cast_or_null<llvm::MDString>(Mod->getModuleFlag("docc.extract.dir"))) {
                    std::filesystem::path docc_extract_dir = S->getString().str();
                    std::filesystem::path module_dir = docc_extract_dir / hash_module_name(*Mod);
                    return module_dir.string();
                }
            }
        }
    }
    return "";
}

std::string get_output_file(const std::vector<std::string>& command) {
    auto it = std::find(command.begin(), command.end(), "-o");
    if (it == command.end()) return std::string{};
    ++it;
    assert(it != command.end() && "Ill-formed command-line");
    return *it;
}

int thinlto_pass(
    std::vector<std::string> cmd,
    const std::string& plugin_path,
    docc::DoccPaths& docc_paths,
    bool allow_for_relink,
    docc::DOCC_CI_LEVEL ci_level,
    bool isVerbose
) {
    // in order to support pass-parameters, we have to trigger LLVM argument-parsing explicitly with a special plugin
    // load
    auto first_custom_arg_it = std::ranges::find_if(cmd, [](const std::string& arg) {
        return arg.starts_with("-mllvm=");
    });

    cmd.insert(first_custom_arg_it, {"--load-pass-plugin=" + plugin_path, "-mllvm=-load=" + plugin_path});


    // --lto-O3 is required for the plugin to work
    cmd.push_back("--lto-O3");

    // Ignore unresolved symbols in object files, there will be a 2nd link pass that should resolve them
    // (link offloading-specific libraries, link additional cuda kernels etc.)
    if (allow_for_relink) {
        cmd.push_back("--unresolved-symbols=ignore-in-object-files");
    }

    // Save the link-time optimisation index and thin-lto results
    cmd.push_back("--save-temps");

    // Forward the index to the plugin
    std::string output_path = get_output_file(cmd);
    assert(!output_path.empty() && "Command-line has no -o argument");
    cmd.push_back("-mllvm=-docc-combined-index=" + output_path + ".index.bc");
    cmd.push_back("-mllvm=-docc-combined-lib-index=" + output_path + ".externals.json");

    if (ci_level != docc::DOCC_CI_LEVEL_NONE) {
        // -docc-instrument=ols
        if (ci_level == docc::DOCC_CI_LEVEL_FULL || ci_level == docc::DOCC_CI_LEVEL_REGIONS) {
            if (std::find_if(cmd.begin(), cmd.end(), [](const std::string& arg) {
                    return arg.starts_with("-mllvm=-docc-instrument");
                }) == cmd.end()) {
                cmd.push_back("-mllvm=-docc-instrument=ols");
            }
        }
        // -docc-capture-args
        if (ci_level == docc::DOCC_CI_LEVEL_FULL || ci_level == docc::DOCC_CI_LEVEL_ARG_CAPTURE) {
            if (std::find_if(cmd.begin(), cmd.end(), [](const std::string& arg) {
                    return arg == "-mllvm=-docc-capture-args";
                }) == cmd.end()) {
                cmd.push_back("-mllvm=-docc-capture-args");
            }
        }
    }

    auto cmd_str = docc::str_join(cmd, " ");

    // Execute the linker
    if (isVerbose) {
        std::cerr << " lto: " << cmd_str << std::endl;
    }

    int system_result = std::system(cmd_str.c_str());
    if (WIFEXITED(system_result)) {
        return WEXITSTATUS(system_result);
    } else {
        return system_result;
    }
}

int final_link_pass(
    std::vector<std::string> final_link_cmd_parts, docc::DoccPaths& docc_paths, bool isVerbose, bool* requiresSaveTemps
) {
    // remove the -mllvm=-docc-tune=cuda if present
    final_link_cmd_parts.erase(
        std::remove_if(
            final_link_cmd_parts.begin(),
            final_link_cmd_parts.end(),
            [](const std::string& arg) { return arg.starts_with("-mllvm=-docc"); }
        ),
        final_link_cmd_parts.end()
    );

    // replace modified modules by thinlto-generated modules
    auto modules = find_modules(final_link_cmd_parts);
    std::vector<std::string> modified_modules;
    std::set<std::string> opts;
    for (auto& module : modules) {
        std::string extract_dir = get_extract_dir(module);
        if (extract_dir.empty()) {
            continue;
        }
        std::filesystem::path cuda_dir = extract_dir;
        std::ifstream infile(cuda_dir / "LINK_OPTS");
        std::string line;
        while (std::getline(infile, line)) {
            if (line == "-ltt_metal") {
                *requiresSaveTemps = true;
            }
            opts.emplace(line);
        }

        modified_modules.push_back(module.string());
    }

    // remove the -mllvm=-docc-tune=cuda if present
    final_link_cmd_parts.erase(
        std::remove_if(
            final_link_cmd_parts.begin(),
            final_link_cmd_parts.end(),
            [](const std::string& arg) { return arg.starts_with("-mllvm=-docc"); }
        ),
        final_link_cmd_parts.end()
    );

    auto target_lib_paths = docc_paths.target_lib_paths();
    for (auto& libdir : target_lib_paths) {
        final_link_cmd_parts.push_back("-L'" + libdir.string() + "'");
    }

    // replace modified modules by temporary files
    for (auto& mod : modified_modules) {
        std::filesystem::path lto_temp_file = mod + ".5.precodegen.bc";

        // Replace the module by the LTO temp file
        final_link_cmd_parts
            .erase(std::remove(final_link_cmd_parts.begin(), final_link_cmd_parts.end(), mod), final_link_cmd_parts.end());
        final_link_cmd_parts.push_back(lto_temp_file.string());
    }

    // Add the CUDA objects
    final_link_cmd_parts.insert(final_link_cmd_parts.end(), opts.begin(), opts.end());

    auto final_link_cmd = docc::str_join(final_link_cmd_parts, " ");
    if (isVerbose) {
        std::cerr << " link-2: " << final_link_cmd << std::endl;
    }

    int system_result = std::system(final_link_cmd.c_str());
    if (WIFEXITED(system_result)) {
        return WEXITSTATUS(system_result);
    } else {
        return system_result;
    }
}

static bool is_docc_save_temps(const std::vector<std::string>& args) {
    for (auto& arg : args) {
        if (arg == "-mllvm=-docc-save-temps" || arg == "-docc-save-temps") return true;
    }
    return false;
}

static bool is_link_single(const std::vector<std::string>& args) {
    std::string flag1 = "-docc-link=";
    std::string flag2 = "-mllvm=-docc-link=";
    std::optional<std::string> mode;
    for (const auto& item : args) {
        if (item.starts_with(flag1)) {
            mode = item.substr(flag1.size());
            break;
        } else if (item.starts_with(flag2)) {
            mode = item.substr(flag2.size());
            break;
        }
    }

    if (mode.has_value()) {
        return mode.value() == "single";
    } else {
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::vector<std::string> cmd{BASE_LINKER};
    for (int i = 1; i < argc; ++i) {
        // Parse option file, i.e., @file
        if (argv[i][0] == '@') {
            std::ifstream infile(&argv[i][1]);
            std::string line;
            while (std::getline(infile, line)) {
                std::istringstream iss(line);
                std::string part;
                while (iss >> part) {
                    // Strip leading and trailing quotes ""
                    if (part.size() >= 2 && part.front() == '"' && part.back() == '"') {
                        part = part.substr(1, part.size() - 2);
                    }
                    cmd.emplace_back(part);
                }
            }
        } else {
            cmd.emplace_back(argv[i]);
        }
    }

    if (docc::is_version(cmd)) {
        std::cout << "docc version: " << DOCC_LLVM_VERSION << std::endl;
        return EXIT_SUCCESS;
    }
    if (docc::is_help(cmd)) {
        return docc::execvp_or_die(cmd);
    }
    auto isVerbose = docc::is_verbose(cmd);

    auto isDoccSaveTemps = is_docc_save_temps(cmd);
    std::string docc_work_dir_arg_name = "-mllvm=-docc-work-dir=";
    auto work_dir_arg_it = std::ranges::find_if(cmd, [docc_work_dir_arg_name](const std::string& arg) {
        return arg.starts_with(docc_work_dir_arg_name);
    });
    std::string work_dir;
    if (work_dir_arg_it != cmd.end()) {
        work_dir = work_dir_arg_it->substr(docc_work_dir_arg_name.size());
    }

    // need to find the plugin and other locations. Can discover them from this optional command line arg
    std::string docc_root_arg_name = "-mllvm=-docc-root=";
    auto docc_root_arg_it = std::ranges::find_if(cmd, [docc_root_arg_name](const std::string& arg) {
        return arg.starts_with(docc_root_arg_name);
    });

    docc::DoccPaths docc_paths;
    if (docc_root_arg_it == cmd.end()) { // if missing, look for paths based on where our current executable is located
        docc_paths = docc::find_docc_paths();
        cmd.push_back(docc_root_arg_name + docc_paths.docc_root_str());
    } else {
        docc_paths = docc::DoccPaths::from_root(docc_root_arg_it->substr(docc_root_arg_name.size()));
    }
    std::string plugin_path = docc_paths.plugin_path.string();

    bool is_single_link = false;
    if (is_link_single(cmd)) {
        is_single_link = true;
    } else if (docc::getEnv("DOCC_LINK") == "force-single") {
        is_single_link = true;
        std::cerr << "[linker] Forcing single-link mode" << std::endl;
    }

    std::string output_path = get_output_file(cmd);

    // Dump cross library inline symbols
    dump_libraries(output_path + ".externals.json", cmd);

    /*** Pass 1: Thin-LTO ***/
    int ret = thinlto_pass(cmd, plugin_path, docc_paths, !is_single_link, docc::DOCC_CI_LEVEL_NONE, isVerbose);
    if (ret != 0) {
        return ret;
    }

    docc::DOCC_CI_LEVEL ci_level = docc::ci_level();

    bool runtimeRequiresTemps = false;

    if (!is_single_link) {
        /*** Pass 2: Link with new modules generated by thinlto ***/

        // Remove the output file from pass 1
        std::filesystem::remove(output_path);

        ret = final_link_pass(cmd, docc_paths, isVerbose, &runtimeRequiresTemps);
        if (ret != 0) {
            return ret;
        }
    }

    /*** Second Execution: Link with instrumentation and produce second binary */

    // If we are not in a CI environment, we do not need to create an instrumented binary
    if (ci_level != docc::DOCC_CI_LEVEL_NONE) {
        std::filesystem::path instrumented_output_path = output_path;
        instrumented_output_path = instrumented_output_path.replace_filename(
            instrumented_output_path.stem().string() + ".instrumented" + instrumented_output_path.extension().string()
        );
        std::replace(cmd.begin(), cmd.end(), output_path, instrumented_output_path.string());

        // Copy libindex file to .instrumented.libindex.json
        std::filesystem::path libindex_path = output_path + ".externals.json";
        std::filesystem::path instrumented_libindex_path = instrumented_output_path.string() + ".externals.json";
        std::filesystem::
            copy_file(libindex_path, instrumented_libindex_path, std::filesystem::copy_options::overwrite_existing);

        ret = thinlto_pass(cmd, plugin_path, docc_paths, !is_single_link, ci_level, isVerbose);
        if (ret != 0) {
            return ret;
        }

        if (!is_single_link) {
            // Remove the output file from pass 1
            std::filesystem::remove(instrumented_output_path);

            // Second binary
            ret = final_link_pass(cmd, docc_paths, isVerbose, &runtimeRequiresTemps);
            if (ret != 0) {
                return ret;
            }
        }
    }

    if (ci_level == docc::DOCC_CI_LEVEL_NONE && !isDoccSaveTemps && !work_dir.empty()) {
        if (runtimeRequiresTemps) {
            std::cerr << "Not deleting work dir (" << work_dir << "), because it is needed by runtime!" << std::endl;
        } else {
            // cleanup temp files
            std::filesystem::remove_all(work_dir);
        }
    }

    return 0;
};
