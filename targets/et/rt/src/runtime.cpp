#include "docc/rt/et/et.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>
#include <runtime/Types.h>
#include <sw-sysemu/SysEmuOptions.h>

#include "docc/rt/et/et-sysemu-config.h"

#define DOCC_RT_ET_MODE_RUNTIME 0
#define DOCC_RT_ET_MODE_DEVICE 1
#define DOCC_RT_ET_MODE_SYSEMU 2

#ifndef DOCC_RT_ET_MODE
#define DOCC_RT_ET_MODE DOCC_RT_ET_MODE_RUNTIME
#endif

// Pull in the trace decoder implementation
#define ET_TRACE_DECODER_IMPL
#include <et-trace/decoder.h>
#include <et-trace/layout.h>

namespace docc::rt::et {

enum class Mode { RuntimeSelected = 0, Device = 1, SysEmu = 2 };

EtRuntimeWrapper::EtRuntimeWrapper(::rt::RuntimePtr runtime) : runtime_(std::move(runtime)) {}

static Mode get_mode() {
#if DOCC_RT_ET_MODE == DOCC_RT_ET_MODE_DEVICE
    return Mode::Device;
#elif DOCC_RT_ET_MODE == DOCC_RT_ET_MODE_SYSEMU
    return Mode::SysEmu;
#else
    auto* selector = std::getenv("DOCC_RT_ET_MODE");
    if (selector == nullptr) {
        return Mode::SysEmu;
    } else if (std::strcmp(selector, "device") == 0) {
        return Mode::Device;
    } else if (std::strcmp(selector, "sys_emu") == 0) {
        return Mode::SysEmu;
    } else {
        std::cerr << "Invalid DOCC_RT_ET_MODE: " << selector << " using SysEmu" << std::endl;
        return Mode::SysEmu;
    }
#endif
}

EtRuntimeWrapper& EtRuntimeWrapper::get_instance() {
    std::shared_ptr<dev::IDeviceLayer> deviceLayer;

    const Mode mode = get_mode();

    if (mode == Mode::Device) {
        deviceLayer = dev::IDeviceLayer::createPcieDeviceLayer(true, false);
    } else if (mode == Mode::SysEmu) {
        // Create SysEmu device (not everyone has a physical card)
        emu::SysEmuOptions sysEmuOptions;
        if (std::filesystem::exists(BOOTROM_TRAMPOLINE_TO_BL2_ELF)) {
            sysEmuOptions.bootromTrampolineToBL2ElfPath = BOOTROM_TRAMPOLINE_TO_BL2_ELF;
        }
        if (std::filesystem::exists(BL2_ELF)) {
            sysEmuOptions.spBL2ElfPath = BL2_ELF;
        }
        if (std::filesystem::exists(MACHINE_MINION_ELF)) {
            sysEmuOptions.machineMinionElfPath = MACHINE_MINION_ELF;
        }
        if (std::filesystem::exists(MASTER_MINION_ELF)) {
            sysEmuOptions.masterMinionElfPath = MASTER_MINION_ELF;
        }
        if (std::filesystem::exists(WORKER_MINION_ELF)) {
            sysEmuOptions.workerMinionElfPath = WORKER_MINION_ELF;
        }
        auto current_path = std::filesystem::current_path();
        sysEmuOptions.executablePath = std::filesystem::path(SYSEMU_INSTALL_DIR) / "sys_emu";
        sysEmuOptions.runDir = current_path;
        sysEmuOptions.maxCycles = std::numeric_limits<uint64_t>::max();
        sysEmuOptions.minionShiresMask = 0x1FFFFFFFFu;
        sysEmuOptions.puUart0Path = current_path / "pu_uart0_tx.log";
        sysEmuOptions.puUart1Path = current_path / "pu_uart1_tx.log";
        sysEmuOptions.spUart0Path = current_path / "spio_uart0_tx.log";
        sysEmuOptions.spUart1Path = current_path / "spio_uart1_tx.log";
        sysEmuOptions.startGdb = false;

        deviceLayer = dev::IDeviceLayer::createSysEmuDeviceLayer(sysEmuOptions, 1);
    } else {
        throw std::runtime_error("Invalid mode selected");
    }

    // Create runtime, get device, create stream
    auto runtime = ::rt::IRuntime::create(deviceLayer);

    static std::unique_ptr<EtRuntimeWrapper> instance_ = std::make_unique<EtRuntimeWrapper>(std::move(runtime));
    return *instance_;
}

::rt::DeviceId EtRuntimeWrapper::get_device(int idx) const {
    auto devs = runtime_->getDevices();
    if (devs.size() <= idx) {
        throw std::runtime_error(
            "Invalid device index " + std::to_string(idx) + ", " + std::to_string(devs.size()) + " devices available"
        );
    }
    return devs[idx];
}

::rt::IRuntime& EtRuntimeWrapper::get_runtime() const { return *runtime_; }

static std::vector<std::byte> readFile(const std::string& path) {
    std::ifstream file(path, std::ios_base::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
        return {};
    }
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> content(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(content.data()), size);
    return content;
}

::rt::KernelId EtRuntimeWrapper::load_kernel_binary_blocking(::rt::StreamId stream, const std::string& path) const {
    auto content = readFile(path);
    auto resp = runtime_->loadCode(stream, content.data(), content.size());
    runtime_->waitForEvent(resp.event_);
    return resp.kernel_;
}

static constexpr size_t kTraceBufferSize = 4096UL * 2048UL; // 8 MB. Way more then enough

std::byte* EtRuntimeWrapper::alloc_trace_buffer(::rt::DeviceId dev, size_t bufSize) const {
    auto* traceDevBuf = runtime_->mallocDevice(dev, bufSize);
    return traceDevBuf;
}

void EtRuntimeWrapper::dump_trace_outputs(::rt::DeviceId dev, ::rt::StreamId stream, std::byte* devPtr, size_t bufSize)
    const {
    std::vector<std::byte> hostTraceBuf(bufSize);
    runtime_->memcpyDeviceToHost(stream, devPtr, hostTraceBuf.data(), bufSize);
    runtime_->waitForStream(stream);

    // Decode and print kernel prints
    auto* traceHeader = reinterpret_cast<const trace_buffer_std_header_t*>(hostTraceBuf.data());
    const trace_entry_header_t* entry = nullptr;
    int count = 0;

    while ((entry = Trace_Decode(traceHeader, entry))) {
        if (entry->type != TRACE_TYPE_STRING) {
            continue;
        }
        auto* strEntry = reinterpret_cast<const trace_string_t*>(entry);
        std::cerr << "[hart " << entry->hart_id << "] " << strEntry->string << "\n";
        ++count;
    }

    std::cerr << "Decoded " << count << " trace string entries.\n";

    // Cleanup
    runtime_->freeDevice(dev, devPtr);
}

} // namespace docc::rt::et
