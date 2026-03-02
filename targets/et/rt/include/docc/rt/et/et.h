#pragma once

#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>
#include <runtime/Types.h>

namespace docc::rt::et {

class EtRuntimeWrapper {
    ::rt::RuntimePtr runtime_;

public:
    static constexpr size_t DEFAULT_TRACE_BUFFER_SIZE = 4096UL * 2 * 1024UL;

    EtRuntimeWrapper(::rt::RuntimePtr runtime);
    static EtRuntimeWrapper& get_instance();

    ::rt::IRuntime& get_runtime() const;

    ::rt::DeviceId get_device(int idx = 0) const;

    ::rt::KernelId load_kernel_binary_blocking(::rt::StreamId stream, const std::string& path) const;

    std::byte* alloc_trace_buffer(::rt::DeviceId dev, size_t bufSize = DEFAULT_TRACE_BUFFER_SIZE) const;

    void dump_trace_outputs(
        ::rt::DeviceId dev, ::rt::StreamId stream, std::byte* devPtr, size_t bufSize = DEFAULT_TRACE_BUFFER_SIZE
    ) const;
};

} // namespace docc::rt::et
