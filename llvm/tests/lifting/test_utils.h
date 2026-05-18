#pragma once

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

// Utility function to parse an LLVM IR string into a Module that can be reused across tests.
inline std::unique_ptr<llvm::Module> loadModuleFromIR(const std::string &ir, llvm::LLVMContext &context) {
    // Wrap the IR string into a memory buffer that LLVM can consume.
    auto mem_buf = llvm::MemoryBuffer::getMemBuffer(ir, "<in-memory IR>", /*RequiresNullTerminator=*/false);

    llvm::SMDiagnostic err;
    // `parseIR` consumes the buffer reference and returns a Module on success.
    auto module = llvm::parseIR(mem_buf->getMemBufferRef(), err, context);

    // When parsing fails, emit the diagnostic to stderr so the failing test provides context.
    if (!module) {
        err.print("loadModuleFromIR", llvm::errs());
    }

    return module;
}
