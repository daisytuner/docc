#pragma once

#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>

#include <string>
#include <utility>
#include <vector>

#include <sdfg/builder/sdfg_builder.h>

#include "docc/lifting/exceptions.h"
#include "docc/utils.h"

namespace docc {
namespace lifting {
namespace utils {

/**** New API ****/

std::string get_name(const llvm::Value* value);

std::string normalize_name(const std::string value);

bool is_literal(const llvm::Value* value);

bool is_null_pointer(const llvm::Value* value);

bool is_symbol(const llvm::Value* value);

std::string as_literal(const llvm::ConstantData* value);

sdfg::symbolic::Expression as_symbol(const llvm::Constant* value);

std::string get_initializer(const llvm::GlobalVariable& global);

bool has_initializer(const llvm::GlobalVariable& global);

std::string as_initializer(const llvm::Constant* initializer);

std::string create_const_name_to_sdfg_name(
    sdfg::builder::SDFGBuilder& builder,
    std::unordered_map<const llvm::Value*, std::string>& constants_mapping,
    const llvm::Value* llvm_ptr
);

std::string find_const_name_to_sdfg_name(
    std::unordered_map<const llvm::Value*, std::string>& constants_mapping, const llvm::Value* llvm_ptr
);

std::unique_ptr<sdfg::types::IType> get_type(
    sdfg::builder::SDFGBuilder& builder,
    std::unordered_map<const llvm::Type*, std::string>& anonymous_types_mapping,
    const llvm::DataLayout& DL,
    llvm::Type* type,
    sdfg::types::StorageType storage_type,
    const std::string& initializer = ""
);

sdfg::types::StorageType get_storage_type(sdfg::FunctionType target_type, size_t address_space);

std::pair<std::string, sdfg::data_flow::Subset>
get_memlet(std::unordered_map<const llvm::Value*, std::string>& constants_mapping, const llvm::GEPOperator* pointer);

/**** Deprecated API ****/

std::vector<llvm::GlobalVariable*> add_nested_global(const llvm::ConstantExpr* expr);

} // namespace utils
} // namespace lifting
} // namespace docc
