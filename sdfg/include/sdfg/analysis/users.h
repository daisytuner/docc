#pragma once

// Backward-compatible aliases; the users analysis lives in sdfg/users/.
#include "sdfg/users/users.h"

namespace sdfg {
namespace analysis {

using users::ForUser;
using users::Use;
using users::User;
using users::Users;
using users::UsersView;

using users::MOVE;
using users::NOP;
using users::READ;
using users::VIEW;
using users::WRITE;

} // namespace analysis
} // namespace sdfg
