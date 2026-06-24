#include "sdfg/context/docc_context.h"

namespace docc::context {

using namespace docc::target;

bool DoccContext::add_target(docc::target::DoccTarget* target) {
    auto res = available_targets.insert_or_assign(target->short_name, target);
    return res.second;
}

docc::target::DoccTarget* DoccContext::get_target_handler(const std::string& target) const {
    auto it = available_targets.find(target);
    if (it != available_targets.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace docc::context
