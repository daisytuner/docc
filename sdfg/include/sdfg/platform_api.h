#pragma once

#include <string>
#include <vector>

namespace sdfg {

class PlatformAPI {
public:
    virtual ~PlatformAPI() = default;
    virtual void some_function() = 0;
};

} // namespace sdfg
