#include "sdfg/targets/memory/plugin.h"
#include <iostream>

void sdfg::memory::Plugin::some_function() {}

void sdfg::memory::Plugin::say_hello() { std::cout << "Hello from memory plugin" << std::endl; }
