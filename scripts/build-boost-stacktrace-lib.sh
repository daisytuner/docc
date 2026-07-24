#!/bin/bash

set -e

curl -O https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-cmake.tar.xz
tar -xJf boost-1.90.0-cmake.tar.xz
cd boost-1.90.0

cmake -B build -DBOOST_STACKTRACE_ENABLE_BACKTRACE=ON -DBOOST_STACKTRACE_ENABLE_FROM_EXCEPTION=ON -DBOOST_STACKTRACE_ENABLE_NOOP=OFF -DBUILD_SHARED_LIBS=ON -G Ninja
cmake --build build -- libs/stacktrace/all

ls -al build/stage/lib
