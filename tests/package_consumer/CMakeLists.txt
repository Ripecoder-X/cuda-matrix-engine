cmake_minimum_required(VERSION 3.24)

project(cme_package_consumer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cme 0.1 REQUIRED CONFIG)

add_executable(cme_package_consumer main.cpp)
target_link_libraries(cme_package_consumer PRIVATE cme::cme)
