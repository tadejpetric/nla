cmake_minimum_required(VERSION 3.26)
project(nla)

set(CMAKE_CXX_STANDARD 23)
find_package(Matplot++ REQUIRED)

include_directories("/home/tadej/programming/eigen")
add_executable(nla main.cpp nla_mehurcek/nla_mehurcek.cpp)
#target_link_libraries(nla_mehurcek PUBLIC Matplot++::matplot)