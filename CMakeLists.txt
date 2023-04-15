cmake_minimum_required(VERSION 3.26)
project(nla)

set(CMAKE_CXX_STANDARD 23)
find_package(Matplot++ REQUIRED)

include_directories("/home/user/programming/eigen")
add_executable(nla main.cpp)
add_executable(nla_mehurcek "nla_mehurcek/nla_mehurcek.cpp")
add_executable(jacobi "jacobi/jacobi_main.cpp")
add_executable(gauss_seidl "gauss_seidl/gauss_seidl_main.cpp" gauss_seidl/gauss_seidl.h)
add_executable(sor "sor/sor_main.cpp" sor/sor.h)
#target_link_libraries(nla_mehurcek PUBLIC Matplot++::matplot)