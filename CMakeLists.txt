cmake_minimum_required(VERSION 3.0.0)
project(nla)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -O3")

include_directories("/home/user/programming/eigen")

add_executable(nla main.cpp)
add_executable(nla_mehurcek "nla_mehurcek/nla_mehurcek.cpp")
add_executable(jacobi "jacobi/jacobi_main.cpp")
add_executable(gauss_seidl "gauss_seidl/gauss_seidl_main.cpp" gauss_seidl/gauss_seidl.h)
add_executable(sor "sor/sor_main.cpp" sor/sor.h)
add_executable(nal1 "nal1/nal1.cpp" nal1/nal1.h)
add_executable(nal2 "nal2/nal2.cpp" nal2/nal2.h)
add_executable(nal3 "nal3/nal3.cpp" nal3/nal3.h)
add_executable(nal4 "nal4/nal4.cpp" nal4/nal4.h)


message(NOTICE ../${CMAKE_CURRENT_SOURCE_DIR})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#target_link_libraries(nla_mehurcek PUBLIC Matplot++::matplot)