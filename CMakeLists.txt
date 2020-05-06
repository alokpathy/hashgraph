cmake_minimum_required(VERSION 3.8) #language level CUDA support was added in CMake 3.8
set(PROJECT "HornetAlg")

project(${PROJECT} LANGUAGES CXX CUDA)

include(compiler-util/CMakeLists.txt)
# include(../compiler-util/CMakeLists.txt)

# add_subdirectory(../hornet build)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

# include_directories(../hornet/include)
# include_directories(include/rmm/detail)
# include_directories(include/rmm)
include_directories(include)
include_directories(test)
# include_directories(mem)
# include_directories(SingleHashGraph)
# include_directories(MultiHashGraph)

# include_directories(/home/ogreen/hash-graph/moderngpu-oded/src)
# include_directories(/home/ogreen/hash-graph/hornetsnest/externals/cnmem/include)
# include_directories(../externals/moderngpu-oded/src)
# include_directories(../externals/cnmem/include)
include_directories(externals/moderngpu-oded/src)
include_directories(externals/rmm/include)
# include_directories(externals/cnmem/include)


###################################################################################################
# - library targets -------------------------------------------------------------------------------

file(GLOB_RECURSE CU_SRCS src/*)
# file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/externals/xlib/src/*)

add_library(alg ${CU_SRCS})

###################################################################################################
# - add executables -------------------------------------------------------------------------------

# add_executable(sing-hash test/SingleHashGraphTest.cu mem/memory.cpp mem/memory_manager.cpp externals/cnmem/src/cnmem.cpp)
# add_executable(sing-hash test/SingleHashGraphTest.cu mem/memory_manager.cpp)
# add_executable(multi-hash test/MultiHashGraphTest.cu mem/memory.cpp mem/memory_manager.cpp externals/cnmem/src/cnmem.cpp)
add_executable(sing-hash test/SingleHashGraphTest.cu )
add_executable(multi-hash test/MultiHashGraphTest.cu )
# add_executable(multi-hash test/MultiHashGraphTest.cu )


# link_directories(externals/rmm)
# target_link_libraries(sing-hash  alg cuda rmm)
target_link_libraries(sing-hash  alg cuda -fopenmp )
# target_link_libraries(multi-hash alg cuda rmm -fopenmp )
target_link_libraries(multi-hash alg cuda -fopenmp )
