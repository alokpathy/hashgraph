set(CMAKE_CUDA_COMPILER "/sw/summit/cuda/10.1.243/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/sw/summit/gcc/6.4.0/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "10.1.243")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "")
set(CMAKE_CUDA20_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "6.4")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/sw/summit/cuda/10.1.243")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/sw/summit/cuda/10.1.243")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/sw/summit/cuda/10.1.243/targets/ppc64le-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/sw/summit/cuda/10.1.243/targets/ppc64le-linux/lib/stubs;/sw/summit/cuda/10.1.243/targets/ppc64le-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-6.4.0/spectrum-mpi-10.3.1.2-20200121-awz2q5brde7wgdqqw4ugalrkukeub4eb/include;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/include/c++/6.4.0;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/include/c++/6.4.0/powerpc64le-none-linux-gnu;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/include/c++/6.4.0/backward;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/lib/gcc/powerpc64le-none-linux-gnu/6.4.0/include;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/lib/gcc/powerpc64le-none-linux-gnu/6.4.0/include-fixed;/usr/local/include;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/sw/summit/cuda/10.1.243/targets/ppc64le-linux/lib/stubs;/sw/summit/cuda/10.1.243/targets/ppc64le-linux/lib;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/lib/gcc/powerpc64le-none-linux-gnu/6.4.0;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/lib/gcc;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/lib64;/lib64;/usr/lib64;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-6.4.0/spectrum-mpi-10.3.1.2-20200121-awz2q5brde7wgdqqw4ugalrkukeub4eb/lib;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/darshan-runtime-3.1.7-cnvxicgf5j4ap64qi6v5gxp67hmrjz43/lib;/autofs/nccs-svm1_sw/summit/gcc/6.4.0/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/sw/summit/xalt/1.2.0/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
