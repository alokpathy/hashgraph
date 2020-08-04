# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /autofs/nccs-svm1_sw/summit/.swci/0-core/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/cmake-3.17.3-ranbt2pk3wzzvd2i7j3ekexaqya3m4f2/bin/cmake

# The command to remove a file.
RM = /autofs/nccs-svm1_sw/summit/.swci/0-core/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/cmake-3.17.3-ranbt2pk3wzzvd2i7j3ekexaqya3m4f2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /autofs/nccs-svm1_home1/alokt/hashgraph

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /autofs/nccs-svm1_home1/alokt/hashgraph/build

# Include any dependencies generated for this target.
include CMakeFiles/alg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/alg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/alg.dir/flags.make

CMakeFiles/alg.dir/src/MultiHashGraph.cu.o: CMakeFiles/alg.dir/flags.make
CMakeFiles/alg.dir/src/MultiHashGraph.cu.o: ../src/MultiHashGraph.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/autofs/nccs-svm1_home1/alokt/hashgraph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/alg.dir/src/MultiHashGraph.cu.o"
	/sw/summit/cuda/10.1.243/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /autofs/nccs-svm1_home1/alokt/hashgraph/src/MultiHashGraph.cu -o CMakeFiles/alg.dir/src/MultiHashGraph.cu.o

CMakeFiles/alg.dir/src/MultiHashGraph.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/alg.dir/src/MultiHashGraph.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/alg.dir/src/MultiHashGraph.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/alg.dir/src/MultiHashGraph.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/alg.dir/src/SingleHashGraph.cu.o: CMakeFiles/alg.dir/flags.make
CMakeFiles/alg.dir/src/SingleHashGraph.cu.o: ../src/SingleHashGraph.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/autofs/nccs-svm1_home1/alokt/hashgraph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/alg.dir/src/SingleHashGraph.cu.o"
	/sw/summit/cuda/10.1.243/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /autofs/nccs-svm1_home1/alokt/hashgraph/src/SingleHashGraph.cu -o CMakeFiles/alg.dir/src/SingleHashGraph.cu.o

CMakeFiles/alg.dir/src/SingleHashGraph.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/alg.dir/src/SingleHashGraph.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/alg.dir/src/SingleHashGraph.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/alg.dir/src/SingleHashGraph.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target alg
alg_OBJECTS = \
"CMakeFiles/alg.dir/src/MultiHashGraph.cu.o" \
"CMakeFiles/alg.dir/src/SingleHashGraph.cu.o"

# External object files for target alg
alg_EXTERNAL_OBJECTS =

libalg.a: CMakeFiles/alg.dir/src/MultiHashGraph.cu.o
libalg.a: CMakeFiles/alg.dir/src/SingleHashGraph.cu.o
libalg.a: CMakeFiles/alg.dir/build.make
libalg.a: CMakeFiles/alg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/autofs/nccs-svm1_home1/alokt/hashgraph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA static library libalg.a"
	$(CMAKE_COMMAND) -P CMakeFiles/alg.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/alg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/alg.dir/build: libalg.a

.PHONY : CMakeFiles/alg.dir/build

CMakeFiles/alg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/alg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/alg.dir/clean

CMakeFiles/alg.dir/depend:
	cd /autofs/nccs-svm1_home1/alokt/hashgraph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /autofs/nccs-svm1_home1/alokt/hashgraph /autofs/nccs-svm1_home1/alokt/hashgraph /autofs/nccs-svm1_home1/alokt/hashgraph/build /autofs/nccs-svm1_home1/alokt/hashgraph/build /autofs/nccs-svm1_home1/alokt/hashgraph/build/CMakeFiles/alg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/alg.dir/depend

