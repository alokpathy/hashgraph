# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /autofs/nccs-svm1_sw/summit/.swci/0-core/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/cmake-3.15.2-xit2o3iepxvqbyku77lwcugufilztu7t/bin/cmake

# The command to remove a file.
RM = /autofs/nccs-svm1_sw/summit/.swci/0-core/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/cmake-3.15.2-xit2o3iepxvqbyku77lwcugufilztu7t/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /autofs/nccs-svm1_home1/alokt/hashgraph

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /autofs/nccs-svm1_home1/alokt/hashgraph/build

# Include any dependencies generated for this target.
include CMakeFiles/sing-hash.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sing-hash.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sing-hash.dir/flags.make

CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.o: CMakeFiles/sing-hash.dir/flags.make
CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.o: ../test/SingleHashGraphTest.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/autofs/nccs-svm1_home1/alokt/hashgraph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.o"
	/sw/summit/cuda/10.1.243/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /autofs/nccs-svm1_home1/alokt/hashgraph/test/SingleHashGraphTest.cu -o CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.o

CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sing-hash
sing__hash_OBJECTS = \
"CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.o"

# External object files for target sing-hash
sing__hash_EXTERNAL_OBJECTS =

sing-hash: CMakeFiles/sing-hash.dir/test/SingleHashGraphTest.cu.o
sing-hash: CMakeFiles/sing-hash.dir/build.make
sing-hash: libalg.a
sing-hash: CMakeFiles/sing-hash.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/autofs/nccs-svm1_home1/alokt/hashgraph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable sing-hash"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sing-hash.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sing-hash.dir/build: sing-hash

.PHONY : CMakeFiles/sing-hash.dir/build

CMakeFiles/sing-hash.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sing-hash.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sing-hash.dir/clean

CMakeFiles/sing-hash.dir/depend:
	cd /autofs/nccs-svm1_home1/alokt/hashgraph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /autofs/nccs-svm1_home1/alokt/hashgraph /autofs/nccs-svm1_home1/alokt/hashgraph /autofs/nccs-svm1_home1/alokt/hashgraph/build /autofs/nccs-svm1_home1/alokt/hashgraph/build /autofs/nccs-svm1_home1/alokt/hashgraph/build/CMakeFiles/sing-hash.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sing-hash.dir/depend

