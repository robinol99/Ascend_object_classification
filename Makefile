# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/robinol/anaconda3/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/robinol/anaconda3/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/robinol/git_projects/onnx_tensorrt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robinol/git_projects/onnx_tensorrt

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/home/robinol/anaconda3/lib/python3.9/site-packages/cmake/data/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/home/robinol/anaconda3/lib/python3.9/site-packages/cmake/data/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/robinol/git_projects/onnx_tensorrt/CMakeFiles /home/robinol/git_projects/onnx_tensorrt//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/robinol/git_projects/onnx_tensorrt/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named object_classification_preprocess

# Build rule for target.
object_classification_preprocess: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 object_classification_preprocess
.PHONY : object_classification_preprocess

# fast build rule for target.
object_classification_preprocess/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/object_classification_preprocess.dir/build.make CMakeFiles/object_classification_preprocess.dir/build
.PHONY : object_classification_preprocess/fast

object_classification_preprocess.o: object_classification_preprocess.cpp.o
.PHONY : object_classification_preprocess.o

# target to build an object file
object_classification_preprocess.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/object_classification_preprocess.dir/build.make CMakeFiles/object_classification_preprocess.dir/object_classification_preprocess.cpp.o
.PHONY : object_classification_preprocess.cpp.o

object_classification_preprocess.i: object_classification_preprocess.cpp.i
.PHONY : object_classification_preprocess.i

# target to preprocess a source file
object_classification_preprocess.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/object_classification_preprocess.dir/build.make CMakeFiles/object_classification_preprocess.dir/object_classification_preprocess.cpp.i
.PHONY : object_classification_preprocess.cpp.i

object_classification_preprocess.s: object_classification_preprocess.cpp.s
.PHONY : object_classification_preprocess.s

# target to generate assembly for a file
object_classification_preprocess.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/object_classification_preprocess.dir/build.make CMakeFiles/object_classification_preprocess.dir/object_classification_preprocess.cpp.s
.PHONY : object_classification_preprocess.cpp.s

utilities/utilities_nvidia.o: utilities/utilities_nvidia.cpp.o
.PHONY : utilities/utilities_nvidia.o

# target to build an object file
utilities/utilities_nvidia.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/object_classification_preprocess.dir/build.make CMakeFiles/object_classification_preprocess.dir/utilities/utilities_nvidia.cpp.o
.PHONY : utilities/utilities_nvidia.cpp.o

utilities/utilities_nvidia.i: utilities/utilities_nvidia.cpp.i
.PHONY : utilities/utilities_nvidia.i

# target to preprocess a source file
utilities/utilities_nvidia.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/object_classification_preprocess.dir/build.make CMakeFiles/object_classification_preprocess.dir/utilities/utilities_nvidia.cpp.i
.PHONY : utilities/utilities_nvidia.cpp.i

utilities/utilities_nvidia.s: utilities/utilities_nvidia.cpp.s
.PHONY : utilities/utilities_nvidia.s

# target to generate assembly for a file
utilities/utilities_nvidia.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/object_classification_preprocess.dir/build.make CMakeFiles/object_classification_preprocess.dir/utilities/utilities_nvidia.cpp.s
.PHONY : utilities/utilities_nvidia.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... object_classification_preprocess"
	@echo "... object_classification_preprocess.o"
	@echo "... object_classification_preprocess.i"
	@echo "... object_classification_preprocess.s"
	@echo "... utilities/utilities_nvidia.o"
	@echo "... utilities/utilities_nvidia.i"
	@echo "... utilities/utilities_nvidia.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
