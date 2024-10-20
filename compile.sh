#! /bin/bash

# create the build directory, skip if it already exists
mkdir -p build

# navigate to the build directory
cd build

# run cmake
cmake ..

# run make
make -j96