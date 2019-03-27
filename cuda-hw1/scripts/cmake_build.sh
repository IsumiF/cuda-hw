#!/usr/bin/env bash

declare -r buildDir=cmake-build-release
declare -r cmakeVersion=3.13.4
declare -r skipDownload=false

declare -r DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

source ${DIR}/cmakeSetup.sh

main() {

mkdir -p ${buildDir}
cd ${buildDir}

local ${FUNCNAME[0]}_cmakePath=""
cmakeSetup cmakePath ${cmakeVersion} ${skipDownload}

# Build 
${cmakePath} .. \
  -DCMAKE_BUILD_TYPE=Release
cmake --build .
cp cuda_hw1 ../hw1

}

main

