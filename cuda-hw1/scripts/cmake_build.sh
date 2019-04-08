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
  
  local cmakePath=""
  cmakeSetup cmakePath ${cmakeVersion} ${skipDownload}
  
  # Build 
  ${cmakePath} .. \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build .
  cp src/cuda_impl/cuda_hw1 ../cuda_hw1
  cp src/openmp_impl/openmp_hw1 ../openmp_hw1

}

main

