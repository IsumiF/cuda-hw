#!/usr/bin/env bash

cmakeSetup() {
  local -n cmakePath_=$1
  local -r cmakeVersion_=$2
  local -r skipDownload_=$3

  local -r cmakeFolder_=cmake-${cmakeVersion}-Linux-x86_64
  if [ ${skipDownload_} == "false" && -not -f ${cmakeFolder_}/bin/cmake ]; then
    local
    cmakeUrl_=https://github.com/Kitware/CMake/releases/download/v${cmakeVersion_}/${cmakeFolder_}.tar.gz
    wget ${cmakeUrl_}
    tar xf ${cmakeFolder_}.tar.gz
  fi
  cmakePath=${cmakeFolder_}/bin/cmake
}

