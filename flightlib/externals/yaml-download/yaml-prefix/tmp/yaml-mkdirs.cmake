# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ssen/flightmare/flightlib/externals/yaml-src"
  "/home/ssen/flightmare/flightlib/externals/yaml-bin"
  "/home/ssen/flightmare/flightlib/externals/yaml-download/yaml-prefix"
  "/home/ssen/flightmare/flightlib/externals/yaml-download/yaml-prefix/tmp"
  "/home/ssen/flightmare/flightlib/externals/yaml-download/yaml-prefix/src/yaml-stamp"
  "/home/ssen/flightmare/flightlib/externals/yaml-download/yaml-prefix/src"
  "/home/ssen/flightmare/flightlib/externals/yaml-download/yaml-prefix/src/yaml-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ssen/flightmare/flightlib/externals/yaml-download/yaml-prefix/src/yaml-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ssen/flightmare/flightlib/externals/yaml-download/yaml-prefix/src/yaml-stamp${cfgdir}") # cfgdir has leading slash
endif()
