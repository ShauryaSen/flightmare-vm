# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ssen/flightmare/flightlib/externals/pybind11-src"
  "/home/ssen/flightmare/flightlib/externals/pybind11-bin"
  "/home/ssen/flightmare/flightlib/externals/pybind11-download/pybind11-prefix"
  "/home/ssen/flightmare/flightlib/externals/pybind11-download/pybind11-prefix/tmp"
  "/home/ssen/flightmare/flightlib/externals/pybind11-download/pybind11-prefix/src/pybind11-stamp"
  "/home/ssen/flightmare/flightlib/externals/pybind11-download/pybind11-prefix/src"
  "/home/ssen/flightmare/flightlib/externals/pybind11-download/pybind11-prefix/src/pybind11-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ssen/flightmare/flightlib/externals/pybind11-download/pybind11-prefix/src/pybind11-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ssen/flightmare/flightlib/externals/pybind11-download/pybind11-prefix/src/pybind11-stamp${cfgdir}") # cfgdir has leading slash
endif()
