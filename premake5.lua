workspace "biscotti"
  configurations { "Release", "Debug" }
  language "C++"
  flags { "C++11" }
  includedirs { ".", "third_party/butteraugli", "third_party/tensorflow",
                "third_party/tensorflow/bazel-tensorflow/external/eigen_archive",
                "third_party/tensorflow/bazel-tensorflow/external/protobuf/src",
                "third_party/tensorflow/bazel-genfiles",
                "third_party/tensorflow/bazel-tensorflow/external/nsync/public" }
  filter "action:vs*"
    platforms { "x86_64", "x86" }
  
  filter "platforms:x86"
    architecture "x86"
  filter "platforms:x86_64"
    architecture "x86_64"
  
  filter "action:gmake"
    symbols "On"

  filter "configurations:Debug"
    symbols "On"
  filter "configurations:Release"
    optimize "Full"
  filter {}

  project "biscotti_static"
    kind "StaticLib"
    files {
      "biscotti/*.cc",
      "biscotti/*.h",
      "third_party/butteraugli/butteraugli/butteraugli.cc",
      "third_party/butteraugli/butteraugli/butteraugli.h"
    }
    removefiles "biscotti/biscotti.cc"
    filter "action:gmake"
      linkoptions { "`pkg-config --libs libpng opencv || libpng-config --static --ldflags`", "-L third_party/tensorflow/bazel-bin/tensorflow", "-ltensorflow_cc", "-ltensorflow_framework" }
      buildoptions { "`pkg-config --cflags libpng opencv || libpng-config --static --cflags`" }

  project "biscotti"
    kind "ConsoleApp"
    filter "action:gmake"
      linkoptions { "`pkg-config --libs libpng opencv || libpng-config --ldflags`", "-L third_party/tensorflow/bazel-bin/tensorflow", "-ltensorflow_cc", "-ltensorflow_framework" }
      buildoptions { "`pkg-config --cflags libpng opencv || libpng-config --cflags`" }
    files {
      "biscotti/*.cc",
      "biscotti/*.h",
      "third_party/butteraugli/butteraugli/butteraugli.cc",
      "third_party/butteraugli/butteraugli/butteraugli.h"
    }