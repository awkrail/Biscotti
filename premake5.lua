workspace "biscotti"
  configurations { "Release", "Debug" }
  language "C++"
  flags { "C++11" }
  includedirs { ".", "third_party/butteraugli", "third_party/tensorflow" } -- TODO: ADD Tensorflow

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
    files
      {
        "Biscotti/*.cc",
        "Biscotti/*.h",
        "third_party/butteraugli/butteraugli/butteraugli.cc",
        "third_party/butteraugli/butteraugli/butteraugli.h",
        "third_party/tensorflow",
        "third_party/tensorflow/bazel-tensorflow/external/eigen_archive",
        "third_party/tensorflow/bazel-tensorflow/external/protobuf/src",
        "third_party/tensorflow/bazel-genfiles",
        "third_party/tensorflow/bazel-tensorflow/external/nsync/public"
      }
    removefiles "Biscotti/biscotti.cc"
    filter "action:gmake"
      linkoptions { "`pkg-config --libs libpng || libpng-config --static --ldflags -L third_party/tensorflow/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework`" }
      buildoptions { "`pkg-config --cflags libpng || libpng-config --static --cflags -L third_party/tensorflow/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework`" }
  
  project "biscotti"
    kind "ConsoleApp"
    filter "action:gmake"
      linkoptions { "`pkg-config --libs libpng || libpng-config --ldflags -L third_party/tensorflow/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework`" }
      buildoptions { "`pkg-config --cflags libpng || libpng-config --cflags -L third_party/tensorflow/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework`" }
    filter "action:vs*"
      links { "shlwapi" }
    filter {}
    files
      {
        "Biscotti/*.cc",
        "Biscotti/*.h",
        "third_party/butteraugli/butteraugli/butteraugli.cc",
        "third_party/butteraugli/butteraugli/butteraugli.h",
        "third_party/tensorflow",
        "third_party/tensorflow/bazel-tensorflow/external/eigen_archive",
        "third_party/tensorflow/bazel-tensorflow/external/protobuf/src",
        "third_party/tensorflow/bazel-genfiles",
        "third_party/tensorflow/bazel-tensorflow/external/nsync/public"
      }