workspace "biscotti"
  configurations { "Release", "Debug" }
  language "C++"
  flags { "C++11" }
  includedirs { ".", "third_party/tensorflow" }

  filter "action:vs*"
    platforms { "x86_64", "x86" }
  
  filter "platforms:x86"
    architecture "x86"
  filter "platforms:x86_64"
    architecture "x86_64"
  
  filter "action:gmake"
    symbols "On"