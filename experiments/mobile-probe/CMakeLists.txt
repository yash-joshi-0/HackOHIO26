cmake_minimum_required(VERSION 3.28)
project(MeshThingProbe LANGUAGES CXX Swift)

set(MESH_LLAMA_ROOT "" CACHE PATH "Private patched llama.cpp clone")
set(MESH_FIXTURE "" CACHE FILEPATH "GLM two-layer fixture")
set(MESH_APP_TEAM "" CACHE STRING "Apple development team")
set(MESH_LIB_ROOT "${MESH_LLAMA_ROOT}/build-ios-probe")

add_executable(MeshThingProbe MACOSX_BUNDLE ProbeApp.swift native_smoke.cpp "${MESH_FIXTURE}")
set_source_files_properties("${MESH_FIXTURE}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
target_include_directories(MeshThingProbe PRIVATE "${MESH_LLAMA_ROOT}/include" "${MESH_LLAMA_ROOT}/src" "${MESH_LLAMA_ROOT}/ggml/include")
target_compile_features(MeshThingProbe PRIVATE cxx_std_17)
target_compile_options(MeshThingProbe PRIVATE "$<$<COMPILE_LANGUAGE:Swift>:-parse-as-library>")
target_link_libraries(MeshThingProbe PRIVATE
  "${MESH_LIB_ROOT}/src/Release-iphoneos/libllama.a"
  "${MESH_LIB_ROOT}/ggml/src/Release-iphoneos/libggml.a"
  "${MESH_LIB_ROOT}/ggml/src/Release-iphoneos/libggml-cpu.a"
  "${MESH_LIB_ROOT}/ggml/src/ggml-metal/Release-iphoneos/libggml-metal.a"
  "${MESH_LIB_ROOT}/ggml/src/ggml-blas/Release-iphoneos/libggml-blas.a"
  "${MESH_LIB_ROOT}/ggml/src/Release-iphoneos/libggml-base.a"
  "-framework Foundation" "-framework Metal" "-framework MetalKit" "-framework Accelerate" "c++")
set_target_properties(MeshThingProbe PROPERTIES
  MACOSX_BUNDLE_GUI_IDENTIFIER "systems.stephenson.meshthing-native-probe"
  MACOSX_BUNDLE_BUNDLE_NAME "MeshThingProbe"
  MACOSX_BUNDLE_BUNDLE_VERSION "1"
  MACOSX_BUNDLE_SHORT_VERSION_STRING "0.1"
  XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${MESH_APP_TEAM}"
  XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
  XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development"
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "systems.stephenson.meshthing-native-probe"
  XCODE_ATTRIBUTE_SWIFT_VERSION "5.0"
  XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
  XCODE_ATTRIBUTE_GENERATE_INFOPLIST_FILE "YES"
  XCODE_ATTRIBUTE_INFOPLIST_KEY_UILaunchScreen_Generation "YES"
  XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Bridge.h")
