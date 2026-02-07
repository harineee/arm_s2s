cmake_minimum_required(VERSION 3.18)
project(TTSModule)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")

add_library(tts_wrapper STATIC tts_wrapper.cpp)
target_include_directories(tts_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# Android: use prebuilt imported target created by parent CMake
# ============================================================
if(ANDROID)
    if(TARGET onnxruntime)
        message(STATUS "TTS: linking ONNX Runtime (Android prebuilt)")
        target_compile_definitions(tts_wrapper PUBLIC USE_ONNXRUNTIME)
        target_link_libraries(tts_wrapper onnxruntime)
        if(ORT_INCLUDE_DIR)
            target_include_directories(tts_wrapper PUBLIC "${ORT_INCLUDE_DIR}")
        endif()
    else()
        message(STATUS "TTS: ONNX Runtime not available — test tone fallback")
    endif()

# ============================================================
# Desktop: use pkg-config or conda
# ============================================================
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ONNXRUNTIME QUIET onnxruntime)
    endif()

    if(ONNXRUNTIME_FOUND)
        message(STATUS "TTS: ONNX Runtime found via pkg-config")
        target_compile_definitions(tts_wrapper PUBLIC USE_ONNXRUNTIME)
        target_include_directories(tts_wrapper PUBLIC ${ONNXRUNTIME_INCLUDE_DIRS})
        target_link_libraries(tts_wrapper ${ONNXRUNTIME_LIBRARIES})
    else()
        set(ORT_ROOT "$ENV{CONDA_PREFIX}")
        if(ORT_ROOT AND EXISTS "${ORT_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
           AND EXISTS "${ORT_ROOT}/lib/libonnxruntime.so")
            message(STATUS "TTS: ONNX Runtime found in conda")
            target_compile_definitions(tts_wrapper PUBLIC USE_ONNXRUNTIME)
            target_include_directories(tts_wrapper PUBLIC "${ORT_ROOT}/include")
            target_link_directories(tts_wrapper PRIVATE "${ORT_ROOT}/lib")
            target_link_libraries(tts_wrapper onnxruntime)
        else()
            message(STATUS "TTS: ONNX Runtime not found — test tone fallback")
        endif()
    endif()
endif()
