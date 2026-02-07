cmake_minimum_required(VERSION 3.18)
project(MTModule)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")

# Sources
set(MT_SOURCES
    mt_wrapper.cpp
    mt_wrapper.h
    marian_mt_wrapper.cpp
    marian_mt_wrapper.h
)

add_library(mt_wrapper STATIC ${MT_SOURCES})
target_include_directories(mt_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# Android: use prebuilt imported targets created by parent CMake
# ============================================================
if(ANDROID)
    # ONNX Runtime
    if(TARGET onnxruntime)
        message(STATUS "MT: linking ONNX Runtime (Android prebuilt)")
        target_compile_definitions(mt_wrapper PUBLIC USE_ONNXRUNTIME)
        target_link_libraries(mt_wrapper onnxruntime)
        if(ORT_INCLUDE_DIR)
            target_include_directories(mt_wrapper PUBLIC "${ORT_INCLUDE_DIR}")
        endif()
    else()
        message(STATUS "MT: ONNX Runtime not available — fallback mode")
    endif()

    # SentencePiece
    if(TARGET sentencepiece)
        message(STATUS "MT: linking SentencePiece (Android prebuilt)")
        target_compile_definitions(mt_wrapper PUBLIC USE_SENTENCEPIECE)
        target_link_libraries(mt_wrapper sentencepiece)
        if(SP_INCLUDE_DIR)
            target_include_directories(mt_wrapper PUBLIC "${SP_INCLUDE_DIR}")
        endif()
    else()
        message(STATUS "MT: SentencePiece not available — fallback tokenization")
    endif()

# ============================================================
# Desktop: use pkg-config or conda
# ============================================================
else()
    # ---------- SentencePiece ----------
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(SENTENCEPIECE QUIET sentencepiece)
    endif()

    if(NOT SENTENCEPIECE_FOUND)
        set(SP_ROOT "$ENV{CONDA_PREFIX}")
        if(SP_ROOT AND EXISTS "${SP_ROOT}/include/sentencepiece_processor.h")
            set(SENTENCEPIECE_FOUND TRUE)
            set(SENTENCEPIECE_INCLUDE_DIRS "${SP_ROOT}/include")
            set(SENTENCEPIECE_LIBRARY_DIRS "${SP_ROOT}/lib")
            set(SENTENCEPIECE_LIBRARIES "sentencepiece")
        endif()
    endif()

    if(SENTENCEPIECE_FOUND)
        message(STATUS "SentencePiece found")
        target_compile_definitions(mt_wrapper PUBLIC USE_SENTENCEPIECE)
        target_include_directories(mt_wrapper PUBLIC ${SENTENCEPIECE_INCLUDE_DIRS})
        if(SENTENCEPIECE_LIBRARY_DIRS)
            target_link_directories(mt_wrapper PRIVATE ${SENTENCEPIECE_LIBRARY_DIRS})
        endif()
        target_link_libraries(mt_wrapper ${SENTENCEPIECE_LIBRARIES})
    else()
        message(STATUS "SentencePiece not found — MT will use fallback tokenization")
    endif()

    # ---------- ONNX Runtime ----------
    if(PkgConfig_FOUND)
        pkg_check_modules(ONNXRUNTIME QUIET onnxruntime)
    endif()

    if(ONNXRUNTIME_FOUND)
        message(STATUS "ONNX Runtime found via pkg-config")
        target_compile_definitions(mt_wrapper PUBLIC USE_ONNXRUNTIME)
        target_include_directories(mt_wrapper PUBLIC ${ONNXRUNTIME_INCLUDE_DIRS})
        target_link_libraries(mt_wrapper ${ONNXRUNTIME_LIBRARIES})
    else()
        set(ORT_ROOT "$ENV{CONDA_PREFIX}")
        if(ORT_ROOT AND EXISTS "${ORT_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
           AND EXISTS "${ORT_ROOT}/lib/libonnxruntime.so")
            message(STATUS "ONNX Runtime found in conda")
            target_compile_definitions(mt_wrapper PUBLIC USE_ONNXRUNTIME)
            target_include_directories(mt_wrapper PUBLIC "${ORT_ROOT}/include")
            target_link_directories(mt_wrapper PRIVATE "${ORT_ROOT}/lib")
            target_link_libraries(mt_wrapper onnxruntime)
        else()
            message(STATUS "ONNX Runtime not found — MT will use fallback")
        endif()
    endif()
endif()
