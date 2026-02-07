cmake_minimum_required(VERSION 3.18)
project(HostTranslation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
    message(STATUS "Host: ARM64 NEON")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
    message(STATUS "Host: x86_64 AVX2")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")

# PortAudio (optional, real-time mic input)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PORTAUDIO QUIET portaudio-2.0)
endif()
if(PORTAUDIO_FOUND)
    message(STATUS "PortAudio found — real-time mode enabled")
    add_definitions(-DUSE_PORTAUDIO)
else()
    message(STATUS "PortAudio not found — file mode only")
endif()

# Component libraries
add_subdirectory(../mt  mt)
add_subdirectory(../asr asr)
add_subdirectory(../tts tts)
add_subdirectory(../pipeline pipeline)

add_executable(translation_host main.cpp)

target_include_directories(translation_host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/../asr
    ${CMAKE_CURRENT_SOURCE_DIR}/../mt
    ${CMAKE_CURRENT_SOURCE_DIR}/../tts
)

target_link_libraries(translation_host
    pipeline asr_wrapper mt_wrapper tts_wrapper
)

if(PORTAUDIO_FOUND)
    target_include_directories(translation_host PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(translation_host ${PORTAUDIO_LIBRARIES})
endif()

if(UNIX)
    target_link_libraries(translation_host pthread)
endif()
