cmake_minimum_required(VERSION 3.18)
project(BenchmarkPlugin)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(ROOT_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(PLUGIN_INTEROP_UNITY_CUDA "${CMAKE_CURRENT_SOURCE_DIR}/Vendors/interopUnityCUDA/Plugin")
set(UTILITIES_INCLUDE "${PLUGIN_INTEROP_UNITY_CUDA}/Utilities/include")
set(PLUGIN_INTEROP_INCLUDE "${PLUGIN_INTEROP_UNITY_CUDA}/PluginInteropUnityCUDA/include")

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

if(UNIX AND NOT APPLE)
    add_definitions(-DUNITY_LINUX)
endif()

add_subdirectory(Vendors)

# Define the executable target
file(GLOB SOURCES "src/*.cpp")
file(GLOB SOURCES_CU "src/kernels/*.cu")
file(GLOB HEADER_CU "include/kernels/*.cuh")
file(GLOB HEADER "include/*.h")
add_library(BenchmarkPlugin SHARED ${SOURCES} ${SOURCES_CU} ${HEADER} ${HEADER_CU})


target_include_directories(BenchmarkPlugin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_directories(BenchmarkPlugin PUBLIC 
    ${CUDAToolkit_LIBRARY_DIR})

# Linking with release libraries
target_link_libraries(BenchmarkPlugin optimized 
    Utilities
    ${OPENGL_LIBRARIES} 
    PluginInteropUnityCUDA
    CUDA::cudart
)


if(UNIX AND NOT APPLE)
    add_definitions(-DUNITY_LINUX)
endif()


set_target_properties(BenchmarkPlugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_NAME}/../"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_NAME}/../"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_NAME}/../"
    DEBUG_OUTPUT_NAME "d_${PROJECT_NAME}"
    RELEASE_OUTPUT_NAME "${PROJECT_NAME}"
    MINSIZEREL_OUTPUT_NAME "${PROJECT_NAME}"
    RELWITHDEBINFO_OUTPUT_NAME "${PROJECT_NAME}"
    LINKER_LANGUAGE CXX
)