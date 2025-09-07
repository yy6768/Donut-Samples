
# Enable CUDA language
enable_language(CUDA)

include(../../donut/compileshaders.cmake)
file(GLOB shaders "*.hlsl")
file(GLOB sources "*.cpp" "*.h")
file(GLOB cuda_sources "*.cu")

set(project tensor)
set(folder "Examples/Tensor")

# Find CUDA
find_package(CUDA REQUIRED)

donut_compile_shaders_all_platforms(
    TARGET ${project}_shaders
    PROJECT_NAME "Tensor"
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    FOLDER ${folder}
    OUTPUT_BASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}
)

# Create executable with CUDA support
add_executable(${project} WIN32 ${sources} ${cuda_sources})

# Set CUDA properties
set_property(TARGET ${project} PROPERTY CUDA_ARCHITECTURES 70 75 80 86 89)
set_property(TARGET ${project} PROPERTY CUDA_STANDARD 17)
set_property(TARGET ${project} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(${project} 
    donut_app 
    donut_engine
    ${CUDA_LIBRARIES}
    cudart
)

# Include CUDA headers
target_include_directories(${project} PRIVATE ${CUDA_INCLUDE_DIRS})

# Add dependencies
add_dependencies(${project} ${project}_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()
