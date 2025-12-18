cmake_minimum_required(VERSION 3.11)

project(SnakeNEAT)

include(FetchContent)

FetchContent_Declare(
    SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.2
)
FetchContent_MakeAvailable(SFML)

get_filename_component(NEAT_ROOT ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
add_subdirectory(${NEAT_ROOT} ${CMAKE_BINARY_DIR}/neat)

add_executable(SnakeNEAT main.cpp) 

target_compile_options(SnakeNEAT PRIVATE
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
	$<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Werror -O3 -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wunreachable-code -Wconversion -Wno-error=maybe-uninitialized -Wpedantic -Wcast-align -Wno-error=vla>
	$<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3 -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wunreachable-code -Wconversion -Wpedantic>
)

target_link_libraries(SnakeNEAT neat SFML::Graphics SFML::Window SFML::System)
