cmake_minimum_required(VERSION 3.29)
project(kunig2)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_PERFORMANCE_OPTIONS "Use IPO/LTO -fopenmp -mavx2 -ffast-math" ON)
option(USE_SANITIZER "Use gcc sanitizers address,undefined" ON)

function(target_enable_lto TARGET ENABLE)
    if (NOT ${ENABLE})
        return()
    endif ()
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        message("IPO/LTO is supported")
        set_target_properties(${TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION $(ENABLE) )
    endif ()
endfunction()

include_directories(include)

if (${USE_SANITIZER})
    set(SANITIZE_OPTIONS "-fsanitize=address,undefined")
    message(STATUS "Using sanitizers ${SANITIZE_OPTIONS}")
    add_compile_options(${SANITIZE_OPTIONS})
    add_link_options(${SANITIZE_OPTIONS})
endif()

add_compile_options(-Werror -Wall -Wpedantic -Wconversion)

add_library(interval_tree INTERFACE
        include/interval_tree.h
)

add_executable(itperftest itperftest.cpp
        include/interval_tree.h
        test/common.h)

target_link_libraries(itperftest INTERFACE interval_tree)
message(STATUS "Using performance options")
target_enable_lto(itperftest ON)
target_enable_lto(interval_tree ON)
target_link_options(itperftest PUBLIC -fopenmp)
target_link_libraries(itperftest PUBLIC tbb)
target_compile_options(itperftest PUBLIC -fopenmp #[[-mavx2 -ffast-math]])

if (${USE_PERFORMANCE_OPTIONS})
    message(STATUS "Using performance options")
    target_enable_lto(itperftest ON)
    target_enable_lto(interval_tree ON)
    target_link_options(itperftest PUBLIC -fopenmp)
    target_link_libraries(itperftest PUBLIC tbb)
    target_compile_options(itperftest PUBLIC -fopenmp -mavx2 -ffast-math)
endif ()


include(CTest)
enable_testing()
add_subdirectory(test)
