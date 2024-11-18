find_package(doctest REQUIRED)

add_executable(test_it test_it.cpp ../include/interval_tree.h
        common.h
)
target_link_libraries(test_it PRIVATE doctest::doctest interval_tree tbb)
target_compile_options(test_it PRIVATE -fopenmp)
target_link_options(test_it PRIVATE -fopenmp)

