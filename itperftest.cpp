//
// Created by arnoldm on 04.10.24.
//
#include <iostream>
#include <mutex>
#include <numeric>
#include <ranges>
#include <execution>
#include <memory>
#include <random>
#include <source_location>
#include "test/common.h"
// #include "kunig.h"
#include "interval_tree.h"

struct simple_timer {
    explicit simple_timer(std::string_view name) : name_{name} {
    }

    ~simple_timer() noexcept {
        auto t2 = std::chrono::high_resolution_clock::now();
        std::println(std::cout, "Timer {}: {:11.3f}s", name_, std::chrono::duration<double>(t2 - start_).count());
    }

    std::chrono::high_resolution_clock::time_point start_{std::chrono::high_resolution_clock::now()};
    std::string name_;
};

int main(int argc, char *argv[]) {




    RandomFixture random_fixture;
#ifdef NDEBUG
    static constexpr int samples_cnt = 100'000;
    random_fixture.set_interval_cnt(500'000);
    static constexpr int interval_spread = 1'000'000;
    static constexpr int interval_max_len = 100;
#else
    static constexpr int samples_cnt = 10'000;
    random_fixture.set_interval_cnt(50'000);
    static constexpr int interval_spread = 50'000;
    static constexpr int interval_max_len = 100;
#endif

    interval_class<int>::gen_params.dist_start.
            param(std::uniform_int_distribution<int>::param_type{0, interval_spread});
    interval_class<int>::gen_params.dist_len.param(std::uniform_int_distribution<int>::param_type{0, interval_max_len});

    auto t1 = std::chrono::high_resolution_clock::now();

    auto i = random_fixture.input();
    auto j = i; // save a copy
    auto reduce = [](interval<int> const &acc, interval<int> const &x) -> interval<int> {
        return {std::min(acc.start_, x.start_), std::max(acc.end_, x.end_)};
    };
    auto const transform = interval_class<>::to_interval;
    auto [start, end] = std::transform_reduce(i.begin(), i.end(),
                                              interval_class<>::to_interval(i[0]),
                                              reduce, transform);

    interval_tree it{std::move(j), interval_class<>::to_interval};
    it.build_tree();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_build = (t2 - t1);

    std::cout << "\nTree statistics:\n";
    auto stats = it.tree_statistics();
    std::cout << stats;

    std::vector<performance_kpis> overall_kpis;
    std::vector<int> indexes;
    indexes.reserve(samples_cnt);
    {
        std::vector<int> candidates((end + 1) - (start - 1));
        std::iota(candidates.begin(), candidates.end(), start - 1);
        std::ranges::sample(candidates, std::back_inserter(indexes), samples_cnt,
                            std::mt19937_64{std::random_device{}()});
    }
    {
        std::mutex my_mutex;
        simple_timer st{"Total search time"};
        std::for_each(std::execution::par, indexes.begin(), indexes.end(),
                      brute_force_test(i, it, my_mutex, overall_kpis)
        );
    }

    int col_a = 30;
    std::println(std::cout, "{:{}}: {:7}", "interval_cnt", col_a, random_fixture.interval_cnt());
    std::println(std::cout, "{:{}}: {:7}", "interval_spread", col_a, interval_spread);
    std::println(std::cout, "{:{}}: {:7}", "interval_max_len", col_a, interval_max_len);
    std::println(std::cout, "{:{}}: {:7}", "samples_cnt", col_a, samples_cnt);
    std::println(std::cout, "{:{}}: {:7}", "start", col_a, start - 1);
    std::println(std::cout, "{:{}}: {:7}", "end", col_a, end + 1);
    std::println(std::cout, "{:{}}: {:11.3f}s", "elapsed_build", col_a, elapsed_build.count());
    // for (size_t i = 0; i < overall_kpis.size(); ++i) {
    //     std::cout << "\nThread " << i << ":\n";
    //     std::cout << overall_kpis[i];
    // }
    // std::cout << "\nOverall:\n";
    std::println(std::cout, "{:{}}: {:7}", "KPI structs", col_a, overall_kpis.size());
    std::cout << std::accumulate(overall_kpis.begin(), overall_kpis.end(), performance_kpis{});

}
