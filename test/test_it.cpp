//
// Created by arnoldm on 09.10.24.
//
#include <random>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <iterator>
#include "interval_tree.h"
#include <algorithm>
#include <ranges>
#include <mutex>
#include <execution>
#include <execution>
#include "interval_tree.h"
#include "common.h"


struct Fixture2 : public TestFixture {
    std::vector<interval_class<int> > input() override {
        return {
            {{1, 100}, 1},
            {{50, 100}, 2},
            {{20, 80}, 3},
            {{30, 40}, 4},
            {{30, 60}, 5},
            {{1, 50}, 6},
            {{70, 80}, 7},
        };
    }
};


struct FixtureJustOne : TestFixture {
    std::vector<interval_class<int> > input() override {
        return {{{2070, 2071}, 1}};
    }
};

struct FixtureTwoClose : TestFixture {
    std::vector<interval_class<int> > input() override {
        return {{{36726, 36765}, 1}, {{36701, 36725}, 2}};
    }
};

TEST_SUITE("Interval Tree") {
    TEST_CASE("Create interval") {
        [[maybe_unused]] interval_class<> i1 = interval_class<>::generate();
        [[maybe_unused]] interval_class<> i2 = interval_class<>::generate();
    }

    TEST_CASE_FIXTURE(TestFixture, "find_midpoint") {
        init();
        auto mp = interval_tree_type::find_midpoint(starts(), ends());
        CHECK_EQ(mp.cnt_to_left_, mp.cnt_to_right_);
    }

    TEST_CASE_FIXTURE(FixtureJustOne, "find_midpoint just 1") {
        init();
        auto mp = interval_tree_type::find_midpoint(starts(), ends());
        CHECK_EQ(mp.cnt_to_left_, mp.cnt_to_right_);
        CHECK_EQ(mp.cnt_to_left_, 0);
        CHECK_EQ(mp.cnt_intersecting_, 1);
    }

    TEST_CASE_FIXTURE(FixtureTwoClose, "find_midpoint two close") {
        init();
        auto mp = interval_tree_type::find_midpoint(starts(), ends());
        CHECK_LE(absolute_unsigned_difference(mp.cnt_to_left_, mp.cnt_to_right_), 1);
        CHECK_EQ(mp.cnt_intersecting_, 1);
    }

    TEST_CASE_FIXTURE(TestFixture, "Create interval tree") {
        interval_tree it{std::move(input()), interval_class<>::to_interval};
        it.build_tree();
    }

    TEST_CASE_FIXTURE(Fixture2, "Create interval tree 2") {
        interval_tree it{std::move(input()), interval_class<>::to_interval};
        it.build_tree();
    }

    TEST_CASE_FIXTURE(RandomFixture, "Create interval tree with many intervals") {
        set_interval_cnt(100'000ull);
        auto i = input();
        interval_tree it{std::move(i), interval_class<>::to_interval};
        it.build_tree();
    }

    TEST_CASE_FIXTURE(TestFixture, "find_intersecting(coord)") {
        interval_tree it{std::move(input()), interval_class<>::to_interval};
        it.build_tree();

        auto test = [&it](int location, int expected) {
            int found = 0;
            auto callback = [&found, &location](interval_class<> const &i) {
                ++found;
                CHECK(i.interval_.contains(location));
            };
            it.find_intersecting(location, callback);
            CHECK_EQ(found, expected);
        };
        test(0, 0);
        test(1, 1);
        test(6, 2);
        test(19, 1);
        test(20, 2);
        test(21, 1);
        test(30, 1);
        test(31, 0);
    }

    TEST_CASE_FIXTURE(TestFixture, "find_overlapping(interval)") {
        interval_tree it{std::move(input()), interval_class<>::to_interval};
        it.build_tree();

        auto test = [&it](int start, int end, int expected) {
            int found = 0;
            auto callback = [&found, search = interval<int>{start, end}](interval_class<> const &i) {
                ++found;
                CHECK(i.interval_.overlaps(search).has_value());
            };
            it.find_overlapping(start, end, callback);
            CHECK_EQ(found, expected);
        };
        test(-1, 0, 0);
        test(1, 2, 1);
        test(6, 15, 2);
        test(22, 23, 1);
        test(0, 31, 3);
        test(40, 44, 0);
    }

    TEST_CASE_FIXTURE(TestFixture, "find_containing(interval)") {
        interval_tree it{std::move(input()), interval_class<>::to_interval};
        it.build_tree();

        auto test = [&it](int start, int end, int expected) {
            int found = 0;
            auto callback = [&found, search = interval<int>{start, end}](interval_class<> const &i) {
                ++found;
                INFO("Search [", search.start_, ", ", search.end_, "] found [", i.interval_.start_, ", ", i.interval_.end_, "]");
                CHECK(i.interval_.contains(search));
            };
            it.find_containing(start, end, callback);
            INFO("Search [", start, ", ", end, "] found ", found, " expected ", expected);
            CHECK_EQ(found, expected);
        };
        test(-1, 0, 0);
        test(1, 2, 1);
        test(6, 15, 1);
        test(1, 23, 0);
        test(11, 23, 0);
        test(0, 44, 0);
        test(21, 26, 1);
    }

    TEST_CASE_FIXTURE(TestFixture, "find_contained(interval)") {
        interval_tree it{std::move(input()), interval_class<>::to_interval};
        it.build_tree();

        auto test = [&it](int start, int end, int expected) {
            int found = 0;
            auto callback = [&found, search = interval<int>{start, end}](interval_class<> const &i) {
                ++found;
                INFO("Search [", search.start_, ", ", search.end_, "] found [", i.interval_.start_, ", ", i.interval_.end_, "]");
                CHECK(search.contains(i.interval_));
            };
            it.find_contained(start, end, callback);
            INFO("Search [", start, ", ", end, "] found ", found, " expected ", expected);
            CHECK_EQ(found, expected);
        };
        test(-1, 0, 0);
        test(1, 2, 0);
        test(1, 15, 1);
        test(1, 23, 2);
        test(5, 23, 1);
        test(0, 44, 3);
    }


    TEST_CASE_FIXTURE(RandomFixture, "Brute force check al find_* methods in large tree") {
#ifdef NDEBUG
        set_interval_cnt(100'000ull);
        static constexpr int interval_spread = 500'000;
        static constexpr int interval_max_len = 100;
#else
        set_interval_cnt(5'000);
        static constexpr int interval_spread = 10'000;
        static constexpr int interval_max_len = 500;
#endif
        interval_class<int>::gen_params.dist_start.
                param(std::uniform_int_distribution<int>::param_type{0, interval_spread});
        interval_class<int>::gen_params.dist_len.param(std::uniform_int_distribution<int>::param_type{
            0, interval_max_len
        });

        auto i = input();
        auto j = i; // save a copy
        auto reduce = [](interval<int> const &acc, interval<int> const &x) -> interval<int> {
            return {std::min(acc.start_, x.start_), std::max(acc.end_, x.end_)};
        };
        auto const transform = interval_class<>::to_interval;
        interval<int> bounds = std::transform_reduce(i.begin(), i.end(),
                                                     interval_class<>::to_interval(i[0]),
                                                     reduce, transform);

        interval_tree it{std::move(j), interval_class<>::to_interval};
        it.build_tree();

        std::mutex my_mutex;
        std::vector<performance_kpis> kpis;
        // TODO: std::ranges::views::iota prevents actual parallel execution of for_each; need to use a real array
        auto range = std::ranges::views::iota(bounds.start_ - 1, bounds.end_ + 1);
        std::for_each(std::execution::par, range.begin(), range.end(),
                      brute_force_test(i, it, my_mutex, kpis)
        );
        // MESSAGE(static_cast<uint32_t>(runs));
        // MESSAGE(elapsed_linear.count());
        // MESSAGE(elapsed_tree.count());
        // MESSAGE(elapsed_tree.count() / elapsed_linear.count() * 100.0 );
    }

}
