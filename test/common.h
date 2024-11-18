//
// Created by arnoldm on 18.10.24.
//

#ifndef COMMON_H
#define COMMON_H

#include <source_location>
#include <interval_tree.h>

template<typename coord_type = int>
struct interval_class {
    static struct RandomTag {
    } random_tag;

    interval_class(interval<coord_type> intvl, coord_type id)
        : interval_{intvl}, id_{id} {
    }

    interval_class() {
        (*this) = generate();
    }

    struct generator_params {
        std::mt19937_64 gen;
        std::uniform_int_distribution<coord_type> dist_start;
        std::uniform_int_distribution<coord_type> dist_len;
        coord_type ID;
    };

    static generator_params gen_params;

    static auto generate(generator_params &p = gen_params) -> interval_class {
        coord_type start = p.dist_start(p.gen);
        coord_type len = p.dist_len(p.gen);
        coord_type id = ++p.ID;
        return {{start, start + len}, id};
    }

    static auto to_interval(interval_class const &i) -> interval<coord_type> {
        return i.interval_;
    }

    interval<coord_type> interval_;
    int id_;
};

template<typename coord_type>
typename interval_class<coord_type>::generator_params interval_class<coord_type>::gen_params{
    .gen = std::mt19937_64{std::random_device{}()},
    .dist_start = std::uniform_int_distribution<coord_type>{0, 100000},
    .dist_len = std::uniform_int_distribution<coord_type>{0, 1000},
    .ID = 0
};

struct TestFixture {
    virtual ~TestFixture() = default;

    using interval_tree_type = interval_tree<interval_class<>, interval<int>(const interval_class<> &)>;
    using interval_id_type = interval_tree_type::interval_id_type;

    void init() {
        auto transformation = [id=0u](auto const &e) mutable -> interval_id_type {
            return {interval_class<int>::to_interval(e), ++id};
        };
        auto inserter = std::back_inserter(starts_);
        auto objs = input();
        std::ranges::transform(objs, inserter, std::move(transformation));
        ends_ = starts_;
        std::ranges::sort(starts_, interval_id_type::start_comparator());
        std::ranges::sort(ends_, interval_id_type::end_comparator());
    }

    virtual std::vector<interval_class<int> > input() {
        return {
            {{1, 10}, 1},
            {{5, 20}, 2},
            {{20, 30}, 3}
        };
    };
    std::vector<interval_id_type> starts_;
    std::vector<interval_id_type> ends_;

    std::span<interval_id_type> starts() {
        return std::span{starts_.begin(), starts_.end()};
    }

    std::span<interval_id_type> ends() {
        return std::span{ends_.begin(), ends_.end()};
    }
};

struct RandomFixture : TestFixture {
    std::vector<interval_class<int> > input() override {
        std::vector<interval_class<int> > result;
        result.resize(interval_cnt_);
        return result;
    }

    size_t interval_cnt_ = 100'000ull;

    [[nodiscard]] size_t interval_cnt() const {
        return interval_cnt_;
    }

    void set_interval_cnt(size_t interval_cnt) {
        interval_cnt_ = interval_cnt;
    }
};

struct performance_kpis {
    struct result {
        uint32_t wrong_ {0};
        uint32_t correct_ {0};
        uint32_t differing_ {0};
        void clear() {
            wrong_ = 0;
            correct_ = 0;
            differing_ = 0;
        }
        result& operator+=(result const & other) {
            wrong_ += other.wrong_;
            correct_ += other.correct_;
            differing_ += other.differing_;
            return *this;
        }
    };
    uint32_t runs_ = 0;
    result intersects_;
    result overlaps_;
    result containing_;
    result contained_;

    std::chrono::duration<double> elapsed_linear_{0};
    std::chrono::duration<double> elapsed_tree_{0};

    [[nodiscard]] inline auto tree_linear_percentage() const noexcept -> double {
        return (elapsed_linear_.count() == 0.) ? 0.0 : elapsed_tree_.count() / elapsed_linear_.count() * 100.0;
    }

    performance_kpis &operator=(performance_kpis const &) = default;

    void clear() {
        runs_ = 0;
        intersects_.clear();
        overlaps_.clear();
        containing_.clear();
        contained_.clear();
        elapsed_linear_ = std::chrono::duration<double>(0);
        elapsed_tree_ = std::chrono::duration<double>(0);
    }

    performance_kpis &operator+=(const performance_kpis &other) {
        runs_ += other.runs_;
        intersects_ += other.intersects_;
        overlaps_ += other.overlaps_;
        containing_ += other.containing_;
        contained_ += other.contained_;
        elapsed_linear_ += other.elapsed_linear_;
        elapsed_tree_ += other.elapsed_tree_;
        return *this;
    }
};

inline
performance_kpis operator+(performance_kpis const &a, performance_kpis &b) {
    return performance_kpis{a} += b;
}

inline
std::ostream &operator<<(std::ostream &os, const performance_kpis &p) {
    int col_a = 30;
    auto print_result = [&](std::string name, performance_kpis::result const &r) {
        std::println(os, "{:{}}: {:7}", name + " wrong", col_a, r.wrong_);
        std::println(os, "{:{}}: {:7}", name + " correct", col_a, r.correct_);
        std::println(os, "{:{}}: {:7}", name + " differing", col_a, r.differing_);
    };
    std::println(os, "{:{}}: {:7}", "runs", col_a, p.runs_);
    print_result("intersects", p.intersects_);
    print_result("overlaps", p.overlaps_);
    print_result("contains", p.containing_);
    print_result("contained", p.contained_);
    std::println(os, "{:{}}: {:11.3f}s", "elapsed_linear", col_a, p.elapsed_linear_.count());
    std::println(os, "{:{}}: {:11.3f}s", "elapsed_tree", col_a, p.elapsed_tree_.count());
    std::println(os, "{:{}}: {:11.3f}%", "tree / linear", col_a, p.tree_linear_percentage());
    return os;
}

inline auto log(std::source_location const &loc) -> void {
    std::println(std::cout, "{}:{}: {}", loc.file_name(), loc.line(), loc.function_name());
}


struct brute_force_test {
    brute_force_test(std::vector<interval_class<int> > const &i,
                     interval_tree<interval_class<>, interval<int>(const interval_class<> &)> &it,
                     std::mutex &mutex, std::vector<performance_kpis> &overall_kpis)
        : i_{i}, it_{it}, mutex_{mutex}, overall_kpis_{overall_kpis} {
        // log(std::source_location::current());
    }

    brute_force_test(const brute_force_test &other)
        : i_{other.i_},
          it_{other.it_},
          mutex_{other.mutex_},
          overall_kpis_{other.overall_kpis_} {
        // log(std::source_location::current());
    }

    brute_force_test(brute_force_test &&other) noexcept
        : i_{other.i_},
          it_{other.it_},
          mutex_{other.mutex_},
          kpis_{other.kpis_},
          overall_kpis_{other.overall_kpis_} {
        // log(std::source_location::current());
        other.kpis_.clear();
    }

    brute_force_test &operator=(const brute_force_test &other) = delete;

    brute_force_test &operator=(brute_force_test &&other) noexcept = delete;

    ~brute_force_test() noexcept {
        // log(std::source_location::current());
        std::scoped_lock lock{mutex_};
        overall_kpis_.push_back(kpis_);
    }

    auto operator()(int x) {
        performance_kpis bf_kpis;
        auto search_overlaps = interval<int>{x, x + 20};
        auto search_containing = interval<int>{x, x + 20};
        auto search_contained = interval<int>{x, x + 500};
        auto t1 = std::chrono::high_resolution_clock::now();
        std::ranges::for_each(
            i_, [&](interval_class<int> const &value)  {
                auto x_contained = value.interval_.contains(x);
                auto overlaps = value.interval_.overlaps(search_overlaps).has_value();
                auto containing = value.interval_.contains(search_containing);
                auto contained = search_contained.contains(value.interval_);
                bf_kpis.intersects_.correct_ += x_contained;
                bf_kpis.intersects_.wrong_ += !x_contained;
                bf_kpis.overlaps_.correct_ += overlaps;
                bf_kpis.overlaps_.wrong_ += !overlaps;
                bf_kpis.containing_.correct_ += containing;
                bf_kpis.containing_.wrong_ += !containing;
                bf_kpis.contained_.correct_ += contained;
                bf_kpis.contained_.wrong_ += !contained;
            });
        auto t2 = std::chrono::high_resolution_clock::now();

        it_.find_intersecting(x, [this, x](interval_class<int> const &value) {
            auto x_contained = value.interval_.contains(x);
#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
            INFO("Search ", x, "] found intersecting [", value.interval_.start_, ", ", value.interval_.end_, "]");
            CHECK(x_contained);
#endif
            kpis_.intersects_.correct_ += x_contained;
            kpis_.intersects_.wrong_ += !x_contained;
        });

        it_.find_overlapping(search_overlaps.start_, search_overlaps.end_,
            [this, search_overlaps](interval_class<int> const &value) {
            auto overlapping = value.interval_.overlaps(search_overlaps).has_value();
#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
            INFO("Search [", search_overlaps.start_, ", ", search_overlaps.end_, "] found overlaps [", value.interval_.start_, ", ", value.interval_.end_, "]");
            CHECK(overlapping);
#endif
            kpis_.overlaps_.correct_ += overlapping;
            kpis_.overlaps_.wrong_ += !overlapping;
        });

        it_.find_containing(search_containing.start_, search_containing.end_,
            [this, search_containing](interval_class<int> const &value) {
            auto containing = value.interval_.contains(search_containing);
#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
            INFO("Search [", search_containing.start_, ", ", search_containing.end_, "] found containing [", value.interval_.start_, ", ", value.interval_.end_, "]");
            CHECK(containing);
#endif
            kpis_.containing_.correct_ += containing;
            kpis_.containing_.wrong_ += !containing;
        });

        it_.find_contained(search_contained.start_, search_contained.end_,
            [this, search_contained](interval_class<int> const &value) {
            auto contained = search_contained.contains(value.interval_);
#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
            INFO("Search [", search_contained.start_, ", ", search_contained.end_, "] found contained [", value.interval_.start_, ", ", value.interval_.end_, "]");
            CHECK(contained);
#endif
            kpis_.contained_.correct_ += contained;
            kpis_.contained_.wrong_ += !contained;
        });
        auto t3 = std::chrono::high_resolution_clock::now();

        kpis_.elapsed_linear_ += std::chrono::duration<double>(t2 - t1);
        kpis_.elapsed_tree_ += std::chrono::duration<double>(t3 - t2);
        ++kpis_.runs_;
    }

    std::vector<interval_class<int> > const &i_;
    interval_tree<interval_class<>, interval<int>(const interval_class<> &)> const &it_;
    std::mutex &mutex_;
    performance_kpis kpis_;
    std::vector<performance_kpis> &overall_kpis_;
    int contained_search_len_ { 500 };
    int containing_search_len_ { 20 };
};

#endif //COMMON_H
