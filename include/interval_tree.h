//
// Created by arnoldm on 05.10.24.

#ifndef SEGMENTTREE_H
#define SEGMENTTREE_H
#include <limits>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <span>
#include <unordered_set>
#include <numeric>
#include <ostream>
#include <ranges>

/**
 * @brief Return the absolut difference of unsigned integrals, avoiding pitfalls of numeric overflow
 * @tparam T An integral type
 * @param a minuend
 * @param b subtrahend
 * @return the absolute difference of a and b
 */
template<std::integral T>
auto absolute_unsigned_difference(T const &a, T const &b) -> T {
    return std::max(a, b) - std::min(a, b);
}

/**
 * @brief A class template defining an interval with start and end inclusive
 *
 * start is supposed to be <= end
 * @tparam T the value_type of start and end
 */
template<typename T>
struct interval {
    using value_type = T;

    explicit operator std::string() const {
        return std::format("[{}; {}]", start_, end_);
    }

    friend std::ostream &operator<<(std::ostream &os, interval const &interval) {
        return os << interval.operator std::string();
    }

    auto contains(value_type const &x) const noexcept -> bool {
        return start_ <= x && x <= end_;
    }

    auto contains(interval const &other) const noexcept -> bool {
        return start_ <= other.start_ && end_ >= other.end_;
    }

    auto overlaps(interval const &other) const noexcept -> std::optional<interval> {
        auto max_start = std::max(start_, other.start_);
        auto min_end = std::min(other.end_, other.end_);
        return max_start <= min_end
                   ? std::optional<interval>{{.start_ = max_start, .end_ = min_end}}
                   : std::optional<interval>{};
    }

    T start_;
    T end_;
};

template<typename T>
auto make_interval(T&& start, T&& end) -> interval<T> {
    return {std::forward<T>(start), std::forward<T>(end)};
}

/**
 * A centered interval tree.
 * @tparam T
 */
template<typename T, typename Projection = std::identity,
    typename Coord_type = typename decltype(std::declval<Projection>()(std::declval<T>()))::value_type,
    typename Index_type = unsigned>
    requires std::invocable<Projection, T> && std::convertible_to<decltype(std::declval<Projection>()(std::declval<T>())
             ), interval<Coord_type> >
class interval_tree {
public:
    using this_type = interval_tree;
    using value_type = T;
    using coord_type = Coord_type;
    using interval_type = interval<coord_type>;
    using index_type = Index_type;

    struct interval_id_type {
        static auto start_comparator() {
            return [](interval_id_type const &a, interval_id_type const &b) {
                return std::tie(a.interval_.start_, a.interval_.end_) < std::tie(b.interval_.start_, b.interval_.end_);
            };
        }

        static auto end_comparator() {
            return [](interval_id_type const &a, interval_id_type const &b) {
                return std::tie(a.interval_.end_, a.interval_.start_) < std::tie(b.interval_.end_, b.interval_.start_);
            };
        }

        static auto start_projection() {
            return [](interval_id_type const& i) { return i.interval_.start_; };
        }
        static auto end_projection() {
            return [](interval_id_type const& i) { return i.interval_.end_; };
        }

        interval_type interval_;
        index_type index_;
    };

    using projection_type = Projection; // Project from value_type to interval

    explicit interval_tree(std::vector<T> &&values, Projection &&projection = Projection())
        : values_(std::move(values)), projection_(std::move(projection)) {
    }

    // protected:
    struct midpoint_result {
        static auto extract_indexes(std::span<interval_id_type> const &values,
                                    std::unordered_set<index_type> &&reuse = {}) {
            std::unordered_set<index_type> result = std::move(reuse);
            result.clear();
            result.reserve(values.size());
            std::ranges::for_each(values, [&result](interval_id_type const &value) {
                result.insert(value.index_);
            });
            return result;
        }

        [[nodiscard]] auto ends_first_not_left() const -> typename std::span<interval_id_type>::iterator {
            return ends_first_not_left_;
        }

        [[nodiscard]] auto starts_last_not_right() const -> typename std::span<interval_id_type>::iterator {
            auto result = starts_first_to_right_;
            std::ranges::advance(result, -1, starts_.begin());
            return result;
        }

        [[nodiscard]] std::span<interval_id_type> ends_to_left() const {
            return {ends_.begin(), ends_first_not_left()};
        }

        [[nodiscard]] std::span<interval_id_type> ends_intersecting_or_to_right() const {
            return {ends_first_not_left(), ends_.end()};
        }

        [[nodiscard]] std::span<interval_id_type> starts_to_right() const {
            return {starts_first_to_right_, starts_.end()};
        }

        [[nodiscard]] std::span<interval_id_type> starts_intersecting_or_to_left() const {
            return {starts_.begin(), starts_first_to_right_};
        }

        std::span<interval_id_type> const &starts_;
        std::span<interval_id_type> const &ends_;
        coord_type midpoint_; // value of the midpoint
        typename std::span<interval_id_type>::iterator starts_first_to_right_; // first start > midpoint
        typename std::span<interval_id_type>::iterator ends_first_not_left_; // first end >= midpoint
        typename std::span<interval_id_type>::iterator ends_last_to_left_; // last end < midpoint
        index_type cnt_to_left_; // intervals left of midpoint
        index_type cnt_to_right_; // intervals right of midpoint
        index_type cnt_intersecting_; // intervals intersecting midpoint
    };

    /**
     * Find a center point p for the intervals represented by the input lists such that the number of
     * intervals which end before p is approximately equal to the number of intervals which start after p;
     * note that there is a third group of intervals intersecting p, i.e. start <= p <= end
     * @param starts the list of interval_id_type ordered by start_ (ascending)
     * @param ends the list of interval_id_type ordered by end_ (ascending)
     * @return a midpoint which splits the intervals
     */
    static auto find_midpoint(std::span<interval_id_type> const &starts,
                              std::span<interval_id_type> const &ends) -> midpoint_result {
        assert(starts.size() == ends.size());
        assert(!starts.empty());
        coord_type min = starts.front().interval_.start_;
        coord_type max = ends.back().interval_.end_;
        midpoint_result result{.starts_ = starts, .ends_ = ends};
        auto starts_begin = starts.begin();
        auto starts_end = starts.end();
        auto ends_begin = ends.begin();
        auto ends_end = ends.end();

        do {
            result.midpoint_ = (min + max) / 2;

            // find intervals which only start right from mid and thus cannot intersect mic
            result.starts_first_to_right_ = std::ranges::upper_bound(starts, result.midpoint_,
                                                             std::less{}, interval_id_type::start_projection());
            // find intervals which end before mid
            result.ends_first_not_left_ = std::ranges::lower_bound(ends, result.midpoint_,
                                                           std::less{}, interval_id_type::end_projection());
            result.ends_last_to_left_ = result.ends_first_not_left_;
            // now ends_last_to_left_  !< midpoint_; i.e. we need to go back to the first element which is strictly < midpoint_
            while (result.ends_last_to_left_ != ends.begin() && !(
                       result.ends_last_to_left_->interval_.end_ < result.midpoint_))
                --result.ends_last_to_left_;

            result.cnt_to_right_ = static_cast<index_type>(std::distance(result.starts_first_to_right_, starts.end()));
            result.cnt_to_left_ = static_cast<index_type>(std::distance(ends.begin(), result.ends_first_not_left_));
            result.cnt_intersecting_ = static_cast<unsigned>(
                starts.size() - result.cnt_to_right_ - result.cnt_to_left_);
            if (absolute_unsigned_difference(result.cnt_to_left_, result.cnt_to_right_) <= 1u) {
                // we have an equal split of intervals -> close enough to perfect
                break;
            } else if (result.cnt_to_left_ < result.cnt_to_right_) {
                min = result.midpoint_;
                starts_begin = result.starts_first_to_right_;
                ends_begin = result.ends_first_not_left_;
            } else if (result.cnt_to_left_ > result.cnt_to_right_) {
                max = result.midpoint_;
                starts_end = result.starts_first_to_right_;
                ends_end = result.ends_first_not_left_;
            }
        } while (starts_begin < starts_end && ends_begin < ends_end && (min + 1 < max));
        return result;
    }

    static constexpr index_type INVALID_INDEX = std::numeric_limits<index_type>::max();

    /**
     * @brief an internal node of the interval tree
     */
    struct node {
        [[nodiscard]] auto has_left() const -> bool { return left_node_index_ != INVALID_INDEX; }
        [[nodiscard]] auto has_right() const -> bool { return right_node_index_ != INVALID_INDEX; }
        [[nodiscard]] auto is_leaf() const -> bool { return !(has_left() || has_right()); }

        // index_type index_;
        coord_type center_;
        index_type parent_node_index_{INVALID_INDEX};
        index_type left_node_index_{INVALID_INDEX};
        index_type right_node_index_{INVALID_INDEX};
        std::vector<interval_id_type> starts_intersecting_;
        std::vector<interval_id_type> ends_intersecting_;
    };

    /**
     * Build the tree recursively
     * @param nodes_ the nodes list
     * @param starts the list of interval_id_type ordered by start_ (ascending)
     * @param ends the list of interval_id_type ordered by end_ (ascending)
     * @return the index of the newly inserted nodes
     */
    static auto build_node(std::vector<node> &nodes_, std::span<interval_id_type> const &starts,
                           std::span<interval_id_type> const &ends,
                           index_type const &parent_node_index = INVALID_INDEX) -> index_type {
        thread_local std::unordered_set<index_type> indexes_to_left;
        thread_local std::unordered_set<index_type> indexes_to_right;

        auto midpoint = find_midpoint(starts, ends);

        // stable partition starts into (to left <> intersecting <> to right)
        indexes_to_left = midpoint_result::extract_indexes(midpoint.ends_to_left(), std::move(indexes_to_left));
        auto starts_first_intersecting = std::stable_partition(starts.begin(), midpoint.starts_first_to_right_,
                                                               [](interval_id_type const &value) {
                                                                   return indexes_to_left.contains(value.index_);
                                                               });
        indexes_to_right = midpoint_result::extract_indexes(midpoint.starts_to_right(), std::move(indexes_to_right));
        std::span<interval_id_type> starts_intersecting = std::span(starts_first_intersecting,
                                                                    midpoint.starts_first_to_right_);

        // stable partition ends into (to left <> intersecting <> to right )
        auto ends_first_not_intersecting = std::stable_partition(midpoint.ends_first_not_left(), ends.end(),
                                                                 [](interval_id_type const &value) {
                                                                     return !indexes_to_right.contains(value.index_);
                                                                 });
        std::span<interval_id_type> ends_intersecting = std::span(midpoint.ends_first_not_left(),
                                                                  ends_first_not_intersecting);

        nodes_.emplace_back(midpoint.midpoint_, parent_node_index);
        index_type new_node_index = static_cast<index_type>(nodes_.size()) - 1;
        nodes_[new_node_index].starts_intersecting_.reserve(midpoint.cnt_intersecting_);
        std::ranges::copy(starts_intersecting, std::back_inserter(nodes_[new_node_index].starts_intersecting_));
        nodes_[new_node_index].ends_intersecting_.reserve(midpoint.cnt_intersecting_);
        std::ranges::copy(ends_intersecting, std::back_inserter(nodes_[new_node_index].ends_intersecting_));

        if (midpoint.cnt_to_left_ > 0) {
            auto starts_left = std::span(starts.begin(), midpoint.cnt_to_left_);
            auto left_node_index = build_node(nodes_, starts_left, midpoint.ends_to_left(), new_node_index);
            nodes_[new_node_index].left_node_index_ = left_node_index;
        }
        if (midpoint.cnt_to_right_ > 0) {
            auto ends_right = std::span(ends_first_not_intersecting, ends.end());
            auto right_node_index = build_node(nodes_, midpoint.starts_to_right(), ends_right, new_node_index);
            nodes_[new_node_index].right_node_index_ = right_node_index;
        }
        return new_node_index;
    }

public:
    auto build_tree() -> void {
        std::vector<interval_id_type> starts;
        starts.reserve(values_.size());
        std::vector<interval_id_type> ends;
        ends.reserve(values_.size());
        for (unsigned i = 0; i < static_cast<unsigned>(values_.size()); ++i) {
            auto const &value = values_[i];
            interval<coord_type> projected_interval = projection_(value);
            starts.emplace_back(projected_interval, i);
            ends.emplace_back(projected_interval, i);
        }
        std::ranges::sort(starts, interval_id_type::start_comparator());
        std::ranges::sort(ends, interval_id_type::end_comparator());
        root_index_ = build_node(nodes_, std::span{starts.begin(), starts.size()},
                                 std::span{ends.begin(), ends.size()});
    }

    /**
     * @brief The number of values in this tree
     * @return the number of values
     */
    auto size() const noexcept { return values_.size(); }

    /**
     * @brief Access the projection function which converts any value_type to an interval
     * @return the projection function
     */
    auto projection() const -> projection_type const & { return projection_; }

    /**
     * @brief Indicate in which direction tree traversal shall continue
     */
    enum class traversal {
        stop = 0, // stop traversal here
        left = 1, // traverse left subtree
        right = 2, // traverse right subtree
        both = 3 // travers both left and right subtree
    };


    /**
     * @brief Query all values which intersect a coordinate
     * @param location the coordinate
     * @param callback will be called for all value_types which intersect the coordinate
     */
    auto find_intersecting(coord_type const &location, auto &&callback) const -> void
        requires requires(value_type const &v)
        {
            { callback(v) } -> std::same_as<void>;
        } {
        auto find_intersecting_impl =
                [this, callback = std::forward<decltype(callback)>(callback), &location](
            node const &current_node) -> traversal {
            if (location == current_node.center_) {
                for (auto const &e: current_node.starts_intersecting_)
                    callback(values_[e.index_]);
                return traversal::stop;
            } else if (location < current_node.center_) {
                for (auto const &e: current_node.starts_intersecting_) {
                    if (e.interval_.start_ > location)
                        break;
                    callback(values_[e.index_]);
                }
                return traversal::left;
            } else {
                // location > current_node.center_
                auto first_intersecting = std::ranges::lower_bound(current_node.ends_intersecting_, location,
                                                                   std::less{},
                                                                   [](interval_id_type const &value) {
                                                                       return value.interval_.end_;
                                                                   });
                for (auto const &e: std::span(first_intersecting, current_node.ends_intersecting_.end()))
                    callback(values_[e.index_]);
                return traversal::right;
            }
            [[unlikely]] return traversal::stop;
        };
        accept(find_intersecting_impl, root_index_);
    }

    /**
     * @brief Find all entries which overlapp the search interval
     * @param start of the search interval
     * @param end of the search interval
     * @param callback
     */
    auto find_overlapping(coord_type const &start, coord_type const &end, auto &&callback) const -> void
        requires requires(value_type const &v)
        {
            { callback(v) } -> std::same_as<void>;
        } {
        auto interval = interval_type{.start_ = start, .end_ = end};
        auto find_overlapping_impl = [this, search = interval, callback = std::forward<decltype(callback)>(callback)
                ](node const &current_node) -> traversal {
            auto contains_center = search.contains(current_node.center_);
            auto is_left_from_center = search.end_ < current_node.center_;
            auto is_right_from_center = current_node.center_ < search.start_;
            if (contains_center) {
                for (auto const &value: current_node.starts_intersecting_)
                    callback(values_[value.index_]);
                return traversal::both;
            }
            if (is_left_from_center) {
                for (auto const &value: current_node.starts_intersecting_) {
                    if (value.interval_.start_ > search.end_)
                        break;
                    callback(values_[value.index_]);
                }
                return traversal::left;
            }
            if (is_right_from_center) {
                auto first_overlapping = std::ranges::lower_bound(current_node.ends_intersecting_, search.start_,
                                                                  std::less{}, interval_id_type::end_projection());
                for (auto const &value: std::span(first_overlapping, current_node.ends_intersecting_.end()))
                    callback(values_[value.index_]);
                return traversal::right;
            }
            [[unlikely]] return traversal::stop;
        };
        accept(find_overlapping_impl, root_index_);
    }

    /**
     * @brief Find all entries which contain the search interval (start <= search.start && search.end <= end)
     * @param start start of search interval
     * @param end end of search interval
     * @param callback
     */
    auto find_containing(coord_type const &start, coord_type const &end, auto &&callback) const -> void
        requires requires(value_type const &v)
        {
            { callback(v) } -> std::same_as<void>;
        } {
        auto interval = interval_type{.start_ = start, .end_ = end};
        auto find_containing_impl = [this, search = interval, callback = std::forward<decltype(callback)>(callback)
                ](node const &current_node) -> traversal {
            auto contains_center = search.contains(current_node.center_);
            auto is_left_from_center = search.end_ < current_node.center_;
            auto is_right_from_center = current_node.center_ < search.start_;
            if (contains_center) {
                for (auto const &value: current_node.starts_intersecting_) {
                    if (value.interval_.start_ > search.start_)
                        break;
                    if (search.end_ <= value.interval_.end_)
                        callback(values_[value.index_]);
                }
                return traversal::both;
            }
            if (is_left_from_center) {
                for (auto const &value: current_node.starts_intersecting_) {
                    if (value.interval_.start_ > search.start_)
                        break;
                    callback(values_[value.index_]);
                }
                return traversal::left;
            }
            if (is_right_from_center) {
                for (auto const &value: current_node.ends_intersecting_) {
                    if (value.interval_.end_ < search.end_)
                        break;
                    callback(values_[value.index_]);
                }
                return traversal::right;
            }
            [[unlikely]] return traversal::stop;
        };
        accept(find_containing_impl, root_index_);
    }

    /**
     * @brief Find all entries contained by the search interval (start >= search.start && search.end >= end)
     * @param start start of search interval
     * @param end end of search interval
     * @param callback
     */
    auto find_contained(coord_type const &start, coord_type const &end, auto &&callback) const -> void requires
        requires(value_type const &v)
        {
            { callback(v) } -> std::same_as<void>;
        } {
        auto interval = interval_type{.start_ = start, .end_ = end};
        auto find_contained_impl = [this, search = interval, callback = std::forward<decltype(callback)>(callback)
                ](node const &current_node) -> traversal {
            auto contains_center = search.contains(current_node.center_);
            auto is_left_from_center = search.end_ < current_node.center_;
            auto is_right_from_center = current_node.center_ < search.start_;
            if (contains_center) {
                auto first_contained = std::ranges::lower_bound(current_node.starts_intersecting_, search.start_,
                                                                std::less{}, interval_id_type::start_projection());
                // Could also search for last_contained in ends and iterate over the array with less candidates ...
                // for now do a forward scan through the starts
                for (auto const &value: std::span(first_contained, current_node.starts_intersecting_.end())) {
                    if (value.interval_.end_ <= search.end_)
                        callback(values_[value.index_]);
                }
                return traversal::both;
            }
            if (is_left_from_center) {
                return traversal::left;
            }
            if (is_right_from_center) {
                return traversal::right;
            }
            [[unlikely]] return traversal::stop;
        };
        accept(find_contained_impl, root_index_);
    }

    struct statistics {
        index_type node_count_ = 0;
        index_type values_count_ = 0;
        index_type leaf_nodes_ = 0;
        index_type empty_nodes_ = 0;
        index_type max_depth_ = 0;
        index_type max_values_per_node_ = 0;

        [[nodiscard]] float avg_values_per_node() const {
            return static_cast<float>(values_count_) / static_cast<float>(node_count_);
        };
        index_type min_values_per_node_ = std::numeric_limits<index_type>::max();
        float pcnt01_values_per_node_ = 0;
        float pcnt1_values_per_node_ = 0;
        float median_values_per_node_ = 0;
        float pcnt99_values_per_node_ = 0;
        float pcnt999_values_per_node_ = 0;
        float std_values_per_node_ = 0;
        float leaf_node_skew_ = 0; // sum(depth(leave(i))) / ( max_depth * nr_of_leaves)

        friend std::ostream &operator<<(std::ostream &os, statistics const &stats) {
            static constexpr size_t col1_width = 30;
            static constexpr size_t col2_width = 7;
            static constexpr char const *const fmt_string_int = "{:{}}: {:{}}";
            static constexpr char const *const fmt_string_float = "{:{}}: {:{}.3f}";
            std::println(os, fmt_string_int, "values_count_", col1_width, stats.values_count_, col2_width);
            std::println(os, fmt_string_int, "node_count_", col1_width, stats.node_count_, col2_width);
            std::println(os, fmt_string_int, "leaf_nodes_", col1_width, stats.leaf_nodes_, col2_width);
            std::println(os, fmt_string_float, "leaf_node_skew_", col1_width, stats.leaf_node_skew_, col2_width);
            std::println(os, fmt_string_int, "empty_nodes_", col1_width, stats.leaf_nodes_, col2_width);
            std::println(os, fmt_string_int, "max_depth_", col1_width, stats.max_depth_, col2_width);
            std::println(os, fmt_string_int, "max_values_per_node_", col1_width, stats.max_values_per_node_,
                         col2_width);
            std::println(os, fmt_string_float, "avg_values_per_node", col1_width, stats.avg_values_per_node(),
                         col2_width);
            std::println(os, fmt_string_int, "min_values_per_node_", col1_width, stats.min_values_per_node_,
                         col2_width);
            std::println(os, fmt_string_float, "pcnt0,1_values_per_node_", col1_width, stats.pcnt01_values_per_node_,
                         col2_width);
            std::println(os, fmt_string_float, "pcnt1_values_per_node_", col1_width, stats.pcnt1_values_per_node_,
                         col2_width);
            std::println(os, fmt_string_float, "median_values_per_node_", col1_width, stats.median_values_per_node_,
                         col2_width);
            std::println(os, fmt_string_float, "pcnt99_values_per_node_", col1_width, stats.pcnt99_values_per_node_,
                         col2_width);
            std::println(os, fmt_string_float, "pcnt99,9_values_per_node_", col1_width, stats.pcnt999_values_per_node_,
                         col2_width);
            std::println(os, fmt_string_float, "std_values_per_node_", col1_width, stats.std_values_per_node_,
                         col2_width);
            return os;
        }
    };

protected:
    static auto find_depth(this_type const &tree, std::vector<index_type> &node_depths,
                           const index_type &node_index) -> index_type {
        if (node_index == INVALID_INDEX)
            return 0;
        if (node_depths[node_index] == INVALID_INDEX) {
            node_depths[node_index] = find_depth(tree, node_depths, tree.nodes_[node_index].parent_node_index_) + 1;
        }
        return node_depths[node_index];
    }

public:
    auto tree_statistics() const -> statistics {
        statistics stats;
        stats.node_count_ = static_cast<index_type>(nodes_.size());
        stats.values_count_ = static_cast<index_type>(values_.size());

        std::vector<index_type> node_depths;
        node_depths.resize(nodes_.size(), INVALID_INDEX);
        std::vector<index_type> node_values;
        node_values.resize(nodes_.size(), 0);
        index_type leaf_depth_sum = 0;

        for (index_type i = 0; i < static_cast<index_type>(nodes_.size()); ++i) {
            auto const &node = nodes_[i];
            auto depth = find_depth(*this, node_depths, i);
            stats.max_depth_ = std::max(depth, stats.max_depth_);
            auto values_cnt = static_cast<index_type>(node.starts_intersecting_.size());
            node_values[i] = values_cnt;
            stats.empty_nodes_ += values_cnt == 0;
            stats.min_values_per_node_ = std::min(values_cnt, stats.min_values_per_node_);
            stats.max_values_per_node_ = std::max(values_cnt, stats.max_values_per_node_);
            bool is_leaf = node.is_leaf();
            stats.leaf_nodes_ += (is_leaf ? 1 : 0);
            leaf_depth_sum += is_leaf * depth;
        }
        stats.leaf_node_skew_ = static_cast<float>(leaf_depth_sum) / (static_cast<float>(stats.leaf_nodes_) *
                                                                      static_cast<float>(stats.max_depth_));
        {
            // compute standard derivation of values per node
            auto sqrdiff = [avg = stats.avg_values_per_node()](auto const &v) {
                auto fv = static_cast<float>(v) - avg;
                return fv * fv;
            };
            stats.std_values_per_node_ = std::sqrt(
                std::transform_reduce(node_values.cbegin(), node_values.cend(), 0.f, std::plus{}, sqrdiff)
                / static_cast<float>(stats.node_count_)
                );
        }

        auto get_percentile = [&node_values, this](float percent) -> float {
            percent = std::clamp(percent, 0.f, 1.f);
            auto index_a = static_cast<index_type>(std::floor(static_cast<float>(nodes_.size() - 1) * percent));
            auto index_b = static_cast<index_type>(std::ceil(static_cast<float>(nodes_.size() - 1) * percent));
            std::ranges::nth_element(node_values, node_values.begin() + index_a);
            std::span ge_span(node_values.begin() + index_b, node_values.end());
            std::ranges::nth_element(ge_span, ge_span.begin());
            return (static_cast<float>(*(node_values.begin() + index_a)) + static_cast<float>(*ge_span.begin())) / 2.f;
        };
        stats.median_values_per_node_ = get_percentile(0.5f);
        stats.pcnt01_values_per_node_ = get_percentile(0.001f);
        stats.pcnt1_values_per_node_ = get_percentile(0.01f);
        stats.pcnt99_values_per_node_ = get_percentile(0.99f);
        stats.pcnt999_values_per_node_ = get_percentile(0.999f);
        return stats;
    }

protected:
    /**
     * @brief An acceptor function for visitors to implement traversing algorithms.
     * The visitor needs to return stop, left, right, or both to indicate
     * how traversal shall continue.
     * @tparam F a function traversal(node const &)
     * @param visitor the visitor of type F
     * @param node_index
     */
    template<std::invocable<node> F>
        requires std::same_as<std::invoke_result_t<F, node>, traversal>
    auto accept(F const &visitor, index_type node_index) const -> void {
        if (node_index == INVALID_INDEX)
            return;
        assert(node_index < nodes_.size());
        node const &current_node = nodes_.at(node_index);
        auto continuation = std::invoke(visitor, current_node);
        std::array<index_type, 2> next_indexes{INVALID_INDEX, INVALID_INDEX};
        index_type next_indexes_size = index_type(0);
        if (continuation == traversal::left || continuation == traversal::both) {
            next_indexes[next_indexes_size++] = current_node.left_node_index_;
        }
        if (continuation == traversal::right || continuation == traversal::both) {
            next_indexes[next_indexes_size++] = current_node.right_node_index_;
        }
        for (auto const &index: std::span(next_indexes.begin(), next_indexes_size)) {
            accept(visitor, index);
        }
    }

private:
    std::vector<T> values_;
    projection_type &&projection_;
    std::vector<node> nodes_;
    index_type root_index_ = INVALID_INDEX;
    std::vector<std::vector<std::pair<coord_type, index_type> > > starts_;
    std::vector<std::vector<std::pair<coord_type, index_type> > > ends_;
};

#endif //SEGMENTTREE_H
