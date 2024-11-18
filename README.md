# `interval_tree`

A library implementing a generic centered
[interval tree](https://en.wikipedia.org/wiki/Interval_tree).

I separated this data structure from a project where I needed a range lookup.
It supports the following search operations:

* `find_intersecting`: find all intervals which intersect (contain) a coordinate
* `find_overlapping`: find all intervals which overlap a given interval
* `find_containing`: find all intervals which contain a given interval
* `find_contained`: find all intervals which are contained by a given interval

## Synopsis

```c++
#include <interval_tree.h>

struct my_type {
    using value_type = int;
    value_type start_;
    value_type len_;
    std::string name_;
};

interval<my_type::value_type> to_interval(my_type const & mt) {
    return { .start_ = mt.start_, .end_ = mt.start_ + mt.len_ - 1 };
}

int main() {
    std::vector<my_type> my_type_list;
    // ...fill my_type_list
    interval_tree tree(std::move(my_type_list), to_interval);
    tree.build_tree();
    
    my_type::value_type x = 42;
    auto callback = [x](my_type const & mt) {
        std::cout << "Found " << mt.name_ << " for x = " << x << '\n';
    };
    tree.find_intersecting(x, std::move(callback));
}
```

## Performance

The project contains `itperftest`, a performance testing tool. 
It generates a sample set of intervals and queries for coords 
the containing intervals both by doing a linear scan over the
full set of intervals as the baseline and then using the 
`interval_tree<>::find_intersecting()` API.
You will find it in the build-folder. 
```bash
&> ./itperftest
Timer Total search time:       2.764s
interval_cnt                  :  500000
interval_spread               : 1000000
interval_max_len              :     100
samples_cnt                   :  100000
start                         :       7
end                           : 1000094
elapsed_build                 :       0.488s
runs                          :  100000
wrong_intervals               :       0
correct_intervals             : 2548413
differing_cnts                :       0
elapsed_linear                :      33.058s
elapsed_tree                  :       0.078s
tree / linear                 :       0.237%
```
The bottom most figure *'tree / linear'* is the percentage of runtime
of the interval tree in terms of the linear search. The linear
search is implemented using `std::ranges::fold_left`.

## Todo - Things to come

Currently, the structure is static. An interface to update the contained intervals
shall follow.

    [X] bulk insert and build tree
    [X] integral coordinates
    [X] tree statistics interface
    [X] find_intersecting intervals given a coordinate (callback interface)
    [X] find_intersecting intervals given an interval
    [X] find_contained intervals given an interval
    [X] find_containing intervals given an interval
    [ ] test with floating point coordinates
    [ ] insert intervals
    [ ] delete/remove intervals
    [ ] iterator interface for find functions
    [ ] persistence (?)
