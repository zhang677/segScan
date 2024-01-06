#include <iostream>
#include <vector>
#include <random>
#include <cmath>

std::vector<int> generateVector(int range, int max_seg, double avg) {
    // range: Elements of index `i` in [0, range)
    // max_seg: Maximum repetition `mi` of each element of index `i`
    // avg: Desired average of `mi`
    int total_count = static_cast<int>(range * avg);
    std::vector<int> result;
    result.reserve(total_count);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(max_seg / 2.0, max_seg / 4.0); // mean max_seg/2, std dev max_seg/4

    int current_sum = 0;
    for (int i = 0; i < range; ++i) {
        int mi = static_cast<int>(std::round(distribution(generator)));

        // Ensure mi is within bounds
        if (mi < 0) mi = 0;
        if (mi >= max_seg) mi = max_seg - 1;

        // Adjust the last element to match the desired total count
        if (i == range - 1) {
            mi = total_count - current_sum;
            if (mi < 0) mi = 0; // Ensure mi is not negative
            if (mi >= max_seg) mi = max_seg - 1; // Ensure mi is within bounds
        }

        for (int j = 0; j < mi; ++j) {
            result.push_back(i);
        }
        current_sum += mi;

        // Early exit if we reached the total count
        if (current_sum >= total_count) break;
    }
    return result;
}

int main() {
    int range = 6; // range of numbers [0, range)
    int max_seg = 4; // max_segaximum repetition
    double avg = 1.67; // Desired average of mi

    std::vector<int> vec = generateVector(range, max_seg, avg);

    for (int num : vec) {
        std::cout << num << " ";
    }
    return 0;
}