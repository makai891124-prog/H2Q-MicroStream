#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

struct Result {
    std::string name;
    std::uint32_t limit = 0;
    std::uint64_t count = 0;
    std::uint64_t elapsed_ms = 0;
    double memory_mb = 0.0;
};

std::uint32_t isqrt(std::uint32_t value) {
    std::uint32_t left = 0;
    std::uint32_t right = value;
    std::uint32_t answer = 0;
    while (left <= right) {
        const std::uint32_t mid = left + (right - left) / 2U;
        const std::uint64_t square = static_cast<std::uint64_t>(mid) * mid;
        if (square <= value) {
            answer = mid;
            left = mid + 1U;
        } else {
            if (mid == 0U) {
                break;
            }
            right = mid - 1U;
        }
    }
    return answer;
}

std::uint64_t run_full_byte_sieve(std::uint32_t limit, std::size_t& memory_bytes) {
    std::vector<std::uint8_t> composite(static_cast<std::size_t>(limit) + 1ULL, 0U);
    memory_bytes = composite.size();
    if (limit < 2U) {
        return 0;
    }

    const std::uint32_t bound = isqrt(limit);
    for (std::uint32_t p = 2U; p <= bound; ++p) {
        if (composite[p] != 0U) {
            continue;
        }
        const std::uint64_t start = static_cast<std::uint64_t>(p) * p;
        for (std::uint64_t value = start; value <= limit; value += p) {
            composite[static_cast<std::size_t>(value)] = 1U;
        }
    }

    std::uint64_t count = 0;
    for (std::uint32_t value = 2U; value <= limit; ++value) {
        count += (composite[value] == 0U) ? 1ULL : 0ULL;
    }
    return count;
}

std::uint64_t run_odd_byte_sieve(std::uint32_t limit, std::size_t& memory_bytes) {
    if (limit < 2U) {
        memory_bytes = 0;
        return 0;
    }

    const std::size_t odd_count = (limit >= 3U) ? static_cast<std::size_t>((limit - 3U) / 2U) + 1ULL : 0ULL;
    std::vector<std::uint8_t> composite(odd_count, 0U);
    memory_bytes = composite.size();

    const std::uint32_t bound = isqrt(limit);
    for (std::size_t idx = 0; idx < odd_count; ++idx) {
        if (composite[idx] != 0U) {
            continue;
        }
        const std::uint32_t p = static_cast<std::uint32_t>(idx * 2ULL + 3ULL);
        if (p > bound) {
            break;
        }
        std::uint64_t start = static_cast<std::uint64_t>(p) * p;
        std::size_t composite_idx = static_cast<std::size_t>((start - 3ULL) / 2ULL);
        while (composite_idx < odd_count) {
            composite[composite_idx] = 1U;
            composite_idx += p;
        }
    }

    std::uint64_t count = 1ULL;
    for (std::uint8_t mark : composite) {
        count += (mark == 0U) ? 1ULL : 0ULL;
    }
    return count;
}

class WheelSieve6 {
public:
    explicit WheelSieve6(std::uint32_t limit)
        : limit_(limit),
          candidate_count_(compute_candidate_count(limit)),
          bits_(words_for_bits(candidate_count_), 0ULL) {}

    void run() {
        const std::size_t sieve_bound = compute_candidate_count(isqrt(limit_));
        for (std::size_t idx = 0; idx < sieve_bound; ++idx) {
            if (is_marked(idx)) {
                continue;
            }

            const std::uint64_t p = candidate_to_number(idx);
            const std::uint64_t p2 = p * p;
            if (p2 > limit_) {
                break;
            }

            std::size_t composite_idx = square_to_index(p2);
            std::size_t first_step = 0;
            std::size_t second_step = 0;
            if ((idx & 1ULL) == 0ULL) {
                first_step = idx * 2ULL + 3ULL;
                second_step = idx * 4ULL + 7ULL;
            } else {
                first_step = idx * 4ULL + 5ULL;
                second_step = idx * 2ULL + 3ULL;
            }

            std::size_t step = first_step;
            while (composite_idx < candidate_count_) {
                mark(composite_idx);
                composite_idx += step;
                step = (step == first_step) ? second_step : first_step;
            }
        }
    }

    std::uint64_t count_primes() const {
        if (limit_ < 2U) {
            return 0;
        }
        std::uint64_t count = 1ULL;
        if (limit_ >= 3U) {
            ++count;
        }
        for (std::uint64_t word : bits_) {
#if defined(__GNUG__) || defined(__clang__)
            count += 64ULL - static_cast<std::uint64_t>(__builtin_popcountll(word));
#else
            std::uint64_t pop = 0ULL;
            while (word != 0ULL) {
                word &= (word - 1ULL);
                ++pop;
            }
            count += 64ULL - pop;
#endif
        }
        count -= static_cast<std::uint64_t>(bits_.size() * 64ULL - candidate_count_);
        return count;
    }

    std::size_t memory_bytes() const {
        return bits_.size() * sizeof(std::uint64_t);
    }

private:
    static std::size_t compute_candidate_count(std::uint32_t limit) {
        if (limit < 5U) {
            return 0;
        }
        const std::uint32_t blocks = (limit - 5U) / 6U;
        std::size_t count = static_cast<std::size_t>(blocks) * 2ULL;
        switch (limit % 6U) {
        case 0U:
            count += 1ULL;
            break;
        case 1U:
        case 2U:
        case 3U:
        case 4U:
            count += 2ULL;
            break;
        case 5U:
            count += 1ULL;
            break;
        default:
            break;
        }
        return count;
    }

    static std::size_t words_for_bits(std::size_t bit_count) {
        return (bit_count + 63ULL) >> 6U;
    }

    static std::uint64_t candidate_to_number(std::size_t idx) {
        return 3ULL * idx + 5ULL - (idx & 1ULL);
    }

    static std::size_t square_to_index(std::uint64_t p2) {
        return static_cast<std::size_t>((p2 - 4ULL) / 3ULL);
    }

    bool is_marked(std::size_t idx) const {
        return (bits_[idx >> 6U] >> (idx & 63ULL)) & 1ULL;
    }

    void mark(std::size_t idx) {
        bits_[idx >> 6U] |= (1ULL << (idx & 63ULL));
    }

    std::uint32_t limit_;
    std::size_t candidate_count_;
    std::vector<std::uint64_t> bits_;
};

std::vector<std::uint32_t> simple_base_primes(std::uint32_t limit) {
    std::size_t ignored_memory = 0;
    const std::size_t odd_count = (limit >= 3U) ? static_cast<std::size_t>((limit - 3U) / 2U) + 1ULL : 0ULL;
    std::vector<std::uint8_t> composite(odd_count, 0U);
    ignored_memory = composite.size();
    (void)ignored_memory;

    const std::uint32_t bound = isqrt(limit);
    for (std::size_t idx = 0; idx < odd_count; ++idx) {
        if (composite[idx] != 0U) {
            continue;
        }
        const std::uint32_t p = static_cast<std::uint32_t>(idx * 2ULL + 3ULL);
        if (p > bound) {
            break;
        }
        std::size_t composite_idx = static_cast<std::size_t>((static_cast<std::uint64_t>(p) * p - 3ULL) / 2ULL);
        while (composite_idx < odd_count) {
            composite[composite_idx] = 1U;
            composite_idx += p;
        }
    }

    std::vector<std::uint32_t> primes;
    if (limit >= 2U) {
        primes.push_back(2U);
    }
    for (std::size_t idx = 0; idx < odd_count; ++idx) {
        if (composite[idx] == 0U) {
            primes.push_back(static_cast<std::uint32_t>(idx * 2ULL + 3ULL));
        }
    }
    return primes;
}

std::uint64_t run_segmented_odd_sieve(std::uint32_t limit, std::size_t& memory_bytes) {
    if (limit < 2U) {
        memory_bytes = 0;
        return 0;
    }

    const std::uint32_t root = isqrt(limit);
    const std::vector<std::uint32_t> base_primes = simple_base_primes(root);
    const std::uint32_t segment_span = 1U << 20U;
    const std::size_t segment_odd_count = static_cast<std::size_t>(segment_span / 2U);
    std::vector<std::uint8_t> composite(segment_odd_count, 0U);
    memory_bytes = composite.size() + base_primes.size() * sizeof(std::uint32_t);

    std::uint64_t count = 1ULL;
    std::uint32_t low = 3U;
    while (low <= limit) {
        std::uint32_t high = low + segment_span - 1U;
        if (high > limit) {
            high = limit;
        }
        if ((low & 1U) == 0U) {
            ++low;
        }
        if ((high & 1U) == 0U) {
            --high;
        }
        if (low > high) {
            break;
        }

        const std::size_t active_count = static_cast<std::size_t>((high - low) / 2U) + 1ULL;
        std::fill(composite.begin(), composite.begin() + static_cast<std::ptrdiff_t>(active_count), 0U);

        for (std::uint32_t p : base_primes) {
            if (p == 2U) {
                continue;
            }
            const std::uint64_t p64 = p;
            std::uint64_t start = p64 * p64;
            if (start < low) {
                start = ((static_cast<std::uint64_t>(low) + p64 - 1ULL) / p64) * p64;
            }
            if ((start & 1ULL) == 0ULL) {
                start += p64;
            }
            for (std::uint64_t value = start; value <= high; value += p64 * 2ULL) {
                composite[static_cast<std::size_t>((value - low) / 2ULL)] = 1U;
            }
        }

        for (std::size_t idx = 0; idx < active_count; ++idx) {
            count += (composite[idx] == 0U) ? 1ULL : 0ULL;
        }

        if (high >= limit - 1U) {
            break;
        }
        low = high + 2U;
    }

    return count;
}

struct AdaptiveWheelConfig {
    std::uint32_t segment_span = 1U << 20U;
    std::size_t wheel_pattern_budget_bytes = 64U * 1024U;
};

std::vector<std::uint32_t> build_adaptive_wheel_primes(
    const std::vector<std::uint32_t>& base_primes,
    std::size_t budget_bytes,
    std::uint64_t& wheel_period) {
    std::vector<std::uint32_t> wheel_primes;
    wheel_period = 1ULL;

    for (std::uint32_t p : base_primes) {
        if (p == 2U) {
            continue;
        }
        const std::uint64_t next_period = wheel_period * static_cast<std::uint64_t>(p);
        if (next_period > budget_bytes) {
            break;
        }
        wheel_primes.push_back(p);
        wheel_period = next_period;
    }

    if (wheel_primes.empty()) {
        wheel_primes.push_back(3U);
        wheel_period = 3ULL;
    }
    return wheel_primes;
}

std::uint64_t run_segmented_adaptive_wheel(std::uint32_t limit, std::size_t& memory_bytes) {
    if (limit < 2U) {
        memory_bytes = 0;
        return 0;
    }

    const AdaptiveWheelConfig cfg{};
    const std::uint32_t root = isqrt(limit);
    const std::vector<std::uint32_t> base_primes = simple_base_primes(root);

    std::uint64_t wheel_period = 1ULL;
    const std::vector<std::uint32_t> wheel_primes =
        build_adaptive_wheel_primes(base_primes, cfg.wheel_pattern_budget_bytes, wheel_period);
    const std::uint32_t wheel_max_prime = wheel_primes.back();

    const std::size_t wheel_slots = static_cast<std::size_t>(wheel_period);
    std::vector<std::uint8_t> wheel_pattern(wheel_slots, 0U);
    for (std::size_t residue = 0; residue < wheel_slots; ++residue) {
        const std::uint64_t value = static_cast<std::uint64_t>(residue);
        bool divisible = false;
        for (std::uint32_t p : wheel_primes) {
            if (value % p == 0ULL) {
                divisible = true;
                break;
            }
        }
        wheel_pattern[residue] = divisible ? 1U : 0U;
    }

    const std::size_t segment_odd_count = static_cast<std::size_t>(cfg.segment_span / 2U);
    std::vector<std::uint8_t> composite(segment_odd_count, 0U);
    memory_bytes = composite.size() + wheel_pattern.size() + base_primes.size() * sizeof(std::uint32_t);

    std::vector<std::uint32_t> mark_primes;
    mark_primes.reserve(base_primes.size());
    for (std::uint32_t p : base_primes) {
        if (p > wheel_max_prime) {
            mark_primes.push_back(p);
        }
    }

    std::uint64_t count = 1ULL;
    std::uint32_t low = 3U;
    while (low <= limit) {
        std::uint32_t high = low + cfg.segment_span - 1U;
        if (high > limit) {
            high = limit;
        }
        if ((low & 1U) == 0U) {
            ++low;
        }
        if ((high & 1U) == 0U) {
            --high;
        }
        if (low > high) {
            break;
        }

        const std::size_t active_count = static_cast<std::size_t>((high - low) / 2U) + 1ULL;

        std::uint64_t residue = static_cast<std::uint64_t>(low) % wheel_period;
        for (std::size_t idx = 0; idx < active_count; ++idx) {
            composite[idx] = wheel_pattern[static_cast<std::size_t>(residue)];
            residue += 2ULL;
            if (residue >= wheel_period) {
                residue -= wheel_period;
            }
        }

        for (std::uint32_t p : mark_primes) {
            const std::uint64_t p64 = p;
            std::uint64_t start = p64 * p64;
            if (start < low) {
                start = ((static_cast<std::uint64_t>(low) + p64 - 1ULL) / p64) * p64;
            }
            if ((start & 1ULL) == 0ULL) {
                start += p64;
            }
            for (std::uint64_t value = start; value <= high; value += p64 * 2ULL) {
                composite[static_cast<std::size_t>((value - low) / 2ULL)] = 1U;
            }
        }

        // 轮基素数本身必须保持为素数（预筛模板中会把它们所在同余类写成合数）。
        for (std::uint32_t p : wheel_primes) {
            if (p >= low && p <= high && (p & 1U) == 1U) {
                composite[static_cast<std::size_t>((p - low) / 2U)] = 0U;
            }
        }

        for (std::size_t idx = 0; idx < active_count; ++idx) {
            count += (composite[idx] == 0U) ? 1ULL : 0ULL;
        }

        if (high >= limit - 1U) {
            break;
        }
        low = high + 2U;
    }

    return count;
}

template <typename Runner>
Result benchmark(const std::string& name, std::uint32_t limit, Runner runner) {
    std::size_t memory_bytes = 0;
    const auto start = std::chrono::steady_clock::now();
    const std::uint64_t count = runner(limit, memory_bytes);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return Result{name, limit, count, static_cast<std::uint64_t>(elapsed), static_cast<double>(memory_bytes) / (1024.0 * 1024.0)};
}

Result benchmark_wheel6(std::uint32_t limit) {
    WheelSieve6 sieve(limit);
    const auto start = std::chrono::steady_clock::now();
    sieve.run();
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return Result{"wheel6_bit", limit, sieve.count_primes(), static_cast<std::uint64_t>(elapsed), static_cast<double>(sieve.memory_bytes()) / (1024.0 * 1024.0)};
}

void print_table(const std::vector<Result>& results, std::uint64_t expected) {
    std::cout << "limit=" << results.front().limit << ", expected=" << expected << "\n";
    std::cout << std::left << std::setw(18) << "algorithm"
              << std::right << std::setw(14) << "count"
              << std::setw(12) << "time_ms"
              << std::setw(14) << "memory_mb"
              << std::setw(10) << "ok"
              << "\n";
    for (const Result& result : results) {
        std::cout << std::left << std::setw(18) << result.name
                  << std::right << std::setw(14) << result.count
                  << std::setw(12) << result.elapsed_ms
                  << std::setw(14) << std::fixed << std::setprecision(3) << result.memory_mb
                  << std::setw(10) << (result.count == expected ? "yes" : "no")
                  << std::defaultfloat << "\n";
    }
    std::cout << "\n";
}

} // namespace bench

int main() {
    using bench::benchmark;
    using bench::benchmark_wheel6;
    using bench::print_table;
    using bench::run_full_byte_sieve;
    using bench::run_odd_byte_sieve;
    using bench::run_segmented_adaptive_wheel;
    using bench::run_segmented_odd_sieve;

    const std::vector<std::pair<std::uint32_t, std::uint64_t>> cases = {
        {10'000'000U, 664'579ULL},
        {100'000'000U, 5'761'455ULL},
        {1'000'000'000U, 50'847'534ULL},
    };

    for (const auto& [limit, expected] : cases) {
        std::vector<bench::Result> results;
        const bool large_case = (limit >= 1'000'000'000U);
        if (!large_case) {
            results.push_back(benchmark("full_byte", limit, run_full_byte_sieve));
            results.push_back(benchmark("odd_byte", limit, run_odd_byte_sieve));
        }
        results.push_back(benchmark_wheel6(limit));
        results.push_back(benchmark("segmented_odd", limit, run_segmented_odd_sieve));
        results.push_back(benchmark("adaptive_wheel", limit, run_segmented_adaptive_wheel));
        print_table(results, expected);
    }

    return 0;
}