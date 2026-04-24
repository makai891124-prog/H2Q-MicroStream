#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace primorial_mvp {

class WheelSieve6 {
public:
    explicit WheelSieve6(std::uint32_t limit)
        : limit_(limit),
          candidate_count_(compute_candidate_count(limit)),
          candidate_bits_(words_for_bits(candidate_count_), 0ULL) {}

    void run() {
        if (limit_ < 5) {
            return;
        }

        const std::size_t sieve_bound = compute_candidate_count(isqrt(limit_));
        for (std::size_t candidate_index = 0; candidate_index < sieve_bound; ++candidate_index) {
            if (is_marked(candidate_index)) {
                continue;
            }

            const std::uint64_t prime = candidate_to_number(candidate_index);
            const std::uint64_t prime_square = prime * prime;
            if (prime_square > limit_) {
                break;
            }

            std::size_t composite_index = square_to_index(prime_square);

            // 在 6k±1 的候选流形上，相邻可表示倍数的索引步长会在两种值之间轮转。
            // 这样内层循环只做“尾数跳跃”，不做模运算与除法。
            std::size_t first_step = 0;
            std::size_t second_step = 0;
            if ((candidate_index & 1ULL) == 0ULL) {
                first_step = candidate_index * 2ULL + 3ULL;
                second_step = candidate_index * 4ULL + 7ULL;
            } else {
                first_step = candidate_index * 4ULL + 5ULL;
                second_step = candidate_index * 2ULL + 3ULL;
            }

            std::size_t current_step = first_step;
            while (composite_index < candidate_count_) {
                mark(composite_index);
                composite_index += current_step;
                current_step = (current_step == first_step) ? second_step : first_step;
            }
        }
    }

    std::uint64_t count_primes() const {
        if (limit_ < 2) {
            return 0;
        }

        std::uint64_t count = 1;
        if (limit_ >= 3) {
            ++count;
        }

        for (std::uint64_t word : candidate_bits_) {
#if defined(__GNUG__) || defined(__clang__)
            count += 64ULL - static_cast<std::uint64_t>(__builtin_popcountll(word));
#else
            count += 64ULL - popcount_fallback(word);
#endif
        }

        const std::size_t valid_bits = candidate_count_;
        const std::size_t padded_bits = candidate_bits_.size() * 64ULL;
        count -= static_cast<std::uint64_t>(padded_bits - valid_bits);
        return count;
    }

    std::size_t memory_bytes() const {
        return candidate_bits_.size() * sizeof(std::uint64_t);
    }

private:
    static std::uint32_t isqrt(std::uint32_t value) {
        std::uint32_t left = 0;
        std::uint32_t right = value;
        std::uint32_t answer = 0;

        while (left <= right) {
            const std::uint32_t mid = left + (right - left) / 2U;
            const std::uint64_t square = static_cast<std::uint64_t>(mid) * static_cast<std::uint64_t>(mid);
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

    static std::uint64_t candidate_to_number(std::size_t candidate_index) {
        return 3ULL * candidate_index + 5ULL - (candidate_index & 1ULL);
    }

    static std::size_t square_to_index(std::uint64_t prime_square) {
        return static_cast<std::size_t>((prime_square - 4ULL) / 3ULL);
    }

    static std::uint64_t popcount_fallback(std::uint64_t value) {
        std::uint64_t count = 0;
        while (value != 0ULL) {
            value &= (value - 1ULL);
            ++count;
        }
        return count;
    }

    bool is_marked(std::size_t candidate_index) const {
        const std::size_t word_index = candidate_index >> 6U;
        const std::size_t bit_offset = candidate_index & 63ULL;
        return (candidate_bits_[word_index] >> bit_offset) & 1ULL;
    }

    void mark(std::size_t candidate_index) {
        const std::size_t word_index = candidate_index >> 6U;
        const std::size_t bit_offset = candidate_index & 63ULL;
        candidate_bits_[word_index] |= (1ULL << bit_offset);
    }

    std::uint32_t limit_;
    std::size_t candidate_count_;
    std::vector<std::uint64_t> candidate_bits_;
};

struct BenchmarkResult {
    std::uint32_t limit = 0;
    std::uint64_t prime_count = 0;
    std::uint64_t elapsed_ms = 0;
    double memory_mb = 0.0;
};

BenchmarkResult run_benchmark(std::uint32_t limit) {
    WheelSieve6 sieve(limit);
    const auto start = std::chrono::steady_clock::now();
    sieve.run();
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return BenchmarkResult{
        limit,
        sieve.count_primes(),
        static_cast<std::uint64_t>(elapsed),
        static_cast<double>(sieve.memory_bytes()) / (1024.0 * 1024.0)};
}

void verify_result(const BenchmarkResult& result, std::uint64_t expected) {
    if (result.prime_count != expected) {
        throw std::runtime_error(
            "素数计数错误: n=" + std::to_string(result.limit) +
            ", expected=" + std::to_string(expected) +
            ", actual=" + std::to_string(result.prime_count));
    }
}

void print_result(const BenchmarkResult& result) {
    std::cout << "上界 n = " << result.limit << '\n';
    std::cout << "  素数个数: " << result.prime_count << '\n';
    std::cout << "  核心筛法耗时: " << result.elapsed_ms << " ms\n";
    std::cout << std::fixed << std::setprecision(3)
              << "  位图内存占用: " << result.memory_mb << " MB\n";
    std::cout << std::defaultfloat;
}

} // namespace primorial_mvp

int main() {
    using primorial_mvp::BenchmarkResult;
    using primorial_mvp::print_result;
    using primorial_mvp::run_benchmark;
    using primorial_mvp::verify_result;

    try {
        std::cout << "=== 素数阶乘基底 6 轮转位图筛 MVP ===\n";
        std::cout << "说明: 位图只存储 6k+1 / 6k+5 候选点，合数通过位移原地标记。\n\n";

        const BenchmarkResult ten_million = run_benchmark(10'000'000U);
        verify_result(ten_million, 664'579ULL);
        print_result(ten_million);
        std::cout << '\n';

        const BenchmarkResult hundred_million = run_benchmark(100'000'000U);
        verify_result(hundred_million, 5'761'455ULL);
        print_result(hundred_million);

        std::cout << "\n正确性验证通过。\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "程序失败: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}