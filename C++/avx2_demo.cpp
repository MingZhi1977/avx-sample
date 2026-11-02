/**
 * @file avx2_demo.cpp
 * @brief CPU AVX2 (Advanced Vector Extensions 2) C++ 物件導向改寫示範
 * 
 * 本程式展示 C++ 改寫 AVX2 的核心優勢：
 * 1. 零成本抽象 (Zero-Cost Abstraction)：
 *    利用 C++ 類別 (Class) 與運算子多載 (Operator Overloading) 封裝 256 位元 YMM 暫存器，
 *    讓向量運算程式碼如同普通數學式 (例如 a * a + b * 3 + a * b) 簡潔直覺。
 * 2. 浮點數 FMA (Fused Multiply-Add) 封裝。
 * 3. 效能對比基準測試 (Benchmark)：
 *    - 傳統純量迴圈 (Scalar)
 *    - C++ AVX2 單執行緒 (Single-Threaded AVX2)
 *    - C++ AVX2 多執行緒並行 (Multi-Threaded AVX2 with std::jthread)
 *
 * 編譯方式：
 *   g++ -std=c++20 -O2 -mavx2 -mfma -pthread avx2_demo.cpp -o avx2_demo
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <immintrin.h> // Intel/AMD SIMD 內建函式 (Intrinsics)

// ============================================================================
// 1. 零成本抽象：256 位元整數向量類別 (8 x int32)
// ============================================================================
class alignas(32) Vec8i {
public:
    __m256i reg;

    // 建構子
    inline Vec8i() : reg(_mm256_setzero_si256()) {}
    inline Vec8i(__m256i r) : reg(r) {}
    inline Vec8i(int scalar) : reg(_mm256_set1_epi32(scalar)) {} // 廣播單一數值

    // 載入與儲存
    static inline Vec8i load(const int* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    inline void store(int* ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), reg);
    }

    // 運算子多載 (編譯器在 -O2 下會自動 Inline，生成的機器碼與直接呼叫 Intrinsics 完全相同)
    inline friend Vec8i operator+(const Vec8i& a, const Vec8i& b) {
        return _mm256_add_epi32(a.reg, b.reg);
    }
    inline friend Vec8i operator-(const Vec8i& a, const Vec8i& b) {
        return _mm256_sub_epi32(a.reg, b.reg);
    }
    inline friend Vec8i operator*(const Vec8i& a, const Vec8i& b) {
        return _mm256_mullo_epi32(a.reg, b.reg);
    }

    // 串流輸出方便除錯與印出
    friend std::ostream& operator<<(std::ostream& os, const Vec8i& v) {
        alignas(32) int buf[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), v.reg);
        os << "[ ";
        for (int i = 0; i < 8; ++i) {
            os << std::setw(3) << buf[i] << " ";
        }
        os << "]";
        return os;
    }
};

// ============================================================================
// 2. 零成本抽象：256 位元單精度浮點向量類別 (8 x float32)
// ============================================================================
class alignas(32) Vec8f {
public:
    __m256 reg;

    inline Vec8f() : reg(_mm256_setzero_ps()) {}
    inline Vec8f(__m256 r) : reg(r) {}
    inline Vec8f(float scalar) : reg(_mm256_set1_ps(scalar)) {}

    static inline Vec8f load(const float* ptr) {
        return _mm256_loadu_ps(ptr);
    }

    inline void store(float* ptr) const {
        _mm256_storeu_ps(ptr, reg);
    }

    inline friend Vec8f operator+(const Vec8f& a, const Vec8f& b) {
        return _mm256_add_ps(a.reg, b.reg);
    }
    inline friend Vec8f operator*(const Vec8f& a, const Vec8f& b) {
        return _mm256_mul_ps(a.reg, b.reg);
    }

    // 融合乘加 (Fused Multiply-Add)：計算 a * b + c
    static inline Vec8f fma(const Vec8f& a, const Vec8f& b, const Vec8f& c) {
        return _mm256_fmadd_ps(a.reg, b.reg, c.reg);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec8f& v) {
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, v.reg);
        os << "[ ";
        for (int i = 0; i < 8; ++i) {
            os << std::fixed << std::setprecision(1) << buf[i] << " ";
        }
        os << "]";
        return os;
    }
};

// ============================================================================
// 3. CPU 支援檢查
// ============================================================================
static bool check_cpu_support() {
    std::cout << "[1] 檢查 CPU 指令集支援狀態:\n";
    bool has_avx2 = __builtin_cpu_supports("avx2");
    bool has_fma  = __builtin_cpu_supports("fma");

    std::cout << "    - AVX2: " << (has_avx2 ? "支援 (YES)" : "不支援 (NO)") << "\n";
    std::cout << "    - FMA:  " << (has_fma  ? "支援 (YES)" : "不支援 (NO)") << "\n";

    if (!has_avx2) {
        std::cerr << "錯誤: 當前 CPU 不支援 AVX2 指令集！\n";
        return false;
    }
    std::cout << "    -> CPU 支援 AVX2，可正常運行！\n\n";
    return true;
}

// ============================================================================
// 4. 示範 C++ 物件封裝語法
// ============================================================================
static void demo_cpp_syntax() {
    std::cout << "========================================================\n";
    std::cout << "[2] C++ 物件導向 SIMD 語法展示 (運算子多載)\n";
    std::cout << "========================================================\n";

    int a[8] = { 1,  2,  3,  4,   5,   6,   7,   8};
    int b[8] = {10, 20, 30, 40,  50,  60,  70,  80};

    Vec8i va = Vec8i::load(a);
    Vec8i vb = Vec8i::load(b);

    // 在 C++ 中直接使用 + 與 *，不再需要繁瑣的 _mm256_add_epi32(...)
    Vec8i v_add = va + vb;
    Vec8i v_mul = va * vb;
    Vec8i v_broadcast = va + 100; // 隱式轉換並廣播

    std::cout << "向量 A:          " << va << "\n";
    std::cout << "向量 B:          " << vb << "\n";
    std::cout << "向量加法 (A+B):   " << v_add << "  <-- 使用 va + vb\n";
    std::cout << "向量乘法 (A*B):   " << v_mul << "  <-- 使用 va * vb\n";
    std::cout << "廣播常數 (A+100): " << v_broadcast << "  <-- 使用 va + 100\n\n";

    std::cout << "[3] C++ 浮點數 FMA 展示\n";
    float fa[8] = {1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f};
    float fb[8] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float fc[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    Vec8f vfa = Vec8f::load(fa);
    Vec8f vfb = Vec8f::load(fb);
    Vec8f vfc = Vec8f::load(fc);
    Vec8f vfma = Vec8f::fma(vfa, vfb, vfc);

    std::cout << "FMA (A*B+C):     " << vfma << "\n\n";
}

// ============================================================================
// 5. 基準測試函式
// ============================================================================

// (A) 純量版本 (禁止自動向量化)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline, optimize("no-tree-vectorize")))
#else
__attribute__((noinline))
#endif
static void scalar_compute(const int* a, const int* b, int* c, size_t n, int iters) {
    for (int iter = 0; iter < iters; ++iter) {
#if defined(__clang__)
        #pragma clang loop vectorize(disable)
#endif
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] * a[i] + b[i] * 3 + a[i] * b[i];
        }
    }
}

// (B) C++ AVX2 單執行緒 (使用 Vec8i 類別)
__attribute__((noinline))
static void avx2_compute_single(const int* a, const int* b, int* c, size_t n, int iters) {
    for (int iter = 0; iter < iters; ++iter) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            Vec8i va = Vec8i::load(a + i);
            Vec8i vb = Vec8i::load(b + i);

            // 像普通算式一樣書寫，編譯後零成本！
            Vec8i res = va * va + vb * 3 + va * vb;
            res.store(c + i);
        }
        for (; i < n; ++i) {
            c[i] = a[i] * a[i] + b[i] * 3 + a[i] * b[i];
        }
    }
}

// (C) C++ AVX2 多執行緒版本 (利用 std::jthread 將資料分段並行)
__attribute__((noinline))
static void avx2_compute_multi(const int* a, const int* b, int* c, size_t n, int iters, unsigned int num_threads) {
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    size_t chunk_size = (n + num_threads - 1) / num_threads;
    // 確保每個 thread 的起點為 8 的整數倍以保證向量對齊
    chunk_size = (chunk_size + 7) & ~7ULL;

    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end   = std::min(start + chunk_size, n);
        if (start >= n) break;

        threads.emplace_back([=]() {
            for (int iter = 0; iter < iters; ++iter) {
                size_t i = start;
                for (; i + 8 <= end; i += 8) {
                    Vec8i va = Vec8i::load(a + i);
                    Vec8i vb = Vec8i::load(b + i);
                    Vec8i res = va * va + vb * 3 + va * vb;
                    res.store(c + i);
                }
                for (; i < end; ++i) {
                    c[i] = a[i] * a[i] + b[i] * 3 + a[i] * b[i];
                }
            }
        });
    }
    // std::jthread 會在解構時自動呼叫 join()
}

static void demo_benchmark() {
    const size_t N = 1024 * 1024; // 100 萬筆資料
    const int ITERS = 200;
    const unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());

    std::cout << "========================================================\n";
    std::cout << "[4] 效能對比基準測試 (Benchmark)\n";
    std::cout << "    資料量: " << N << " 個整數, 重複 " << ITERS << " 次\n";
    std::cout << "    偵測硬體核心數: " << num_threads << " 核心\n";
    std::cout << "========================================================\n";

    std::vector<int> a(N), b(N);
    std::vector<int> c_scalar(N), c_avx2_single(N), c_avx2_multi(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<int>(i % 100);
        b[i] = static_cast<int>((i * 2) % 100);
    }

    // 1. 純量
    std::cout << "1. 執行「傳統純量」運算...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    scalar_compute(a.data(), b.data(), c_scalar.data(), N, ITERS);
    auto t1 = std::chrono::high_resolution_clock::now();
    double scalar_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "   -> 純量耗時:         " << std::fixed << std::setprecision(4) << scalar_time << " 秒\n";

    // 2. C++ AVX2 單執行緒
    std::cout << "2. 執行「C++ AVX2 類別包裝」單執行緒...\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    avx2_compute_single(a.data(), b.data(), c_avx2_single.data(), N, ITERS);
    auto t3 = std::chrono::high_resolution_clock::now();
    double avx2_single_time = std::chrono::duration<double>(t3 - t2).count();
    std::cout << "   -> AVX2 單執行緒耗時: " << avx2_single_time << " 秒\n";
    std::cout << "   -> 單線程加速比:     " << std::setprecision(2) << (scalar_time / avx2_single_time) << " 倍\n";

    // 3. C++ AVX2 多執行緒 (SIMD + 多核心)
    std::cout << "3. 執行「C++ AVX2 + " << num_threads << " 核心多執行緒」並行...\n";
    auto t4 = std::chrono::high_resolution_clock::now();
    avx2_compute_multi(a.data(), b.data(), c_avx2_multi.data(), N, ITERS, num_threads);
    auto t5 = std::chrono::high_resolution_clock::now();
    double avx2_multi_time = std::chrono::duration<double>(t5 - t4).count();
    std::cout << "   -> AVX2 多執行緒耗時: " << std::setprecision(4) << avx2_multi_time << " 秒\n";
    std::cout << "   -> 整體總體加速比:   " << std::setprecision(2) << (scalar_time / avx2_multi_time) << " 倍\n";

    // 4. 正確性比對
    bool pass = true;
    for (size_t i = 0; i < N; ++i) {
        if (c_scalar[i] != c_avx2_single[i] || c_scalar[i] != c_avx2_multi[i]) {
            pass = false;
            std::cerr << "比對失敗於索引 " << i << "！\n";
            break;
        }
    }
    std::cout << "-> 正確性驗證: " << (pass ? "通過 (所有結果一致)" : "失敗") << "\n";
    std::cout << "========================================================\n\n";
}

int main() {
    std::cout << "########################################################\n";
    std::cout << "#         CPU AVX2 SIMD 指令集展示程式 (C++ 版)        #\n";
    std::cout << "########################################################\n\n";

    if (!check_cpu_support()) {
        return 1;
    }

    demo_cpp_syntax();
    demo_benchmark();

    std::cout << "示範完成！\n";
    return 0;
}
