/**
 * @file avx2_demo.c
 * @brief CPU AVX2 (Advanced Vector Extensions 2) 指令集功能示範
 * 
 * 本程式示範：
 * 1. 執行時期 CPU AVX2 / FMA 支援度動態偵測
 * 2. 基礎 256-bit 整數 SIMD 運算（一次處理 8 個 32-bit 整數）
 * 3. 浮點數 FMA (Fused Multiply-Add) 融合乘加指令（一次處理 8 個 float）
 * 4. 傳統純量 (Scalar) 運算 vs AVX2 向量化運算的效能對比 (Benchmark)
 *
 * 編譯方式：
 *   gcc -O2 -mavx2 -mfma avx2_demo.c -o avx2_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <immintrin.h> // Intel/AMD SIMD 內建函式 (Intrinsics)

// 取得高精度時間（秒）
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================================
// 1. CPU 功能偵測
// ============================================================================
static bool check_cpu_support(void) {
    printf("[1] 檢查 CPU 指令集支援狀態:\n");
    
    // GCC / Clang 提供的內建函式，底層使用 CPUID 指令
    bool has_avx  = __builtin_cpu_supports("avx");
    bool has_avx2 = __builtin_cpu_supports("avx2");
    bool has_fma  = __builtin_cpu_supports("fma");

    printf("    - AVX  (256-bit 浮點支援) : %s\n", has_avx  ? "支援 (YES)" : "不支援 (NO)");
    printf("    - AVX2 (256-bit 整數擴展) : %s\n", has_avx2 ? "支援 (YES)" : "不支援 (NO)");
    printf("    - FMA  (融合乘加指令集)   : %s\n", has_fma  ? "支援 (YES)" : "不支援 (NO)");

    if (!has_avx2) {
        fprintf(stderr, "\n錯誤: 當前 CPU 不支援 AVX2 指令集，無法執行向量運算示範！\n");
        return false;
    }
    printf("    -> CPU 支援 AVX2，可正常運行所有向量指令！\n\n");
    return true;
}

// ============================================================================
// 2. 示範：256-bit 整數向量運算 (AVX2 的標誌性功能)
// ============================================================================
static void demo_integer_operations(void) {
    printf("========================================================\n");
    printf("[2] AVX2 基礎整數運算示範 (一次處理 8 個 32-bit 整數)\n");
    printf("========================================================\n");

    // 256 bits = 32 bytes = 8 * 4 bytes (8 個 int)
    int a[8] = { 1,  2,  3,  4,   5,   6,   7,   8};
    int b[8] = {10, 20, 30, 40,  50,  60,  70,  80};
    int res_add[8];
    int res_mul[8];
    int res_broadcast[8];

    // (1) 載入資料至 256 位元 YMM 暫存器 (__m256i)
    // _mm256_loadu_si256 中的 'u' 代表 unaligned (不要求記憶體必須 32-byte 對齊)
    __m256i va = _mm256_loadu_si256((const __m256i*)a);
    __m256i vb = _mm256_loadu_si256((const __m256i*)b);

    // (2) 向量加法：_mm256_add_epi32 同時將 8 組 32-bit 整數相加
    __m256i v_add = _mm256_add_epi32(va, vb);

    // (3) 向量乘法：_mm256_mullo_epi32 同時將 8 組 32-bit 整數相乘，保留低 32 位元
    __m256i v_mul = _mm256_mullo_epi32(va, vb);

    // (4) 常數廣播：_mm256_set1_epi32 將單一數值複製填滿整個 256-bit 暫存器的 8 個通道
    __m256i v_const = _mm256_set1_epi32(100);
    __m256i v_broadcast_res = _mm256_add_epi32(va, v_const);

    // (5) 將運算結果存回一般的記憶體陣列
    _mm256_storeu_si256((__m256i*)res_add, v_add);
    _mm256_storeu_si256((__m256i*)res_mul, v_mul);
    _mm256_storeu_si256((__m256i*)res_broadcast, v_broadcast_res);

    printf("陣列 A:          [ ");
    for (int i = 0; i < 8; i++) printf("%3d ", a[i]);
    printf("]\n");

    printf("陣列 B:          [ ");
    for (int i = 0; i < 8; i++) printf("%3d ", b[i]);
    printf("]\n");

    printf("向量加法 (A+B):   [ ");
    for (int i = 0; i < 8; i++) printf("%3d ", res_add[i]);
    printf("] -> 8 個加法同步在 1 個指令中完成\n");

    printf("向量乘法 (A*B):   [ ");
    for (int i = 0; i < 8; i++) printf("%3d ", res_mul[i]);
    printf("] -> 8 個乘法同步在 1 個指令中完成\n");

    printf("廣播加法 (A+100): [ ");
    for (int i = 0; i < 8; i++) printf("%3d ", res_broadcast[i]);
    printf("] -> 廣播常數 100 至 8 個元素\n\n");
}

// ============================================================================
// 3. 示範：浮點數 FMA (Fused Multiply-Add)
// ============================================================================
static void demo_float_fma(void) {
    printf("========================================================\n");
    printf("[3] 浮點數 FMA 融合乘加示範 (計算 A * B + C)\n");
    printf("========================================================\n");

    // 8 個單精度浮點數 (32-bit float)
    float a[8] = {1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f};
    float b[8] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float c[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float res[8];

    // __m256 用於儲存 8 個 32-bit 單精度浮點數
    __m256 va = _mm256_loadu_ps(a);
    __m256 vb = _mm256_loadu_ps(b);
    __m256 vc = _mm256_loadu_ps(c);

    // _mm256_fmadd_ps(a, b, c) = a * b + c
    // 單一硬體週期完成，且只在最後進行一次捨入，比分開執行 mul + add 精度更高、速度更快
    __m256 vfma = _mm256_fmadd_ps(va, vb, vc);

    _mm256_storeu_ps(res, vfma);

    printf("A:            [ ");
    for (int i = 0; i < 8; i++) printf("%.1f ", a[i]);
    printf("]\n");

    printf("B:            [ ");
    for (int i = 0; i < 8; i++) printf("%.1f ", b[i]);
    printf("]\n");

    printf("C:            [ ");
    for (int i = 0; i < 8; i++) printf("%.1f ", c[i]);
    printf("]\n");

    printf("FMA (A*B+C):  [ ");
    for (int i = 0; i < 8; i++) printf("%.1f ", res[i]);
    printf("]\n\n");
}

// ============================================================================
// 4. 效能對比 (Benchmark)：純量 (Scalar) vs AVX2 向量加速
// ============================================================================

// 避免編譯器自動向量化傳統迴圈，以精確測量「純量運算」vs「手動 AVX2」的真實差距
__attribute__((noinline, optimize("no-tree-vectorize")))
static void scalar_compute(const int *a, const int *b, int *c, size_t n, int iters) {
    for (int iter = 0; iter < iters; iter++) {
        for (size_t i = 0; i < n; i++) {
            // 多項式混合運算：包含多次乘法與加法
            c[i] = a[i] * a[i] + b[i] * 3 + a[i] * b[i];
        }
    }
}

// 使用 AVX2 指令手動向量化
__attribute__((noinline))
static void avx2_compute(const int *a, const int *b, int *c, size_t n, int iters) {
    __m256i v3 = _mm256_set1_epi32(3);
    for (int iter = 0; iter < iters; iter++) {
        size_t i = 0;
        // 主迴圈：每次迭代處理 8 個整數
        for (; i + 8 <= n; i += 8) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            
            __m256i a2 = _mm256_mullo_epi32(va, va);       // a * a
            __m256i b3 = _mm256_mullo_epi32(vb, v3);       // b * 3
            __m256i ab = _mm256_mullo_epi32(va, vb);       // a * b

            __m256i sum = _mm256_add_epi32(_mm256_add_epi32(a2, b3), ab);
            _mm256_storeu_si256((__m256i*)(c + i), sum);
        }
        // 結尾處理：無法湊滿 8 個的剩餘元素以純量計算
        for (; i < n; i++) {
            c[i] = a[i] * a[i] + b[i] * 3 + a[i] * b[i];
        }
    }
}

static void demo_benchmark(void) {
    const size_t N = 1024 * 1024; // 1,048,576 個整數 (4 MB)
    const int ITERS = 200;        // 重複 200 次以穩定測量時間

    printf("========================================================\n");
    printf("[4] 效能對比基準測試 (Benchmark)\n");
    printf("    資料規模: %zu 個整數，重複執行 %d 次迴圈\n", N, ITERS);
    printf("========================================================\n");

    int *a = (int*)malloc(N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));
    int *c_scalar = (int*)malloc(N * sizeof(int));
    int *c_avx2   = (int*)malloc(N * sizeof(int));

    if (!a || !b || !c_scalar || !c_avx2) {
        fprintf(stderr, "記憶體配置失敗！\n");
        return;
    }

    // 初始化測試資料
    for (size_t i = 0; i < N; i++) {
        a[i] = (int)(i % 100);
        b[i] = (int)((i * 2) % 100);
    }

    // (1) 測量純量版本
    printf("正在執行「傳統純量」運算...\n");
    double t0 = get_time_sec();
    scalar_compute(a, b, c_scalar, N, ITERS);
    double t1 = get_time_sec();
    double scalar_time = t1 - t0;
    printf("-> 純量耗時: %.4f 秒\n", scalar_time);

    // (2) 測量 AVX2 向量版本
    printf("正在執行「AVX2 向量」運算...\n");
    double t2 = get_time_sec();
    avx2_compute(a, b, c_avx2, N, ITERS);
    double t3 = get_time_sec();
    double avx2_time = t3 - t2;
    printf("-> AVX2 耗時: %.4f 秒\n", avx2_time);

    // (3) 結果比對
    bool pass = true;
    for (size_t i = 0; i < N; i++) {
        if (c_scalar[i] != c_avx2[i]) {
            pass = false;
            printf("錯誤：在索引 %zu 處結果不符 (純量: %d, AVX2: %d)\n", i, c_scalar[i], c_avx2[i]);
            break;
        }
    }

    printf("-> 正確性驗證: %s\n", pass ? "通過 (兩者計算結果完全一致)" : "失敗");
    if (avx2_time > 0) {
        printf("-> 效能加速比: %.2f 倍 (加速提升約 %.1f%%)\n", 
               scalar_time / avx2_time, 
               ((scalar_time - avx2_time) / scalar_time) * 100.0);
    }

    free(a);
    free(b);
    free(c_scalar);
    free(c_avx2);
    printf("========================================================\n\n");
}

int main(void) {
    printf("########################################################\n");
    printf("#           CPU AVX2 SIMD 指令集展示程式               #\n");
    printf("########################################################\n\n");

    if (!check_cpu_support()) {
        return 1;
    }

    demo_integer_operations();
    demo_float_fma();
    demo_benchmark();

    printf("示範完成！\n");
    return 0;
}
