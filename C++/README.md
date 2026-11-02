# CPU AVX2 指令集範例程式 (C++ 物件導向改寫版)

本目錄提供使用現代 **C++ (C++20)** 改寫的 AVX2 / FMA 指令集示範程式。

---

## C++ 改寫帶來的核心亮點

1. **零成本抽象 (Zero-Cost Abstraction)**：
   - 封裝 `Vec8i`（8 個 32-bit 整數）與 `Vec8f`（8 個 32-bit 浮點數）類別。
   - 透過**運算子多載（Operator Overloading）**，原本晦澀難懂的 Intrinsics 呼叫：
     ```cpp
     // C 語言寫法
     __m256i res = _mm256_add_epi32(_mm256_add_epi32(a2, b3), ab);
     ```
     在 C++ 中可以像普通算式一樣直覺書寫：
     ```cpp
     // C++ 物件導向寫法
     Vec8i res = va * va + vb * 3 + va * vb;
     ```
   - 在 `-O2` 編譯時會完全 Inline 展開，**零額外記憶體與運算開銷**，產生與手寫 C 語言完全一致的機器碼。

2. **SIMD + 多核心並行 (Multi-Threading)**：
   - 使用 C++20 的 `std::jthread` 將數據分塊，並行調用多核心進行 AVX2 計算。
   - 示範如何在單核 SIMD 加速（~5.7x）的基礎上，結合 12 核心獲得 **約 17 倍的總體加速**。

---

## 目錄結構

- [`avx2_demo.cpp`](file:///home/mingzhi/AI-WS/avx-sample/C++/avx2_demo.cpp)：C++ 示範程式原始碼
- [`Makefile`](file:///home/mingzhi/AI-WS/avx-sample/C++/Makefile)：編譯與執行腳本

---

## 編譯與執行

```bash
# 編譯並直接執行
make run

# 僅編譯
make

# 清理產生的執行檔
make clean
```

手動編譯指令：
```bash
g++ -std=c++20 -O2 -mavx2 -mfma -pthread avx2_demo.cpp -o avx2_demo
./avx2_demo
```
