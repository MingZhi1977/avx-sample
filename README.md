# CPU AVX2 指令集範例程式

本專案提供一個簡明且完整的 C 語言程式，示範如何使用 CPU 中的 **AVX2 (Advanced Vector Extensions 2)** 指令集進行 SIMD (Single Instruction, Multiple Data) 平行計算。

---

## 什麼是 AVX2？

- **SIMD (單指令多資料流)**：傳統 CPU 指令一次只能處理一筆資料（例如 $a + b$）；SIMD 指令則可在單一指令週期內，對多組資料同時執行相同運算。
- **AVX vs. AVX2**：
  - **AVX (2011)**：將暫存器擴展至 256 位元（YMM0 ~ YMM15），但主要針對單精度/雙精度浮點數進行向量化。
  - **AVX2 (2013)**：將 **整數運算** 完整擴展至 256 位元（`__m256i`），支援一次同時處理：
    - 32 個 8 位元整數 (`int8_t` / `char`)
    - 16 個 16 位元整數 (`int16_t` / `short`)
    - 8 個 32 位元整數 (`int32_t` / `int`)
    - 4 個 64 位元整數 (`int64_t` / `long long`)
  - 同時引入了 **FMA (Fused Multiply-Add)**、向量廣播 (Broadcast)、Gather 收集指令等功能。

---

## 程式碼結構與示範內容

程式檔案：[`avx2_demo.c`](file:///home/mingzhi/AI-WS/avx-sample/avx2_demo.c)

1. **CPU 支援檢查 (`check_cpu_support`)**：
   - 透過 `__builtin_cpu_supports("avx2")` 動態檢查 CPU 是否支援 AVX2，避免在不支援的機器上拋出 `Illegal instruction`。
2. **256-bit 整數向量運算 (`demo_integer_operations`)**：
   - 示範 `_mm256_loadu_si256`、`_mm256_add_epi32`、`_mm256_mullo_epi32`、`_mm256_set1_epi32`、`_mm256_storeu_si256`。
3. **浮點數 FMA 融合乘加 (`demo_float_fma`)**：
   - 示範 `_mm256_fmadd_ps`，在一個指令週期內完成 8 組單精度浮點數的 $A \times B + C$ 運算。
4. **效能基準測試 (`demo_benchmark`)**：
   - 針對 100 萬筆數據進行多項式混合運算，比對傳統迴圈（純量 Scalar）與手動 AVX2 向量化程式碼的執行時間與加速比（通常可獲得約 **5 ~ 6 倍** 加速）。

---

## 如何編譯與執行

### 使用 Makefile

```bash
# 編譯並直接執行
make run

# 僅編譯
make

# 清理產生的執行檔
make clean
```

### 使用 GCC 手動編譯

```bash
gcc -O2 -mavx2 -mfma avx2_demo.c -o avx2_demo
./avx2_demo
```

> **編譯參數說明**：
> - `-mavx2`：啟用 AVX2 擴展指令集。
> - `-mfma`：啟用 Fused Multiply-Add (FMA) 融合乘加指令集。
> - `-O2`：標準優化等級。

---

## Intel Intrinsics 命名慣例

| 前綴 / 後綴 | 意義 | 範例 |
| :--- | :--- | :--- |
| `_mm256_` | 操作 256 位元暫存器 (YMM) | `_mm256_add_epi32` |
| `loadu` / `storeu` | 未對齊 (unaligned) 載入/儲存 | `_mm256_loadu_si256` |
| `epi32` | Extended Packed 32-bit Integer (8 個 32 位元整數) | `_mm256_add_epi32` |
| `ps` | Packed Single-precision float (8 個 32 位元浮點數) | `_mm256_fmadd_ps` |
| `pd` | Packed Double-precision float (4 個 64 位元雙精度浮點數) | `_mm256_add_pd` |
| `set1_` | 將單一數值廣播填滿整個向量暫存器 | `_mm256_set1_epi32` |
