# 📊 Benchmark Verification

Validation of EASI implementations against official reported scores.

## 🟢 Status Legend & Methodology

The status is based on the absolute difference $\lvert\Delta\rvert$.

| Symbol | Status | Criteria |
| :---: | :--- | :--- |
| ✅ | **Strong Agreement** | $0.0\\% \\le \\lvert\\Delta\\rvert \\le 2.5\\%$ |
| ☑️ | **Acceptable Variance** | $2.5\\% < \\lvert\\Delta\\rvert \le 5.0\\%$ |
| ❌ | **Discrepancy** | $5.0\\% < \\lvert\\Delta\\rvert$ |

> **📝 Note on $\Delta$ Calculation:**
> * Formula: $\Delta = \text{EASI (Corresponding backend)} - \text{Target Score}$
> * **Target Source:** We prioritize the **Official Code** (local run of the official codebase) to strictly verify implementation correctness. If strict reproduction is not performed, we align with the **Paper Reported** score.
---

## 📑 Index
*(Matches the order in [Supported Benchmarks](./Support_bench_models.md))*

1. [MindCube](#1-mindcube)
2. [ViewSpatial](#2-viewspatial)
3. [EmbSpatial-Bench](#3-embspatial-bench)
4. [MMSI-Bench (no circular)](#4-mmsi-bench-no-circular)
5. [VSI-Bench](#5-vsi-bench)
6. [VSI-Bench-Debiased](#6-vsi-bench-debiased)
7. [SITE-Bench](#7-site-bench)
8. [SPAR-Bench](#8-spar-bench)
9. [STARE-Bench](#9-stare-bench)
10. [Spatial-Visualization-Benchmark](#10-spatial-visualization-benchmark)
11. [OmniSpatial](#11-omnispatial)
12. [ERQA](#12-erqa)
13. [RefSpatial-Bench](#13-refspatial-bench)
14. [RoboSpatial-Home](#14-robospatial-home)
15. [SPBench](#15-spbench)
16. [MMSI-Video-Bench](#16-mmsi-video-bench)
17. [VSI-SUPER-Recall](#17-vsi-super-recall)
18. [VSI-SUPER-Count](#18-vsi-super-count)
19. [STI-Bench](#19-sti-bench)
20. [DSR-Bench](#20-dsr-bench)
21. [ERIQ](#21-eriq)
22. [OSI-Bench](#22-osi-bench)


---

## 🔬 Detailed Verification

### 1. MindCube
* **Metric:** Accuracy

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct | `MindCubeBench_tiny_raw_qa` | 37.81 | - | 37.88 | +0.07 | ✅ |
| Qwen2.5-VL-3B-Instruct | `MindCubeBench_raw_qa` | 33.21 | 36.08 | 35.65 | -0.43 | ✅ |
| Qwen2.5-VL-7B-Instruct | `MindCubeBench_raw_qa` | 29.26 | 31.12 | 31.48 | +0.36 | ✅ |


### 2. ViewSpatial
* **Metric:** Accuracy

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct | `ViewSpatialBench` | 35.85 | - | 31.97 | -3.88 | ☑️ |
| Qwen2.5-VL-7B-Instruct | `ViewSpatialBench` | 36.85 | - | 36.85 | +0.00 | ✅ |
| InternVL3-14B | `ViewSpatialBench` | 40.28 | - | 40.53 | +0.25 | ✅ |


### 3. EmbSpatial-Bench
* **Metric:** Accuracy

| Model | Benchmark | Paper | Qwen3-VL-Report | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-4B-Instruct | `EmbSpatialBench` | - | 79.60 | 78.70 | -0.90 | ✅ |
| Qwen3-VL-8B-Instruct | `EmbSpatialBench` | - | 78.50 | 77.70 | -0.80 | ✅ |


### 4. MMSI-Bench (no circular)
* **Metric:** Accuracy

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct | `MMSIBench_wo_circular` | 26.50 | - | 28.60 | +2.10 | ✅ |
| Qwen2.5-VL-7B-Instruct | `MMSIBench_wo_circular` | 25.90 | - | 26.80 | +0.90 | ✅ |
| InternVL3-2B | `MMSIBench_wo_circular` | 25.30 | - | 26.50 | +1.20 | ✅ |
| InternVL3-8B | `MMSIBench_wo_circular` | 25.70 | - | 28.00 | +2.30 | ✅ |


### 5. VSI-Bench
* **Metric:** Accuracy && MRA

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct | `VSI-Bench_128frame` | - | 26.80 | 26.60 | -0.20 | ✅ |
| Qwen2.5-VL-7B-Instruct | `VSI-Bench_128frame` | - | 33.50 | 33.70 | +0.20 | ✅ |
| InternVL3_5-8B | `VSI-Bench_128frame` | - | 56.30 | 54.20 | -2.10 | ✅ |
| Cambrian-S-3B | `VSI-Bench_32frame` | - | 54.73 | 56.08 | +1.35 | ✅ |
| Cambrian-S-7B | `VSI-Bench_32frame` | - | 63.61 | 62.93 | -0.68 | ✅ |
| SenseNova-SI-1.1-Qwen3-VL-8B | `VSI-Bench_32frame` | 62.90 | - | 62.90 | +0.00 | ✅ |
| SenseNova-SI-1.2-InternVL3-8B | `VSI-Bench_32frame` | 68.70 | - | 68.70 | +0.00 | ✅ |
| SenseNova-SI-1.1-BAGEL-7B-MoT | `VSI-Bench_32frame` | 41.60 | - | 41.60 | +0.00 | ✅ |

*(For the SenseNova-SI-Qwen series models, VSI-Bench should be evaluated using multiple image pathway)*


### 6. VSI-Bench-Debiased
* **Metric:** Accuracy && MRA

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct | `VSI-Bench-Debiased_128frame` | 22.70 | - | 22.80 | +0.10 | ✅ |
| Qwen2.5-VL-7B-Instruct | `VSI-Bench-Debiased_128frame` | 29.60 | - | 29.10 | -0.50 | ✅ |
| InternVL3_5-8B | `VSI-Bench-Debiased_128frame` | 49.70 | - | 48.40 | -1.30 | ✅ |
| Cambrian-S-3B | `VSI-Bench-Debiased_32frame` | - | 46.47 | 48.76 | +2.29 | ✅ |
| Cambrian-S-7B | `VSI-Bench-Debiased_32frame` | - | 55.58 | 55.35 | -0.23 | ✅ |

### 7. SITE-Bench
* **Metric:** CAA
> **Note:** SiteBench scores here are generally not aligned with the original report. We found issues in the official repo's native interleaved evaluation and fixed them in EASI.

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct | `SiteBenchImage`<br>`SiteBenchVideo_32frame` | 29.50 | - | 33.10 | +3.60 | ☑️ |
| Qwen2.5-VL-7B-Instruct | `SiteBenchImage`<br>`SiteBenchVideo_32frame` | 31.40 | - | 37.6 | +5.3 | ❌ |



### 8. SPAR-Bench
* **Metric:** Accuracy && MRA

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-72B-Instruct | `SparBench_tiny` | 39.40 | - | 39.84 | +0.44 | ✅ |
| Qwen2.5-VL-7B-Instruct | `SparBench` | 33.07 | - | 33.78 | +0.71 | ✅ |
| Qwen2.5-VL-72B-Instruct | `SparBench` | 37.01 | - | 38.94 | +1.93 | ✅ |
| SpaceR-SFT-7B | `SparBench` | 37.55 | - | 34.12 | -3.43 | ☑️ |


### 9. STARE-Bench
* **Metric:** Accuracy && F1 score

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct  | `StareBench_CoT` | 32.3 | - | 33.7 | +1.4 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `StareBench_CoT` | 36.7 | - | 37.6 | +0.9 | ✅ |


### 10. Spatial-Visualization-Benchmark
* **Metric:** Accuracy

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct  | `SpatialVizBench` | 26.10 | 25.00 | 23.98 | -1.02 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `SpatialVizBench` | 30.76 | - | 31.02 | +0.26 | ✅ |
| InternVL3-8B  | `SpatialVizBench` | 30.25 | - | 31.86 | +1.61 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `SpatialVizBench_CoT` | 27.97 | - | 27.54 | -0.43 | ✅ |
| InternVL3-8B  | `SpatialVizBench_CoT` | 30.08 | - | 30.00 | -0.08 | ✅ |


### 11. OmniSpatial
* **Metric:** Accuracy

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct  | `OmniSpatialBench_manual_cot` | 40.30 | 40.73 | 37.70 | -3.03 | ☑️ |
| Qwen2.5-VL-7B-Instruct  | `OmniSpatialBench_manual_cot` | 40.30 | - | 39.18 | -1.12 | ✅ |
| InternVL3-2B  | `OmniSpatialBench_manual_cot` | 37.98 | - | 42.01 | +4.03 | ☑️ |
| InternVL3-8B  | `OmniSpatialBench_manual_cot` | 41.6 | - | 45.34 | +3.74 | ☑️ |


### 12. ERQA
* **Metric:** Accuracy

| Model | Benchmark | Paper | Qwen3-VL-Report | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B-Instruct  | `ERQA` | - | 45.8 | 43 | -2.8 | ☑️ |


### 13. RefSpatial-Bench
* **Metric:** 2D coordinates eval

| Model | Benchmark | Paper | Qwen3-VL-Report | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B-Instruct  | `RefSpatial_wo_unseen` | - | 54.2 | 56.5 | +2.3 | ✅ |


### 14. RoboSpatial-Home
* **Metric:** Accuracy && 2D coordinates eval

| Model | Benchmark | Paper | Qwen3-VL-Report | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B-Instruct  | `RoboSpatialHome` | - | 66.9 | 62.0 | -4.9 | ☑️ |


### 15. SPBench
* **Metric:** Accuracy && MRA

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct  | `SPBench-MV` | 36.6 | - | 38.4 | +1.8 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `SPBench-MV` | 37.3 | - | 40.7 | +3.4 | ☑️ |
| Qwen2.5-VL-3B-Instruct  | `SPBench-SI` | 40.3 | - | 41.2 | +0.9 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `SPBench-SI` | 48.4 | - | 48.1 | -0.3 | ✅ |


### 16. MMSI-Video-Bench
* **Metric:** Accuracy

**Main table:**

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B-Instruct  | `MMSIVideoBench_50frame` | 29.7 | - | 26.9 | -2.8 | ☑️ |
| Qwen3-VL-8B-Instruct  | `MMSIVideoBench_50frame` | 27.6 | - | 28.3 | +0.7 | ✅ |
| InternVL3-8B  | `MMSIVideoBench_50frame` | 30.4 | - | 30.2 | -0.2 | ✅ |
| InternVL3-78B  | `MMSIVideoBench_50frame` | 32.7 | - | 32.6 | -0.1 | ✅ |
| Gemini-3-pro-preview  | `MMSIVideoBench_50frame` | 38.0 | - | 40.4 | +2.4 | ✅ |

<!-- **Sub bench table:**

| Model | Hard | Med| Easy | **Avg** | &nbsp; | Hard(EASI) | Med(EASI) | Easy(EASI) | **Avg(EASI)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| **Qwen2.5-VL-7B** | 11.3 | 29.0 | 46.2 | **29.7** | | 16.2 | 24.8 | 38.3 | **26.9** |
| **Qwen3-VL-8B** | 8.0 | 21.8 | 50.7 | **27.6** | | 11.0 | 25.0 | 46.7 | **28.3** |
| **InternVL3-8B** | 13.8 | 27.5 | 47.8 | **30.4** | | 17.4 | 28.5 | 43.0 | **30.2** |


| Model | IS | Robot | Grd | &nbsp; | IS(EASI) | Robot(EASI) | Grd(EASI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-VL-7B** | 27.1 | 34.8 | 26.6 | | 25.8 | 27.5 | 30.0 |
| **Qwen3-VL-8B** | 28.7 | 27.0 | 28.7 | | 30.2 | 27.0 | 26.6 |
| **InternVL3-8B** | 27.0 | 37.8 | 31.9 | | 28.9 | 35.3 | 31.0 |

*Note: **IS**: Indoor Scene Perception; **Grd**: Grounding.* -->


### 17. VSI-SUPER-Recall
* **Metric:** Accuracy

| Model | Benchmark | Cambrian-S Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Cambrian-S-7B  | `VsiSuperRecall_10mins_128frame` | 26.7 | - | 26.7 | +0.0 | ✅ |
| Cambrian-S-7B  | `VsiSuperRecall_30mins_128frame` | 21.7 | - | 21.7 | +0.0 | ✅ |
| Cambrian-S-7B  | `VsiSuperRecall_60mins_128frame` | 23.3 | - | 23.3 | +0.0 | ✅ |
| Cambrian-S-7B  | `VsiSuperRecall_120mins_128frame` | 30.0 | - | 30.0 | +0.0 | ✅ |
| Cambrian-S-7B  | `VsiSuperRecall_240mins_128frame` | 28.2 | - | 30.0 | +1.8 | ✅ |


### 18. VSI-SUPER-Count (No streaming)
* **Metric:** Accuracy

| Model | Benchmark | Cambrian-S Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Cambrian-S-7B  | `VsiSuperCount_10mins_128frame` | 16.0 | - | 16.2 | +0.2 | ✅ |
| Cambrian-S-7B  | `VsiSuperCount_30mins_128frame` | 0.0 | - | 0.0 | +0.0 | ✅ |


### 19. STI-Bench
* **Metric:** Accuracy

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-72B-Instruct  | `STI-Bench_30frame` | 40.7 | - | 42.1 | +1.4 | ✅ |v


### 20. DSR-Bench
* **Metric:** Accuracy

| Model | Benchmark | Paper | Offical Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B-Instruct  | `DSRBench_1fps` | 23.5 | - |24.7 | +1.2 | ✅ |
| Qwen3-VL-8B-Instruct  | `DSRBench_1fps` | 28.7 | - | 30.6 | +1.9 | ✅ |
| InternVL3_5-8B  | `DSRBench_1fps` | 25.4 | - | 26.6 | +1.2 | ✅ |


### 21. ERIQ
* **Metric:** Accuracy

| Model | Benchmark | Paper | Offical Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct  | `ERIQ` | 58.64 | - | 60.18 | +1.54 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `ERIQ` | 66.79 | - | 67.25 | +0.56 | ✅ |
| Qwen3-VL-8B-Instruct  | `ERIQ` | 75.53 | - | 77.18 | +1.65 | ✅ |
| InternVL3_5-8B  | `ERIQ` | 66.72 | - | 68.34 | +1.62 | ✅ |


### 22. OSI-Bench
* **Metric:** Accuracy && MRA
> **Note:** We also provide `OSI-Bench_visual_first` results because the official implementation places video frames after the text prompt. This variant puts visual information before text for reference.

| Model | Benchmark | Paper | Official Code | EASI (backend=VLMEvalKit) | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-2B-Instruct | `OSI-Bench_32frame` | 18.4 | - | 17.1 | -1.3 | ✅ |
| Qwen3-VL-8B-Instruct | `OSI-Bench_32frame` | 31.2 | - | 31.1 | -0.1 | ✅ |
| InternVL3_5-8B | `OSI-Bench_32frame` | 28.5 | - | 28.0 | -0.5 | ✅ |
|  |  |  |  |  |  |  |
| Qwen3-VL-2B-Instruct | `OSI-Bench_visual_first_32frame` | - | - | 20.7 | - | - |
| Qwen3-VL-8B-Instruct | `OSI-Bench_visual_first_32frame` | - | - | 33.5 | - | - |
| InternVL3_5-8B | `OSI-Bench_visual_first_32frame` | - | - | 28.4 | - | - |
