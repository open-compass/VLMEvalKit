# TrafficQA Dataset

TrafficQA is a video question answering benchmark for traffic scenes from CVPR 2021. It evaluates models on 6 types of reasoning tasks: Basic Understanding (U), Attribution (A), Event Forecasting (F), Reverse Reasoning (R), Counterfactual Inference (C), and Introspection (I).

## Dataset Download

**IMPORTANT: MANUAL DOWNLOAD REQUIRED**

Due to licensing restrictions, the TrafficQA dataset must be downloaded manually from the official source:

1. **Official Download Page**: https://sutdcv.github.io/SUTD-TrafficQA/
2. **Dataset Size**: ~12GB (10,080 compressed videos)
3. **Required Files**:
   - `annotations/R2_all.jsonl` (all 62,533 QA pairs)
   - `annotations/R2_train.jsonl` (56,459 QA pairs for training)
   - `annotations/R2_test.jsonl` (6,074 QA pairs for evaluation)
   - `compressed_videos/*.mp4` (10,080 video files)

After downloading, set the `TRAFFICQA_DATA_PATH` environment variable:

```bash
export TRAFFICQA_DATA_PATH=/path/to/SUTD-TrafficQA
```

Or ensure the dataset is at the default path: `/storage/disk3/datasets/SUTD-TrafficQA/`

Expected directory structure:
```
SUTD-TrafficQA/
├── annotations/
│   ├── R2_all.jsonl
│   ├── R2_train.jsonl
│   └── R2_test.jsonl
└── compressed_videos/
    └── *.mp4 (10,080 files)
```

## Dataset Overview

| Property | Value |
|----------|-------|
| **Paper** | [TrafficQA: A Benchmark Dataset for Question Answering on Traffic Scenes](https://arxiv.org/abs/2103.15247) (CVPR 2021) |
| **Task** | Video Question Answering (Multiple Choice) |
| **Format** | Video + Text Question → Option Letter (A/B/C/D) |
| **Videos** | 10,080 traffic videos (12GB) |
| **Questions** | 62,533 QA pairs total |
| **Test Set** | 6,075 QA pairs (recommended for evaluation) |

## Dataset Structure

```
/storage/disk3/datasets/SUTD-TrafficQA/
├── annotations/
│   ├── R2_all.jsonl      (62,533 QA pairs)
│   ├── R2_train.jsonl    (56,459 QA pairs)
│   └── R2_test.jsonl     (6,074 QA pairs)
└── compressed_videos/
    └── 10,080 .mp4 files
```

## Key Dataset Characteristics

### 1. Variable Option Positioning

**42.3% of entries have non-contiguous options!** Options can be in any positions:

```json
// Example 1: option1, option3 filled (B, D positions)
["...", "Does the road have clear markings?", "U",
 "", "Yes, the road is marked.", "No, the road is unmarked.", "Partially marked.", 1]
// Answer: 1 → option1 (B)

// Example 2: option2, option3 filled (C, D positions)
["...", "What caused the accident?", "U",
 "", "Violation of rules", "Sharp lane change", "Weather conditions", 2]
// Answer: 2 → option2 (C)
```

**Implementation**: We preserve original option positions and map the answer index (0-3) to the corresponding letter (A-D).

### 2. Train/Test Video Overlap

**99.5% of test QAs come from videos that are ALSO in the training set.**

This is by design - TrafficQA follows the "same video, different questions" paradigm used in other VQA datasets like VQA v2 and GQA. The same traffic video can have multiple questions testing different reasoning aspects.

| Split | Questions | Videos | Overlap with Test |
|-------|-----------|--------|-------------------|
| Train | 56,459 | 10,051 | 4,082 videos (99.3%) |
| Test | 6,075 | 4,111 | - |

### 3. Six Reasoning Tasks

| Code | Task | Percentage | Description |
|------|------|------------|-------------|
| U | Basic Understanding | 59.2% | Direct comprehension of traffic scenes |
| A | Attribution | 26.7% | Identifying causes and reasons |
| R | Reverse Reasoning | 4.9% | Inferring preceding events |
| F | Event Forecasting | 3.7% | Predicting what happens next |
| C | Counterfactual Inference | 3.0% | Reasoning about hypothetical scenarios |
| I | Introspection | 2.4% | Understanding agent intentions |

## Usage

### Basic Evaluation

```bash
# Evaluate with 8 frames (default)
python run.py --data TrafficQA_test_8frame --model Qwen3-VL-4B-Instruct

# Evaluate with 2fps sampling
python run.py --data TrafficQA_test_2fps --model Qwen3-VL-4B-Instruct

# Evaluate with 64 frames
python run.py --data TrafficQA_test_64frame --model <model_name>
```

### Frame Sampling Options

**Fixed Frame Count:**
- `TrafficQA_test_8frame` - 8 frames uniformly sampled
- `TrafficQA_test_16frame` - 16 frames uniformly sampled
- `TrafficQA_test_32frame` - 32 frames uniformly sampled
- `TrafficQA_test_64frame` - 64 frames uniformly sampled

**FPS-based Sampling:**
- `TrafficQA_test_0.5fps` - 0.5 frames per second
- `TrafficQA_test_1fps` - 1 frame per second
- `TrafficQA_test_2fps` - 2 frames per second

### Custom Dataset Path

Set the `TRAFFICQA_DATA_PATH` environment variable if your dataset is in a different location:

```bash
export TRAFFICQA_DATA_PATH=/path/to/SUTD-TrafficQA
python run.py --data TrafficQA_test_8frame --model <model_name>
```

## Output Format

The evaluation produces results with:

1. **Overall Accuracy** - Performance across all 6,075 test questions

2. **Per-Reasoning-Task Accuracy** - Performance for each of the 6 reasoning types:
   - `basic_understanding` (U)
   - `attribution` (A)
   - `event_forecasting` (F)
   - `reverse_reasoning` (R)
   - `counterfactual_inference` (C)
   - `introspection` (I)

Example output:
```
TrafficQA Evaluation Results:
Overall Accuracy: 0.7234 (4394/6075)

Per-Reasoning-Task Accuracy:
  U (Basic Understanding): 0.7541 (2711/3595)
  A (Attribution): 0.7028 (1142/1625)
  R (Reverse Reasoning): 0.6856 (205/299)
  F (Event Forecasting): 0.6323 (141/223)
  C (Counterfactual Inference): 0.6811 (126/185)
  I (Introspection): 0.7027 (104/148)
```

## Implementation Details

### File Location

- Dataset implementation: `vlmeval/dataset/trafficqa.py`
- Configuration: `vlmeval/dataset/video_dataset_config.py`

### Key Classes

```python
class TrafficQA(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    DATASET_PATH = '/storage/disk3/datasets/SUTD-TrafficQA'

    def __init__(self, dataset='TrafficQA', split='test', nframe=0, fps=-1):
        # split: 'test', 'train', or 'all'
        # nframe: fixed number of frames (mutually exclusive with fps)
        # fps: frames per second for sampling (mutually exclusive with nframe)
```

### Critical Implementation Features

1. **JSONL Array Parsing**: TrafficQA uses JSONL array format (not JSON objects)
2. **Bidirectional Option Mapping**: Handles variable option positioning correctly
3. **Answer Index to Letter**: Converts 0,1,2,3 to A,B,C,D for model output
4. **Per-Task Evaluation**: Computes metrics for each of the 6 reasoning types

## Citation

If you use TrafficQA in your research, please cite:

```bibtex
@inproceedings{trafficqa2021,
  title={TrafficQA: A Benchmark Dataset for Question Answering on Traffic Scenes},
  author={Cheng, Ta and others},
  booktitle={CVPR},
  year={2021}
}
```

## Notes

- The test set should be used for standard evaluation
- Document the train/test video overlap when reporting results
- Per-reasoning-task metrics provide insight into model capabilities across different reasoning types
