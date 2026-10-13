# Tokenize COCO captions before computing metrics

## Context
Issue #1308 questions raw-caption scoring and reports unexpected COCO metrics. Current COCO_Caption_Scorer passes raw strings to BLEU, ROUGE and CIDEr. The official COCO evaluator applies Stanford PTB tokenization to both reference and predicted captions first, normalizing case, punctuation and contractions.

## Decision
Apply pycocoevalcap's PTBTokenizer to both input dictionaries before the existing metrics. Adapt strings to the tokenizer's caption-record shape without mutating caller dictionaries. Keep the metric set, output names and existing scale unchanged.

## Alternatives
A whitespace split or jieba would implement a different protocol. Changing the metrics themselves would conflate tokenization with score definitions. Fixing prediction tokenization alone leaves reference tokenization inconsistent.

## Consequences
Scores follow the official preprocessing and can differ from raw-string historical scores. A working Java runtime on PATH is required by Stanford PTBTokenizer, in addition to the already-used optional pycocoevalcap package; Quickstart documents this requirement. No provider API or learned-model inference runs during tokenization. This does not claim to reproduce or resolve every cause of the issue's full dataset metrics.

## Validation
Primary COCO evaluator pinned at 3a9afb2682141a03e1cdc02b0df6770d2c884f6f: https://github.com/tylin/coco-caption/blob/3a9afb2682141a03e1cdc02b0df6770d2c884f6f/pycocoevalcap/eval.py. Regression cases compare the actual scorer against the real PTBTokenizer and metric implementations on two-caption corpora with case/punctuation and contractions; baseline differs from the reference pipeline. Tests also check that caller reference dictionaries remain unchanged. Local Java and library versions, baseline/candidate logs and lint checks are retained in campaign evidence. No downloaded COCO corpus, GPU model run or full reported dataset score is claimed.

Candidate: the two real-library reference-pipeline cases pass with pycocoevalcap 1.2 and local Java 1.8.0_471. Tests declare the optional package/Java requirement and skip clearly when absent; this campaign executes both with zero skips. New code/test lint and diff whitespace checks pass. Existing scoring regression checks are run alongside these cases in final evidence.

Final impacted run: six tests pass (two caption reference comparisons and four existing judge-score controls), zero skips. No full corpus/model accuracy change is asserted.
