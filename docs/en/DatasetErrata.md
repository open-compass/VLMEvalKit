# Dataset errata

The MD5-pinned TSV releases reported in GitHub Issue #1754 contain known
ambiguous or inconsistent annotations. VLMEvalKit excludes only the indices
listed in that issue from inference and evaluation. It does not scan question
content, rewrite answers, or change the answer-matching rules.

The explicit dataset names, release MD5s, and indices are recorded in
`vlmeval/dataset/utils/data_errata.py`. The main issue list contains 112 rows
across 17 TSVs; the four rows also mentioned for `MMT-Bench_VAL_MI` bring the
covered total to 116 rows across 18 TSVs. The six MTL-MMBench tables use the
registered `MMBench_dev_<lang>` names.

For MMBench, circular variants of the listed questions are excluded together,
using the existing `g_index` or `index % 1000000` grouping convention. Other
datasets use exact indices only. MMBench v1.1, unlisted datasets, and unlisted
questions are unaffected. The exclusions are disabled if the registered release
MD5 changes, so a corrected future release does not inherit an old blacklist.

Filtering happens after image references are resolved, so a retained question
can still use an image originally stored on an excluded row. Evaluation also
filters old prediction files using the same errata; the original hosted TSVs
and prediction contents are not rewritten by the errata filter.

Warnings report the number of excluded rows. These scores use a reduced sample
set and should not be presented as directly comparable to scores computed on
the unfiltered releases. Previously cached evaluation results and aggregate
scores should be regenerated rather than reused.
