# Prepare GMAI-MMBench validation data before loading

## Context

Issue #1173 reports a fresh run failing because GMAI-MMBench_VAL.tsv is missing. The dataset declares a download URL and checksum, but its validation branch bypasses the common download/checksum preparation and reads directly from the cache. Existing cached data is also accepted without checksum validation.

## Decision

Delegate the validation branch to the inherited ImageBaseDataset.load_data implementation. It selects the declared URL and checksum, prepares the cache, and preserves the existing localization behavior for large TSVs. The multipart test dataset branch remains unchanged.

## Alternatives

Duplicating download/checksum code in this branch would create another cache implementation to maintain. Requiring users to manually download validation data would leave the declared dataset inconsistent with the other image datasets.

## Consequences

Fresh validation runs obtain the declared TSV; stale caches use the existing checksum-refresh path; valid caches remain reusable. Network/download errors retain the base loader's existing behavior. No URLs, checksum values, dataset records, test-shard assembly or evaluation scores change.

## Validation

Frozen upstream ec4a4ed412456a4d774e57974b21f90c80b2f519: the corrected harness produces two failures and one pass for missing, stale and valid caches. Candidate: these three tests plus four existing judge-score tests pass (seven total). The stale case emits the existing checksum refresh warning.

The tests execute the actual dataset load_data and common prepare_tsv methods extracted through the AST, using real temporary TSVs, pandas parsing and MD5 checks. The download boundary writes a local synthetic TSV, so no medical dataset was downloaded or model/provider called. Large-file localization, multipart test downloads, GPU inference and full dataset integration were not executed. Source lint is compared against upstream; pre-existing findings are retained. New test lint and git diff --check pass.

Initial harness runs errored because the extracted base class retained its ABCMeta keyword. Those errors are preserved in private logs and excluded from regression evidence; the corrected harness clears that unrelated class-construction requirement. Native execution costs are unmeasured; no provider spend was incurred.
