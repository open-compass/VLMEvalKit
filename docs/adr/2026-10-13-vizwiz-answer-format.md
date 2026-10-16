# Ask for VizWiz short answers and unanswerable cases

## Context
Issue #1090 reports default VizWiz prompts producing descriptive responses that do not match its short-answer VQA scoring contract. Four participants corroborate the missing instructions. Unlike ImageVQADataset, VizWiz inherits the bare question builder.

## Decision
Append the short-answer instruction and explicit Unanswerable instruction to the default VizWiz text message. Preserve the base builder's images, row/index handling and source question. Follow the primary lmms-eval VizWiz template at commit 32d527db1e1359a3c3d0162a2df09cb286a8e280: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/32d527db1e1359a3c3d0162a2df09cb286a8e280/lmms_eval/tasks/vizwiz_vqa/_default_template_vqa_yaml.

## Alternatives
Changing the scorer to accept arbitrary descriptions changes benchmark semantics. Changing the common image builder affects unrelated datasets. A VizWiz-specific override addresses the reported prompt/scorer mismatch directly.

## Consequences
Default VizWiz prompts state both output format and the insufficient-information response. Inference prompting changes; old and new runs should record that difference. No quality or accuracy improvement is newly measured here. Existing model-specific custom prompts continue their own path.

## Validation
Frozen upstream ec4a4ed412456a4d774e57974b21f90c80b2f519 lacks the instructions in all four combinations of indexed/row input and metadata/localized image paths. Tests execute the actual base and VizWiz prompt methods through isolated AST loading, verify both image references and unchanged input questions, and require the cited output contract. No model, GPU or provider calls were made. Raw baseline/candidate and lint evidence stay local.

Candidate: six tests pass across the four prompt cases and two existing image helper controls. The new test file passes flake8; full source lint retains the same 83 pre-existing diagnostics after accounting for shifted line numbers. No new source diagnostic was introduced. Diff whitespace validation passes.
