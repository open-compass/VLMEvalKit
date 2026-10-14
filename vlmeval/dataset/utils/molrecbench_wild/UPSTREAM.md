# MolRecBench-Wild evaluator provenance

The evaluator and prediction converter in this directory are derived from:

- Project: [opendatalab/MolRecBench-Wild](https://github.com/opendatalab/MolRecBench-Wild)
- Fixed commit: `500da87d767ba8ea48b1fec5f765a01e8c9a2394`
- Source-code license: Apache License 2.0
- Upstream license SHA-256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- Retrieved for this integration: 2026-08-20

VLMEvalKit is also distributed under Apache License 2.0; its root `LICENSE`
contains the applicable license text. The fixed upstream tree refers to a
`THIRD_PARTY_NOTICES.md`, but that file is absent from the tree at this commit.
This integration therefore vendors source code only. It does not vendor the
benchmark data, molecule images, prediction files, or prompt raster assets.

## Vendored files

| Local file | Upstream file | Upstream SHA-256 |
| --- | --- | --- |
| `constants.py` | `evaluate/constants.py` | `2c46a67de6430840c5f56fb3d10fc5ec424a23fd429ea4db8b0fb0dd062c5a1e` |
| `utils.py` | `evaluate/utils.py` | `911f7bf1a2a9946eeb05a93de6e6d25f1fce88dc6858a467de4a95f5233f8f61` |
| `mol_graph.py` | `evaluate/MolGraph.py` | `3b4cc3b86710124a1c60e085460fca7e2ca01d97997140d3856e23eacc8e3838` |
| `smiles_metric.py` | `evaluate/smiles_metric.py` | `6aaf9be278333a7ced6586cfeda5555e64ac6fca6dc488f90afc6488aef079e3` |
| `evaluator.py` | `evaluate/Evaluator.py` | `4d90f5eefde687c0a0ecae1980177dd2bb0a22ed0327ce846be566c88a8fb6d7` |
| `converter.py` | `scripts/convert_result.py` | `d2a0ea3389d7192d03cfd40a964075323fbea0bebc8efe7c23ff3f807036cc16` |

The hashes describe the unmodified upstream inputs, before the adaptations
below.

## VLMEvalKit adaptations

- Changed package-relative imports and lower-cased `MolGraph.py` for the local
  package layout.
- Removed the converter's command-line `sys.path` mutation.
- Added track-name normalization, strict string-ID validation, and a tolerant
  one-record conversion API.
- Added an in-memory records scoring facade. It delegates to the upstream
  SMILES and Graph/S-Graph evaluators in-process and returns both the upstream
  `Full`/`A`/`B`/`C` aggregates and per-record details.
- Added a dependency-free package facade so importing the package does not
  import RDKit or NetworkX. Optional chemistry dependencies are loaded only
  when scoring starts.

These adaptations do not change bond normalization, molecular graph matching,
bracket comparison, SMILES expansion/canonicalization, R-group handling, GT
filtering, or accuracy denominators.

## Golden parity check

On 2026-08-20 the records API was run against all 5,024 fixed-release GT
records and the upstream `GPT-5.6-sol` predictions with Python 3.10.20,
NetworkX 3.4.2, and RDKit 2025.09.5. It reproduced:

- SMILES: `1791 / 2392 = 0.7487458193979933`
- S-Graph: `1828 / 5024 = 0.3638535031847134`
- Graph with the timeout disabled: `1636 / 5024 = 0.32563694267515925`

Each run returned 5,024 per-record details, and its A/B/C correct, scored, and
accuracy values matched the fixed upstream evaluator.

## Prompt hashes (not vendored here)

Runtime data/prompt preparation can validate fixed-commit downloads against:

- `prompts/smiles.txt`: `bd121c7613b74e5ddce498e78a49068e88d2fe49b9b7884dc6f7b481b6dd8ba0`
- `prompts/graph_simple.txt`: `3de24344e17ab6e232c92c870cab5213c2553cf20bc523aa7043b389aeb8cd11`
- `prompts/graph.txt`: `3d160dfebc9a44a5b39141ec79b2290059b3a77746b2913dd7db334268ea3505`
- `prompts/cases.png`: `a49ba87e886b1ec212b187f937f01f05351c216a7389dddbb0855801bc5ce010`
- `prompts/visual_example.png`: `43dd6fac823d26cf622f2abc11ae2fe7e9e4de59bebf45653d8f28bc6a8a7224`
