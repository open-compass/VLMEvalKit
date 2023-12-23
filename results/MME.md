# MME Evaluation Results

> No specific prompt template adopted for **ALL VLMs**.

### MME Scores (Vanilla / ChatGPT Answer Extraction)

In each cell, we list `vanilla score / ChatGPT Answer Extraction Score` if the two scores are difference. Otherwise, we only list one score.

VLMs are sorted by the descending order of Total score.

| Model                |       Total |  perception | reasoning |
| :------------------- | ----------: | ----------: | --------: |
| Full                 |        2800 |        2000 |       800 |
| GeminiProVision      | 2131 / 2149 | 1601 / 1609 | 530 / 540 |
| XComposer            |        1874 |        1497 |       377 |
| qwen_chat            | 1849 / 1860 | 1457 / 1468 |       392 |
| sharegpt4v_7b        | 1799 / 1808 |        1491 | 308 / 318 |
| llava_v1.5_13b       | 1800 / 1805 | 1485 / 1490 |       315 |
| mPLUG-Owl2           | 1781 / 1786 | 1435 / 1436 | 346 / 350 |
| llava_v1.5_7b        |        1775 |        1490 |       285 |
| GPT-4v (detail: low) | 1737 / 1771 | 1300 / 1334 |       437 |
| TransCore_M          | 1682 / 1701 | 1427 / 1429 | 254 / 272 |
| instructblip_13b     | 1624 / 1646 | 1381 / 1383 | 243 / 263 |
| idefics_80b_instruct | 1507 / 1519 | 1276 / 1285 | 231 / 234 |
| instructblip_7b      | 1313 / 1391 | 1084 / 1137 | 229 / 254 |
| idefics_9b_instruct  |        1177 |         942 |       235 |
| PandaGPT_13B         |        1072 |         826 |       246 |
| MiniGPT-4-v1-13B     |  648 / 1067 |   533 / 794 | 115 / 273 |
| MiniGPT-4-v1-7B      |  806 / 1048 |   622 / 771 | 184 / 277 |
| llava_v1_7b          | 1027 / 1044 |   793 / 807 | 234 / 238 |
| MiniGPT-4-v2         |         968 |         708 |       260 |
| VisualGLM_6b         |         738 |         628 |       110 |
| flamingov2           |         607 |         535 |        72 |
| qwen_base            |     6 / 483 |     0 / 334 |   6 / 149 |

### Comments

For most VLMs, using ChatGPT as the answer extractor or not may not significantly affect the final score. However, for some VLMs including instructblip_7b, MiniGPT-4-v1, and qwen_base, the score improvement with ChatGPT answer extractor is significant. The table below demonstrates the score gap between two answer extraction strategies: 

| MME Score Improvement with ChatGPT Answer Extractor | Models                                                       |
| --------------------------------------------------- | ------------------------------------------------------------ |
| **No (0)**                                          | XComposer, llava_v1.5_7b, idefics_9b_instruct, PandaGPT_13B, MiniGPT-4-v2, <br>VisualGLM_6b, flamingov2 |
| **Minor (1~20)**                                    | qwen_chat (11), llava_v1.5_13b (5), mPLUG-Owl2 (5), idefics_80b_instruct (12), llava_v1_7b (17), <br>sharegpt4v_7b (9), TransCore_M (19), GeminiProVision (18) |
| **Moderate (21~100)**                               | instructblip_13b (22), instructblip_7b (78), GPT-4v (34)     |
| **Huge (> 100)**                                    | MiniGPT-4-v1-7B (242), MiniGPT-4-v1-13B (419), qwen_base (477) |