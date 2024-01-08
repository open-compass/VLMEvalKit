# MME Evaluation Results

> No specific prompt template adopted for **ALL VLMs**.

### MME Scores (Vanilla / ChatGPT Answer Extraction)

In each cell, we list `vanilla score / ChatGPT Answer Extraction Score` if the two scores are difference. Otherwise, we only list one score.

VLMs are sorted by the descending order of Total score.

| Model                         | Total       | Perception   | Reasoning   |
|:------------------------------|:------------|:-------------|:------------|
| GeminiProVision               | 2131 / 2149 | 1601 / 1609  | 530 / 540   |
| InternLM-XComposer-VL         | 1874        | 1497         | 377         |
| Qwen-VL-Chat                  | 1849 / 1860 | 1457 / 1468  | 392         |
| ShareGPT4V-7B                 | 1799 / 1808 | 1491         | 308 / 317   |
| LLaVA-v1.5-13B                | 1800 / 1805 | 1485 / 1490  | 315         |
| mPLUG-Owl2                    | 1781 / 1786 | 1435 / 1436  | 346 / 350   |
| LLaVA-v1.5-7B                 | 1775        | 1490         | 285         |
| GPT-4v (detail: low)          | 1737 / 1771 | 1300 / 1334  | 437         |
| LLaVA-v1.5-13B (LoRA, XTuner) | 1766        | 1475         | 291         |
| CogVLM-17B-Chat               | 1727 / 1737 | 1437 / 1438  | 290 / 299   |
| LLaVA-v1.5-7B (LoRA, XTuner)  | 1716        | 1434         | 282         |
| TransCore-M                   | 1681 / 1701 | 1427 / 1429  | 254 / 272   |
| instructblip_13b              | 1624 / 1646 | 1381 / 1383  | 243 / 263   |
| SharedCaptioner               | 1592 / 1643 | 1247 / 1295  | 345 / 348   |
| LLaVA-InternLM-7B (LoRA)      | 1637        | 1393         | 244         |
| IDEFICS-80B-Instruct          | 1507 / 1519 | 1276 / 1285  | 231 / 234   |
| InstructBLIP-7B               | 1313 / 1391 | 1084 / 1137  | 229 / 254   |
| IDEFICS-9B-Instruct           | 1177        | 942          | 235         |
| PandaGPT-13B                  | 1072        | 826          | 246         |
| MiniGPT-4-v1-13B              | 648 / 1067  | 533 / 794    | 115 / 273   |
| MiniGPT-4-v1-7B               | 806 / 1048  | 622 / 771    | 184 / 277   |
| LLaVA-v1-7B                   | 1027 / 1044 | 793 / 807    | 234 / 237   |
| MiniGPT-4-v2                  | 968         | 708          | 260         |
| VisualGLM                     | 738         | 628          | 110         |
| OpenFlamingo v2               | 607         | 535          | 72          |
| Qwen-VL                       | 6 / 483     | 0 / 334      | 6 / 149     |

### Comments

For most VLMs, using ChatGPT as the answer extractor or not may not significantly affect the final score. However, for some VLMs including instructblip_7b, MiniGPT-4-v1, and qwen_base, the score improvement with ChatGPT answer extractor is significant. The table below demonstrates the score gap between two answer extraction strategies: 

| MME Score Improvement with ChatGPT Answer Extractor | Models                                                       |
| --------------------------------------------------- | ------------------------------------------------------------ |
| **No (0)**                                          | XComposer, llava_v1.5_7b, idefics_9b_instruct, PandaGPT_13B, MiniGPT-4-v2, <br>VisualGLM_6b, flamingov2, LLaVA-XTuner Series |
| **Minor (1~20)**                                    | qwen_chat (11), llava_v1.5_13b (5), mPLUG-Owl2 (5), idefics_80b_instruct (12), llava_v1_7b (17), <br>sharegpt4v_7b (9), TransCore_M (19), GeminiProVision (18), CogVLM-17B-Chat (10) |
| **Moderate (21~100)**                               | instructblip_13b (22), instructblip_7b (78), GPT-4v (34), SharedCaptioner (51)  |
| **Huge (> 100)**                                    | MiniGPT-4-v1-7B (242), MiniGPT-4-v1-13B (419), qwen_base (477) |