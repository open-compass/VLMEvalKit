# MME Evaluation Results

> No specific prompt template adopted for **ALL VLMs**.

### MME Scores (Vanilla / ChatGPT Answer Extraction)

In each cell, we list `vanilla score / ChatGPT Answer Extraction Score` if the two scores are difference. Otherwise, we only list one score.

VLMs are sorted by the descending order of Total score.

| Model                       | Total       | Perception   | Reasoning   |
|:----------------------------|:------------|:-------------|:------------|
| QwenVLPlus                  | 2152 / 2229 | 1684 / 1692  | 468 / 537   |
| GeminiProVision             | 2131 / 2149 | 1601 / 1609  | 530 / 540   |
| TransCore-M                 | 1890 / 1898 | 1593 / 1594  | 297 / 304   |
| Monkey-Chat                 | 1888        | 1506         | 382         |
| ShareGPT4V-7B               | 1872 / 1874 | 1530         | 342 / 344   |
| InternLM-XComposer-VL       | 1874        | 1497         | 377         |
| LLaVA-InternLM2-20B (QLoRA) | 1867        | 1512         | 355         |
| Qwen-VL-Chat                | 1849 / 1860 | 1457 / 1468  | 392         |
| ShareGPT4V-13B              | 1828        | 1559         | 269         |
| LLaVA-v1.5-13B              | 1800 / 1805 | 1485 / 1490  | 315         |
| mPLUG-Owl2                  | 1781 / 1786 | 1435 / 1436  | 346 / 350   |
| LLaVA-v1.5-7B               | 1775        | 1490         | 285         |
| GPT-4v (detail: low)        | 1737 / 1771 | 1300 / 1334  | 437         |
| LLaVA-v1.5-13B (QLoRA)      | 1766        | 1475         | 291         |
| Monkey                      | 1760        | 1472         | 288         |
| CogVLM-17B-Chat             | 1727 / 1737 | 1437 / 1438  | 290 / 299   |
| LLaVA-v1.5-7B (QLoRA)       | 1716        | 1434         | 282         |
| EMU2-Chat                   | 1653 / 1678 | 1322 / 1345  | 331 / 333   |
| InstructBLIP-13B            | 1624 / 1646 | 1381 / 1383  | 243 / 263   |
| SharedCaptioner             | 1592 / 1643 | 1247 / 1295  | 345 / 348   |
| LLaVA-InternLM-7B (QLoRA)   | 1637        | 1393         | 244         |
| IDEFICS-80B-Instruct        | 1507 / 1519 | 1276 / 1285  | 231 / 234   |
| InstructBLIP-7B             | 1313 / 1391 | 1084 / 1137  | 229 / 254   |
| IDEFICS-9B-Instruct         | 1177        | 942          | 235         |
| PandaGPT-13B                | 1072        | 826          | 246         |
| MiniGPT-4-v1-13B            | 648 / 1067  | 533 / 794    | 115 / 273   |
| MiniGPT-4-v1-7B             | 806 / 1048  | 622 / 771    | 184 / 277   |
| LLaVA-v1-7B                 | 1027 / 1044 | 793 / 807    | 234 / 237   |
| MiniGPT-4-v2                | 968         | 708          | 260         |
| VisualGLM                   | 738         | 628          | 110         |
| OpenFlamingo v2             | 607         | 535          | 72          |
| Qwen-VL                     | 6 / 483     | 0 / 334      | 6 / 149     |

### Comments

For most VLMs, using ChatGPT as the answer extractor or not may not significantly affect the final score. However, for some VLMs including instructblip_7b, MiniGPT-4-v1, and qwen_base, the score improvement with ChatGPT answer extractor is significant. The table below demonstrates the score gap between two answer extraction strategies: 

| MME Score Improvement with ChatGPT Answer Extractor | Models                                                       |
| --------------------------------------------------- | ------------------------------------------------------------ |
| **Minor (1~20)**                                    | Qwen-VL-Chat (11), LLaVA-v1.5-13B (5), mPLUG-Owl2 (5), IDEFICS-80B-Instruct (12), LLaVA-v1-7B (17), <br>ShareGPT4V-7B (2), TransCore_M (8), GeminiProVision (18), CogVLM-17B-Chat (10) |
| **Moderate (21~100)**                               | InstructBLIP-13B (22), InstructBLIP-7B (78), GPT-4v (34), SharedCaptioner (51), QwenVLPlus (77), <br>EMU2-Chat (25), SharedCaptioner (51)|
| **Huge (> 100)**                                    | MiniGPT-4-v1-7B (242), MiniGPT-4-v1-13B (419), Qwen-VL (477) |