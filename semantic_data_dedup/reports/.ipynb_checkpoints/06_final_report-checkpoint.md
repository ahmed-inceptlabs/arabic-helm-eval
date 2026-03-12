# Dataset Quality & Deduplication Report
**Source:** `ar93_en7_mcq85_open15_cot96_211k.jsonl`
**Total rows:** 211,075
**Final clean rows:** 200,428
**Total removed:** 10,647 (5.04%)

## 1. Profile Summary
- Arabic user prompts: 92.7% (expected ~93%)
- MCQ fraction: 23.3% (expected ~85%)
- CoT (<think>) fraction: 95.7% (expected ~96%)
- User prompt length: mean=665.9, min=10, max=19922
- Assistant length: mean=1027.9, min=4, max=13265

## 2. Exact Dedup (Phase 3)
- **L1_raw_pair**: 211,075 unique, 0 removable (0.00%)
- **L2_ws_pair**: 211,075 unique, 0 removable (0.00%)
- **L3_think_stripped_pair**: 211,058 unique, 17 removable (0.01%)
- **L4_user_only**: 211,053 unique, 22 removable (0.01%)
- **L5_user_arabic_canon**: 210,899 unique, 176 removable (0.08%)
- Output (L3): kept 211,058, removed 17

## 3. Semantic Dedup (Phase 4)
- Clusters: 3,508
- Rows in clusters: 13,947
- Removable: 10,439
- Similarity stats: mean=0.7697, median=0.7798, p95=0.9479, p99=0.9924

| Band | Count |
|------|-------|
| 0.00-0.50 | 5,968 |
| 0.50-0.70 | 36,812 |
| 0.70-0.80 | 84,033 |
| 0.80-0.85 | 40,265 |
| 0.85-0.90 | 22,214 |
| 0.90-0.93 | 7,819 |
| 0.93-0.95 | 3,812 |
| 0.95-0.96 | 1,622 |
| 0.96-0.97 | 1,760 |
| 0.97-0.98 | 1,898 |
| 0.98-0.99 | 2,151 |
| 0.99-1.01 | 2,704 |

## 4. Leakage Check (Phase 5)
- Direct benchmark mentions: 182
- Rubric leak rows: 31,914
- Total flagged (heuristic): 167,268

| Benchmark | Flagged | % |
|-----------|---------|---|
| aratrust | 3,963 | 1.9% |
| arabic_mmlu | 26,692 | 12.7% |
| alghafa | 128,644 | 61.0% |
| arabic_exams | 37,849 | 17.9% |
| arabic_mmmlu | 4 | 0.0% |
| alrage | 8 | 0.0% |

## 5. Final Removal Breakdown

| Reason | Rows Removed |
|--------|--------------|
| semantic_duplicate | 10,439 |
| leakage_high_confidence | 200 |
| exact_duplicate | 17 |
| **Total unique removed** | **10,647** |

## 6. Output Files
- `data/06_clean.jsonl` — 200,428 rows
- `data/06_removal_log.jsonl` — 10,647 removal entries
