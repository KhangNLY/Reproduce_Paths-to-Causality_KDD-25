# Kết quả của project vs bài báo (F1 %, protocol `all`)

_tạo bởi tools/compare_with_paper.py lúc 2026-11-04T23:51:38+07:00_

* Bài báo: mistral-7b-instruct, llama-3-8b-instruct, gemma-7b-it (chạy local, prompt thô, greedy, 6 token). Project: llama-3.1-8b-instruct, ministral-8b-2512 (OpenRouter chat + chỉ dẫn 1 từ).
* **Protocol `all`** (không rò rỉ nhãn) không có số tương ứng trong bài: cột `paper` là số của bài (protocol oracle_top5) chỉ để tham chiếu, Δ không phải so sánh cùng điều kiện.
* `paper(avg)` = trung bình 3 LLM của bài; `ours(avg)` = trung bình các LLM của project. **Không kỳ vọng khớp từng số** (khác LLM, khác cách gọi, test chỉ 24-38 cặp).

## Tóm tắt: F1 trung bình theo LLM (paper = 3 LLM, ours = 2 LLM)

| method | COMAGC paper / ours | ADE paper / ours | GENEC paper / ours | SEMEVAL paper / ours | **avg paper / ours** |
|---|---|---|---|---|---|
| no-subgraph (pair + context) | 46.2 / 75.0 | 66.6 / 95.2 | 33.8 / 46.8 | 50.9 / 73.2 | **49.4 / 72.6** |
| random | 68.4 / 78.7 | 79.5 / 88.0 | 40.4 / 44.7 | 59.2 / 70.9 | **61.9 / 70.6** |
| similarity (MiniLM) | 69.6 / 76.2 | 82.0 / 88.8 | 42.0 / 47.0 | 64.7 / 72.0 | **64.6 / 71.0** |
| GPT-based (RankGPT) | 79.0 / 78.0 | 83.8 / 90.4 | 44.8 / 47.0 | 62.8 / 70.9 | **67.6 / 71.6** |
| Ranker: RoBERTa-RMSE | 76.8 / 79.0 | 83.6 / 90.5 | 44.7 / 42.2 | 67.7 / 70.9 | **68.2 / 70.7** |
| Ranker: RoBERTa-RankNet | 78.1 / 80.5 | 85.7 / 87.6 | 52.8 / 47.7 | 62.1 / 73.6 | **69.7 / 72.4** |
| Ranker: RoBERTa-ListNet | 75.8 / 75.4 | 84.4 / 90.5 | 49.7 / 46.4 | 61.4 / 70.9 | **67.8 / 70.8** |
| Ranker: XGBoost | 74.8 / 77.3 | 83.7 / 86.9 | 42.1 / 45.6 | 64.6 / 70.9 | **66.3 / 70.2** |

## COMAGC

| method | paper mistral | paper llama | paper gemma | **paper(avg)** | ours llama-3.1-8b-instruct | ours ministral-8b-2512 | **ours(avg)** | Δ(ours−paper) |
|---|---|---|---|---|---|---|---|---|
| no-subgraph (pair + context) | 55.2 | 33.3 | 50.0 | **46.2** | 71.1 | 79.0 | **75.0** | +28.9 |
| random | 66.7 | 62.9 | 75.6 | **68.4** | 79.1 | 78.3 | **78.7** | +10.3 |
| similarity (MiniLM) | 66.7 | 64.9 | 77.3 | **69.6** | 75.6 | 76.9 | **76.2** | +6.6 |
| GPT-based (RankGPT) | 87.5 | 72.2 | 77.3 | **79.0** | 79.1 | 76.9 | **78.0** | -1.0 |
| Ranker: RoBERTa-RMSE | 81.2 | 71.8 | 77.3 | **76.8** | 79.1 | 79.0 | **79.0** | +2.2 |
| Ranker: RoBERTa-RankNet | 77.4 | 77.8 | 79.1 | **78.1** | 80.0 | 81.1 | **80.5** | +2.5 |
| Ranker: RoBERTa-ListNet | 78.8 | 70.3 | 78.3 | **75.8** | 73.9 | 76.9 | **75.4** | -0.4 |
| Ranker: XGBoost | 83.9 | 64.9 | 75.6 | **74.8** | 75.6 | 79.0 | **77.3** | +2.5 |

## ADE

| method | paper mistral | paper llama | paper gemma | **paper(avg)** | ours llama-3.1-8b-instruct | ours ministral-8b-2512 | **ours(avg)** | Δ(ours−paper) |
|---|---|---|---|---|---|---|---|---|
| no-subgraph (pair + context) | 53.9 | 66.7 | 79.4 | **66.6** | 94.4 | 95.9 | **95.2** | +28.5 |
| random | 69.0 | 83.1 | 86.6 | **79.5** | 93.5 | 82.5 | **88.0** | +8.4 |
| similarity (MiniLM) | 71.2 | 86.6 | 88.2 | **82.0** | 94.4 | 83.1 | **88.8** | +6.8 |
| GPT-based (RankGPT) | 73.3 | 88.2 | 89.9 | **83.8** | 95.9 | 84.8 | **90.4** | +6.6 |
| Ranker: RoBERTa-RMSE | 71.2 | 86.6 | 93.0 | **83.6** | 94.4 | 86.6 | **90.5** | +6.9 |
| Ranker: RoBERTa-RankNet | 77.4 | 89.9 | 89.9 | **85.7** | 95.9 | 79.4 | **87.6** | +1.9 |
| Ranker: RoBERTa-ListNet | 73.3 | 89.9 | 89.9 | **84.4** | 94.4 | 86.6 | **90.5** | +6.2 |
| Ranker: XGBoost | 73.3 | 93.0 | 84.8 | **83.7** | 94.4 | 79.4 | **86.9** | +3.2 |

## GENEC

| method | paper mistral | paper llama | paper gemma | **paper(avg)** | ours llama-3.1-8b-instruct | ours ministral-8b-2512 | **ours(avg)** | Δ(ours−paper) |
|---|---|---|---|---|---|---|---|---|
| no-subgraph (pair + context) | 20.7 | 53.3 | 27.3 | **33.8** | 53.7 | 40.0 | **46.8** | +13.1 |
| random | 28.6 | 41.2 | 51.4 | **40.4** | 54.8 | 34.5 | **44.7** | +4.3 |
| similarity (MiniLM) | 28.6 | 42.4 | 55.0 | **42.0** | 54.0 | 40.0 | **47.0** | +5.0 |
| GPT-based (RankGPT) | 26.1 | 55.6 | 52.6 | **44.8** | 47.4 | 46.7 | **47.0** | +2.3 |
| Ranker: RoBERTa-RMSE | 41.7 | 43.8 | 48.6 | **44.7** | 50.0 | 34.5 | **42.2** | -2.5 |
| Ranker: RoBERTa-RankNet | 41.7 | 57.1 | 59.5 | **52.8** | 54.0 | 41.4 | **47.7** | -5.0 |
| Ranker: RoBERTa-ListNet | 43.5 | 51.6 | 54.0 | **49.7** | 51.4 | 41.4 | **46.4** | -3.3 |
| Ranker: XGBoost | 34.8 | 38.7 | 52.9 | **42.1** | 55.6 | 35.7 | **45.6** | +3.5 |

## SEMEVAL

| method | paper mistral | paper llama | paper gemma | **paper(avg)** | ours llama-3.1-8b-instruct | ours ministral-8b-2512 | **ours(avg)** | Δ(ours−paper) |
|---|---|---|---|---|---|---|---|---|
| no-subgraph (pair + context) | 35.3 | 52.6 | 64.7 | **50.9** | 70.6 | 75.9 | **73.2** | +22.3 |
| random | 40.0 | 68.8 | 68.8 | **59.2** | 68.6 | 73.3 | **70.9** | +11.8 |
| similarity (MiniLM) | 52.6 | 72.7 | 68.8 | **64.7** | 70.6 | 73.3 | **72.0** | +7.3 |
| GPT-based (RankGPT) | 52.6 | 64.7 | 71.0 | **62.8** | 68.6 | 73.3 | **70.9** | +8.2 |
| Ranker: RoBERTa-RMSE | 57.1 | 75.0 | 71.0 | **67.7** | 68.6 | 73.3 | **70.9** | +3.2 |
| Ranker: RoBERTa-RankNet | 44.4 | 71.0 | 71.0 | **62.1** | 68.6 | 78.6 | **73.6** | +11.4 |
| Ranker: RoBERTa-ListNet | 42.1 | 71.0 | 71.0 | **61.4** | 68.6 | 73.3 | **70.9** | +9.6 |
| Ranker: XGBoost | 47.6 | 72.7 | 73.3 | **64.6** | 68.6 | 73.3 | **70.9** | +6.4 |

## Các kết luận chính của bài - có tái lập được không?

Chênh lệch F1 trung bình trên các ô (dataset × LLM); `thắng` = số ô có chênh lệch > 0 / tổng số ô. Các dòng 'best Subgraph Ranker' lấy max của 4 ranker nên lạc quan (thiên lệch chọn lọc); dòng 'C3b/C4b' dùng trung bình 4 ranker để đối chứng.

| claim | paper: Δ F1 | paper: thắng | ours: Δ F1 | ours: thắng |
|---|---|---|---|---|
| C1  random subgraph − no-subgraph | 12.51 | 11/12 | -2.00 | 2/8 |
| C2  best Subgraph Ranker − no-subgraph | 23.11 | 12/12 | 0.89 | 6/8 |
| C3  best Subgraph Ranker − random | 10.60 | 12/12 | 2.89 | 7/8 |
| C4  best Subgraph Ranker − similarity | 7.89 | 12/12 | 2.46 | 7/8 |
| C3b mean of 4 Subgraph Rankers − random (no max-over-4 bias) | 6.12 | 12/12 | 0.45 | 5/8 |
| C4b mean of 4 Subgraph Rankers − similarity (no max-over-4 bias) | 3.41 | 9/12 | 0.01 | 4/8 |
| C5  best Subgraph Ranker − GPT-based | 4.88 | 11/12 | 1.87 | 5/8 |
| C6  RankNet − RMSE (pairwise vs pointwise) | 1.49 | 6/12 | 1.69 | 6/8 |
| C7  RMSE − ListNet (pointwise vs listwise) | 0.39 | 5/12 | -0.14 | 2/8 |
| C8  mean(cross-encoders) − XGBoost | 2.26 | 8/12 | 1.10 | 6/8 |

## Module 2 - Subgraph Ranker (Table D2: NDCG@5 / Recall@5 của RoBERTa-base)

`ours (paper-style)` dùng đúng cách tính NDCG của tác giả (độ liên quan gán ngược, xem README); `ours (corrected)` là cách tính đã sửa.

| dataset | loss | paper NDCG@5 | ours paper-style | ours corrected (pool 5) | random corrected | paper Recall@5 | ours Recall@5 |
|---|---|---|---|---|---|---|---|
| COMAGC | rmse | 56.60 | 56.80 | 82.81 | 86.05 | 71.43 | 71.43 |
| COMAGC | ranknet | 57.12 | 54.00 | 87.22 | 86.05 | 71.43 | 71.43 |
| COMAGC | listnet | 55.38 | 53.90 | 87.38 | 86.05 | 71.43 | 71.43 |
| ADE | rmse | 57.61 | 56.10 | 86.16 | 86.50 | 72.18 | 72.18 |
| ADE | ranknet | 55.54 | 54.72 | 88.33 | 86.50 | 72.18 | 72.18 |
| ADE | listnet | 56.64 | 55.04 | 87.84 | 86.50 | 72.18 | 72.18 |
| GENEC | rmse | 56.79 | 55.41 | 85.51 | 86.07 | 71.77 | 71.77 |
| GENEC | ranknet | 55.89 | 55.76 | 85.08 | 86.07 | 71.77 | 71.77 |
| GENEC | listnet | 55.52 | 56.24 | 84.33 | 86.07 | 71.77 | 71.77 |
| SEMEVAL | rmse | 91.92 | 88.76 | 87.61 | 88.18 | 100.00 | 100.00 |
| SEMEVAL | ranknet | 89.42 | 85.92 | 90.46 | 88.18 | 100.00 | 100.00 |
| SEMEVAL | listnet | 88.99 | 89.16 | 87.21 | 88.18 | 100.00 | 100.00 |

## Chưa tái lập

* **Table 3 (SACHS vs PC / A* / DirectLiNGAM)**: cần dữ liệu quan sát SACHS + metapath Hetionet cho 11 protein; repo tác giả không có dữ liệu/mã cho phần này.
  Số của bài (không chạy lại): no-subgraph F1=27.27, RMSE=59.2, RankNet=48.0, ListNet=77.42, XGBoost=59.26; PC=44.0, A*=23.0, DirectLiNGAM=36.0.
* **Trích metapath từ Hetionet (Neo4j) / Wikidata (CLOCQ)** và **chấm điểm Module 1 bằng mistral-7b-instruct-v0.1 local**: mã không công bố; project dùng ứng viên đã trích của tác giả và có `m1_score_metapaths.py` để tự chấm lại bằng LLM khác.
