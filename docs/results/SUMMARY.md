<!-- generated 2026-11-04T23:51:37+07:00 -->
# Module 2

| task | ranker | run_id | train_s | paper_NDCG@5 | paper_Recall@5 | corr_NDCG@5_top5 | Top1Hit_top5 | corr_NDCG@5_all | Top1Hit_all | rand_paper_NDCG@5 | rand_corr_NDCG@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ade | ce_listnet | 20261104-230710 | 69.0 | 0.55035 | 0.7218 | 0.87837 | 0.28947 | 0.73504 | 0.13158 | 0.55887 | 0.86496 |
| ade | ce_ranknet | 20261104-230557 | 72.8 | 0.54724 | 0.7218 | 0.88326 | 0.28947 | 0.74477 | 0.18421 | 0.55887 | 0.86496 |
| ade | ce_rmse | 20261104-230441 | 75.8 | 0.561 | 0.7218 | 0.8616 | 0.15789 | 0.73531 | 0.13158 | 0.55887 | 0.86496 |
| ade | xgboost | 20261104-230115 | 3.3 | 0.5642 | 0.7218 | 0.85657 | 0.15789 | 0.7331 | 0.15789 | 0.55887 | 0.86496 |
| comagc | ce_listnet | 20261104-230341 | 59.3 | 0.53903 | 0.71429 | 0.87376 | 0.26471 | 0.73996 | 0.20588 | 0.54749 | 0.86045 |
| comagc | ce_ranknet | 20261104-230237 | 63.1 | 0.54002 | 0.71429 | 0.87221 | 0.23529 | 0.74127 | 0.20588 | 0.54749 | 0.86045 |
| comagc | ce_rmse | 20261104-230123 | 73.8 | 0.56802 | 0.71429 | 0.82814 | 0.11765 | 0.72657 | 0.11765 | 0.54749 | 0.86045 |
| comagc | xgboost | 20261104-230058 | 16.5 | 0.52957 | 0.71429 | 0.88866 | 0.35294 | 0.74395 | 0.29412 | 0.54749 | 0.86045 |
| gene | ce_listnet | 20261104-230941 | 39.7 | 0.56243 | 0.71769 | 0.84328 | 0.17143 | 0.68848 | 0.11429 | 0.55078 | 0.86068 |
| gene | ce_ranknet | 20261104-230859 | 40.9 | 0.55757 | 0.71769 | 0.85084 | 0.2 | 0.69949 | 0.14286 | 0.55078 | 0.86068 |
| gene | ce_rmse | 20261104-230819 | 39.8 | 0.55407 | 0.71769 | 0.85509 | 0.2 | 0.72499 | 0.17143 | 0.55078 | 0.86068 |
| gene | xgboost | 20261104-230118 | 1.3 | 0.55006 | 0.71769 | 0.86196 | 0.28571 | 0.70056 | 0.17143 | 0.55078 | 0.86068 |
| semeval | ce_listnet | 20261104-231113 | 25.5 | 0.89158 | 1.0 | 0.87213 | 0.25 | 0.87213 | 0.25 | 0.88188 | 0.88184 |
| semeval | ce_ranknet | 20261104-231046 | 26.3 | 0.85916 | 1.0 | 0.90456 | 0.375 | 0.90456 | 0.375 | 0.88188 | 0.88184 |
| semeval | ce_rmse | 20261104-231021 | 25.3 | 0.88759 | 1.0 | 0.87613 | 0.33333 | 0.87613 | 0.33333 | 0.88188 | 0.88184 |
| semeval | xgboost | 20261104-230120 | 0.7 | 0.85812 | 1.0 | 0.90559 | 0.5 | 0.90559 | 0.5 | 0.88188 | 0.88184 |

# Module 3

| run | task | option | variant | eval_split | n | P | R | F1 | F1_ci95 | Acc | F1_authors_verbalizer | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | no_subgraph | 0 | test | 34 | 57.14 | 94.12 | 71.11 |  | 61.76 | 71.11 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | random:1 | 0 | test | 34 | 65.38 | 100.0 | 79.07 |  | 73.53 | 79.07 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | random:2 | 0 | test | 34 | 62.96 | 100.0 | 77.27 |  | 70.59 | 77.27 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | random:3 | 0 | test | 34 | 68.0 | 100.0 | 80.95 |  | 76.47 | 80.95 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | similarity_recompute | 0 | test | 34 | 60.71 | 100.0 | 75.56 |  | 67.65 | 75.56 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | ranker:xgboost | 0 | test | 34 | 60.71 | 100.0 | 75.56 |  | 67.65 | 75.56 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | ranker:ce_rmse | 0 | test | 34 | 65.38 | 100.0 | 79.07 |  | 73.53 | 79.07 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | ranker:ce_ranknet | 0 | test | 34 | 69.57 | 94.12 | 80.0 |  | 76.47 | 80.0 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | ranker:ce_listnet | 0 | test | 34 | 58.62 | 100.0 | 73.91 |  | 64.71 | 73.91 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | gpt_rerank | 0 | test | 34 | 65.38 | 100.0 | 79.07 |  | 73.53 | 79.07 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | no_subgraph | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | random:1 | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | random:2 | 0 | test | 38 | 100.0 | 86.84 | 92.96 |  | 86.84 | 92.96 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | random:3 | 0 | test | 38 | 100.0 | 86.84 | 92.96 |  | 86.84 | 92.96 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | similarity_recompute | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | ranker:xgboost | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | ranker:ce_rmse | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | ranker:ce_ranknet | 0 | test | 38 | 100.0 | 92.11 | 95.89 |  | 92.11 | 95.89 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | ranker:ce_listnet | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | gpt_rerank | 0 | test | 38 | 100.0 | 92.11 | 95.89 |  | 92.11 | 95.89 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | no_subgraph | 0 | test | 35 | 47.83 | 61.11 | 53.66 |  | 45.71 | 53.66 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | random:1 | 0 | test | 35 | 52.63 | 55.56 | 54.05 |  | 51.43 | 54.05 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | random:2 | 0 | test | 35 | 52.63 | 55.56 | 54.05 |  | 51.43 | 54.05 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | random:3 | 0 | test | 35 | 52.38 | 61.11 | 56.41 |  | 51.43 | 56.41 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | similarity_recompute | 0 | test | 35 | 52.63 | 55.56 | 54.05 |  | 51.43 | 54.05 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | ranker:xgboost | 0 | test | 35 | 55.56 | 55.56 | 55.56 |  | 54.29 | 55.56 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | ranker:ce_rmse | 0 | test | 35 | 50.0 | 50.0 | 50.0 |  | 48.57 | 50.0 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | ranker:ce_ranknet | 0 | test | 35 | 52.63 | 55.56 | 54.05 |  | 51.43 | 54.05 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | ranker:ce_listnet | 0 | test | 35 | 52.94 | 50.0 | 51.43 |  | 51.43 | 51.43 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | gpt_rerank | 0 | test | 35 | 45.0 | 50.0 | 47.37 |  | 42.86 | 47.37 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | no_subgraph | 0 | test | 24 | 54.55 | 100.0 | 70.59 |  | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | random:1 | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | random:2 | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | random:3 | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | similarity_recompute | 0 | test | 24 | 54.55 | 100.0 | 70.59 |  | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | ranker:xgboost | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | ranker:ce_rmse | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | ranker:ce_ranknet | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | ranker:ce_listnet | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | gpt_rerank | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | ade | random(mean of 3 seeds) | 0 | test | 38 | 100.0 | 87.72 | 93.45 |  | 87.72 | 93.45 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | comagc | random(mean of 3 seeds) | 0 | test | 34 | 65.45 | 100.0 | 79.1 |  | 73.53 | 79.1 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | gene | random(mean of 3 seeds) | 0 | test | 35 | 52.55 | 57.41 | 54.84 |  | 51.43 | 54.84 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1 | semeval | random(mean of 3 seeds) | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | comagc | no_subgraph | 0 | full | 672 | 24.91 | 97.12 | 39.65 | [35, 44] | 38.84 | 39.65 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | comagc | random:1 | 0 | full | 672 | 24.38 | 98.56 | 39.09 | [34, 44] | 36.46 | 39.09 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | comagc | random:2 | 0 | full | 672 | 24.36 | 96.4 | 38.9 | [34, 44] | 37.35 | 38.9 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | comagc | random:3 | 0 | full | 672 | 24.42 | 97.84 | 39.08 | [34, 44] | 36.9 | 39.08 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | comagc | similarity_recompute | 0 | full | 672 | 24.37 | 97.84 | 39.02 | [34, 44] | 36.76 | 39.02 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | comagc | ranker:xgboost | 0 | full | 672 | 25.18 | 100.0 | 40.23 | [35, 45] | 38.54 | 40.23 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | ade | no_subgraph | 0 | full | 838 | 100.0 | 94.27 | 97.05 | [96, 98] | 94.27 | 97.05 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | ade | random:1 | 0 | full | 838 | 100.0 | 90.69 | 95.12 | [94, 96] | 90.69 | 95.12 | 1 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | ade | random:2 | 0 | full | 838 | 100.0 | 89.98 | 94.72 | [94, 96] | 89.98 | 94.72 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | ade | random:3 | 0 | full | 838 | 100.0 | 90.93 | 95.25 | [94, 96] | 90.93 | 95.25 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | ade | similarity_recompute | 0 | full | 838 | 100.0 | 91.89 | 95.77 | [95, 97] | 91.89 | 95.77 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | ade | ranker:xgboost | 0 | full | 838 | 100.0 | 91.29 | 95.45 | [94, 96] | 91.29 | 95.45 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | gene | no_subgraph | 0 | full | 552 | 29.15 | 78.81 | 42.56 | [37, 48] | 54.53 | 42.56 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | gene | random:1 | 0 | full | 552 | 28.12 | 68.64 | 39.9 | [34, 46] | 55.8 | 39.9 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | gene | random:2 | 0 | full | 552 | 27.18 | 66.1 | 38.52 | [32, 44] | 54.89 | 38.52 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | gene | random:3 | 0 | full | 552 | 30.0 | 73.73 | 42.65 | [36, 48] | 57.61 | 42.65 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | gene | similarity_recompute | 0 | full | 552 | 28.72 | 72.03 | 41.06 | [35, 47] | 55.8 | 41.06 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | gene | ranker:xgboost | 0 | full | 552 | 28.28 | 69.49 | 40.2 | [34, 46] | 55.8 | 40.2 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | semeval | no_subgraph | 0 | full | 347 | 11.07 | 100.0 | 19.93 | [14, 26] | 30.55 | 19.93 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | semeval | random:1 | 0 | full | 347 | 10.25 | 96.67 | 18.53 | [13, 24] | 26.51 | 18.53 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | semeval | random:2 | 0 | full | 347 | 10.42 | 100.0 | 18.87 | [13, 25] | 25.65 | 18.87 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | semeval | random:3 | 0 | full | 347 | 10.34 | 100.0 | 18.75 | [13, 25] | 25.07 | 18.75 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | semeval | similarity_recompute | 0 | full | 347 | 10.34 | 100.0 | 18.75 | [13, 24] | 25.07 | 18.75 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | semeval | ranker:xgboost | 0 | full | 347 | 10.32 | 96.67 | 18.65 | [13, 24] | 27.09 | 18.65 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | ade | random(mean of 3 seeds) | 0 | full | 838 | 100.0 | 90.53 | 95.03 |  | 90.53 | 95.03 | 1 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | comagc | random(mean of 3 seeds) | 0 | full | 672 | 24.39 | 97.6 | 39.02 |  | 36.9 | 39.02 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | gene | random(mean of 3 seeds) | 0 | full | 552 | 28.43 | 69.49 | 40.36 |  | 56.1 | 40.36 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_full | semeval | random(mean of 3 seeds) | 0 | full | 347 | 10.34 | 98.89 | 18.72 |  | 25.74 | 18.72 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | no_subgraph | 0 | test | 34 | 57.14 | 94.12 | 71.11 | [54, 85] | 61.76 | 71.11 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | no_subgraph | 1 | test | 34 | 65.38 | 100.0 | 79.07 | [63, 91] | 73.53 | 79.07 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | no_subgraph | 2 | test | 34 | 65.38 | 100.0 | 79.07 | [63, 91] | 73.53 | 79.07 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | no_subgraph | 3 | test | 34 | 58.62 | 100.0 | 73.91 | [57, 86] | 64.71 | 73.91 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | no_subgraph | 4 | test | 34 | 53.12 | 100.0 | 69.39 | [52, 83] | 55.88 | 69.39 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | random:1 | 0 | test | 34 | 65.38 | 100.0 | 79.07 | [63, 91] | 73.53 | 79.07 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | random:1 | 1 | test | 34 | 68.0 | 100.0 | 80.95 | [65, 92] | 76.47 | 80.95 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | random:1 | 2 | test | 34 | 57.14 | 94.12 | 71.11 | [54, 85] | 61.76 | 71.11 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | random:1 | 3 | test | 34 | 58.62 | 100.0 | 73.91 | [58, 86] | 64.71 | 73.91 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | random:1 | 4 | test | 34 | 60.71 | 100.0 | 75.56 | [59, 88] | 67.65 | 75.56 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | similarity_recompute | 0 | test | 34 | 60.71 | 100.0 | 75.56 | [59, 88] | 67.65 | 75.56 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | similarity_recompute | 1 | test | 34 | 68.0 | 100.0 | 80.95 | [65, 92] | 76.47 | 80.95 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | similarity_recompute | 2 | test | 34 | 55.17 | 94.12 | 69.57 | [52, 84] | 58.82 | 69.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | similarity_recompute | 3 | test | 34 | 58.62 | 100.0 | 73.91 | [58, 86] | 64.71 | 73.91 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | similarity_recompute | 4 | test | 34 | 58.62 | 100.0 | 73.91 | [57, 87] | 64.71 | 73.91 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | ranker:xgboost | 0 | test | 34 | 60.71 | 100.0 | 75.56 | [59, 89] | 67.65 | 75.56 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | ranker:xgboost | 1 | test | 34 | 68.0 | 100.0 | 80.95 | [65, 92] | 76.47 | 80.95 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | ranker:xgboost | 2 | test | 34 | 61.54 | 94.12 | 74.42 | [59, 88] | 67.65 | 74.42 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | ranker:xgboost | 3 | test | 34 | 54.84 | 100.0 | 70.83 | [54, 85] | 58.82 | 70.83 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | ranker:xgboost | 4 | test | 34 | 60.71 | 100.0 | 75.56 | [59, 88] | 67.65 | 75.56 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | no_subgraph | 0 | test | 38 | 100.0 | 89.47 | 94.44 | [88, 99] | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | no_subgraph | 1 | test | 38 | 100.0 | 92.11 | 95.89 | [90, 100] | 92.11 | 95.89 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | no_subgraph | 2 | test | 38 | 100.0 | 94.74 | 97.3 | [93, 100] | 94.74 | 97.3 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | no_subgraph | 3 | test | 38 | 100.0 | 92.11 | 95.89 | [90, 100] | 92.11 | 95.89 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | no_subgraph | 4 | test | 38 | 100.0 | 97.37 | 98.67 | [96, 100] | 97.37 | 98.67 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | random:1 | 0 | test | 38 | 100.0 | 89.47 | 94.44 | [88, 99] | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | random:1 | 1 | test | 38 | 100.0 | 86.84 | 92.96 | [85, 99] | 86.84 | 92.96 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | random:1 | 2 | test | 38 | 100.0 | 81.58 | 89.86 | [81, 96] | 81.58 | 89.86 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | random:1 | 3 | test | 38 | 100.0 | 81.58 | 89.86 | [81, 96] | 81.58 | 89.86 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | random:1 | 4 | test | 38 | 100.0 | 92.11 | 95.89 | [90, 100] | 92.11 | 95.89 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | similarity_recompute | 0 | test | 38 | 100.0 | 89.47 | 94.44 | [87, 99] | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | similarity_recompute | 1 | test | 38 | 100.0 | 81.58 | 89.86 | [79, 96] | 81.58 | 89.86 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | similarity_recompute | 2 | test | 38 | 100.0 | 81.58 | 89.86 | [81, 96] | 81.58 | 89.86 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | similarity_recompute | 3 | test | 38 | 100.0 | 89.47 | 94.44 | [88, 99] | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | similarity_recompute | 4 | test | 38 | 100.0 | 92.11 | 95.89 | [90, 100] | 92.11 | 95.89 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | ranker:xgboost | 0 | test | 38 | 100.0 | 89.47 | 94.44 | [88, 99] | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | ranker:xgboost | 1 | test | 38 | 100.0 | 84.21 | 91.43 | [83, 97] | 84.21 | 91.43 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | ranker:xgboost | 2 | test | 38 | 100.0 | 86.84 | 92.96 | [85, 97] | 86.84 | 92.96 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | ranker:xgboost | 3 | test | 38 | 100.0 | 89.47 | 94.44 | [87, 99] | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | ranker:xgboost | 4 | test | 38 | 100.0 | 94.74 | 97.3 | [93, 100] | 94.74 | 97.3 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | no_subgraph | 0 | test | 35 | 47.83 | 61.11 | 53.66 | [32, 70] | 45.71 | 53.66 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | no_subgraph | 1 | test | 35 | 55.0 | 61.11 | 57.89 | [36, 74] | 54.29 | 57.89 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | no_subgraph | 2 | test | 35 | 61.11 | 61.11 | 61.11 | [39, 77] | 60.0 | 61.11 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | no_subgraph | 3 | test | 35 | 52.17 | 66.67 | 58.54 | [38, 75] | 51.43 | 58.54 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | no_subgraph | 4 | test | 35 | 58.82 | 55.56 | 57.14 | [34, 74] | 57.14 | 57.14 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | random:1 | 0 | test | 35 | 52.63 | 55.56 | 54.05 | [32, 71] | 51.43 | 54.05 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | random:1 | 1 | test | 35 | 53.85 | 38.89 | 45.16 | [21, 67] | 51.43 | 45.16 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | random:1 | 2 | test | 35 | 43.75 | 38.89 | 41.18 | [19, 61] | 42.86 | 41.18 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | random:1 | 3 | test | 35 | 42.11 | 44.44 | 43.24 | [23, 62] | 40.0 | 43.24 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | random:1 | 4 | test | 35 | 50.0 | 50.0 | 50.0 | [27, 68] | 48.57 | 50.0 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | similarity_recompute | 0 | test | 35 | 52.63 | 55.56 | 54.05 | [31, 71] | 51.43 | 54.05 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | similarity_recompute | 1 | test | 35 | 57.14 | 44.44 | 50.0 | [25, 70] | 54.29 | 50.0 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | similarity_recompute | 2 | test | 35 | 52.94 | 50.0 | 51.43 | [29, 70] | 51.43 | 51.43 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | similarity_recompute | 3 | test | 35 | 50.0 | 55.56 | 52.63 | [31, 70] | 48.57 | 52.63 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | similarity_recompute | 4 | test | 35 | 52.63 | 55.56 | 54.05 | [30, 71] | 51.43 | 54.05 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | ranker:xgboost | 0 | test | 35 | 55.56 | 55.56 | 55.56 | [33, 72] | 54.29 | 55.56 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | ranker:xgboost | 1 | test | 35 | 61.54 | 44.44 | 51.61 | [29, 71] | 57.14 | 51.61 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | ranker:xgboost | 2 | test | 35 | 42.86 | 33.33 | 37.5 | [14, 59] | 42.86 | 37.5 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | ranker:xgboost | 3 | test | 35 | 37.5 | 33.33 | 35.29 | [13, 56] | 37.14 | 35.29 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | ranker:xgboost | 4 | test | 35 | 56.25 | 50.0 | 52.94 | [31, 72] | 54.29 | 52.94 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | no_subgraph | 0 | test | 24 | 54.55 | 100.0 | 70.59 | [50, 86] | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | no_subgraph | 1 | test | 24 | 60.0 | 100.0 | 75.0 | [56, 89] | 66.67 | 75.0 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | no_subgraph | 2 | test | 24 | 57.14 | 100.0 | 72.73 | [52, 87] | 62.5 | 72.73 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | no_subgraph | 3 | test | 24 | 54.55 | 100.0 | 70.59 | [50, 86] | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | no_subgraph | 4 | test | 24 | 54.55 | 100.0 | 70.59 | [50, 86] | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | random:1 | 0 | test | 24 | 52.17 | 100.0 | 68.57 | [48, 83] | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | random:1 | 1 | test | 24 | 54.55 | 100.0 | 70.59 | [50, 85] | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | random:1 | 2 | test | 24 | 57.14 | 100.0 | 72.73 | [52, 87] | 62.5 | 72.73 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | random:1 | 3 | test | 24 | 57.14 | 100.0 | 72.73 | [52, 87] | 62.5 | 72.73 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | random:1 | 4 | test | 24 | 52.17 | 100.0 | 68.57 | [48, 83] | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | similarity_recompute | 0 | test | 24 | 54.55 | 100.0 | 70.59 | [50, 86] | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | similarity_recompute | 1 | test | 24 | 57.14 | 100.0 | 72.73 | [52, 87] | 62.5 | 72.73 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | similarity_recompute | 2 | test | 24 | 63.16 | 100.0 | 77.42 | [58, 91] | 70.83 | 77.42 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | similarity_recompute | 3 | test | 24 | 54.55 | 100.0 | 70.59 | [50, 86] | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | similarity_recompute | 4 | test | 24 | 52.17 | 100.0 | 68.57 | [48, 83] | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | ranker:xgboost | 0 | test | 24 | 52.17 | 100.0 | 68.57 | [48, 83] | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | ranker:xgboost | 1 | test | 24 | 54.55 | 100.0 | 70.59 | [50, 85] | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | ranker:xgboost | 2 | test | 24 | 57.14 | 100.0 | 72.73 | [52, 87] | 62.5 | 72.73 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | ranker:xgboost | 3 | test | 24 | 57.14 | 100.0 | 72.73 | [52, 87] | 62.5 | 72.73 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | ranker:xgboost | 4 | test | 24 | 52.17 | 100.0 | 68.57 | [48, 83] | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | no_subgraph [mean of 5 prompt variants] | mean | test | 38 | 100.0 | 93.16 | 96.44 |  | 93.16 | 96.44 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | ranker:xgboost [mean of 5 prompt variants] | mean | test | 38 | 100.0 | 88.95 | 94.11 |  | 88.95 | 94.11 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | ade | similarity_recompute [mean of 5 prompt variants] | mean | test | 38 | 100.0 | 86.84 | 92.9 |  | 86.84 | 92.9 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | no_subgraph [mean of 5 prompt variants] | mean | test | 34 | 59.93 | 98.82 | 74.51 |  | 65.88 | 74.51 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | ranker:xgboost [mean of 5 prompt variants] | mean | test | 34 | 61.16 | 98.82 | 75.46 |  | 67.65 | 75.46 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | comagc | similarity_recompute [mean of 5 prompt variants] | mean | test | 34 | 60.22 | 98.82 | 74.78 |  | 66.47 | 74.78 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | no_subgraph [mean of 5 prompt variants] | mean | test | 35 | 54.99 | 61.11 | 57.67 |  | 53.71 | 57.67 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | ranker:xgboost [mean of 5 prompt variants] | mean | test | 35 | 50.74 | 43.33 | 46.58 |  | 49.14 | 46.58 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | gene | similarity_recompute [mean of 5 prompt variants] | mean | test | 35 | 53.07 | 52.22 | 52.43 |  | 51.43 | 52.43 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | no_subgraph [mean of 5 prompt variants] | mean | test | 24 | 56.16 | 100.0 | 71.9 |  | 60.83 | 71.9 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | ranker:xgboost [mean of 5 prompt variants] | mean | test | 24 | 54.63 | 100.0 | 70.64 |  | 58.33 | 70.64 | 0 |
| or_llama-3.1-8b-instruct_chat_all_k1_pv | semeval | similarity_recompute [mean of 5 prompt variants] | mean | test | 24 | 56.31 | 100.0 | 71.98 |  | 60.83 | 71.98 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | no_subgraph | 0 | test | 34 | 57.14 | 94.12 | 71.11 |  | 61.76 | 71.11 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | no_subgraph_pair | 0 | test | 34 | 55.56 | 58.82 | 57.14 |  | 55.88 | 57.14 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | random:1 | 0 | test | 34 | 68.0 | 100.0 | 80.95 |  | 76.47 | 80.95 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | random:2 | 0 | test | 34 | 62.96 | 100.0 | 77.27 |  | 70.59 | 77.27 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | random:3 | 0 | test | 34 | 62.96 | 100.0 | 77.27 |  | 70.59 | 77.27 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | similarity | 0 | test | 34 | 62.96 | 100.0 | 77.27 |  | 70.59 | 77.27 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | gpt | 0 | test | 34 | 62.96 | 100.0 | 77.27 |  | 70.59 | 77.27 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | ranker:xgboost | 0 | test | 34 | 60.71 | 100.0 | 75.56 |  | 67.65 | 75.56 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | ranker:ce_rmse | 0 | test | 34 | 62.96 | 100.0 | 77.27 |  | 70.59 | 77.27 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | ranker:ce_ranknet | 0 | test | 34 | 64.0 | 94.12 | 76.19 |  | 70.59 | 76.19 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | ranker:ce_listnet | 0 | test | 34 | 56.67 | 100.0 | 72.34 |  | 61.76 | 72.34 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | no_subgraph | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | no_subgraph_pair | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | random:1 | 0 | test | 38 | 100.0 | 92.11 | 95.89 |  | 92.11 | 95.89 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | random:2 | 0 | test | 38 | 100.0 | 86.84 | 92.96 |  | 86.84 | 92.96 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | random:3 | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | similarity | 0 | test | 38 | 100.0 | 92.11 | 95.89 |  | 92.11 | 95.89 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | gpt | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | ranker:xgboost | 0 | test | 38 | 100.0 | 86.84 | 92.96 |  | 86.84 | 92.96 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | ranker:ce_rmse | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | ranker:ce_ranknet | 0 | test | 38 | 100.0 | 92.11 | 95.89 |  | 92.11 | 95.89 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | ranker:ce_listnet | 0 | test | 38 | 100.0 | 89.47 | 94.44 |  | 89.47 | 94.44 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | no_subgraph | 0 | test | 35 | 47.83 | 61.11 | 53.66 |  | 45.71 | 53.66 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | no_subgraph_pair | 0 | test | 35 | 62.5 | 27.78 | 38.46 |  | 54.29 | 38.46 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | random:1 | 0 | test | 35 | 52.38 | 61.11 | 56.41 |  | 51.43 | 56.41 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | random:2 | 0 | test | 35 | 50.0 | 55.56 | 52.63 |  | 48.57 | 52.63 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | random:3 | 0 | test | 35 | 52.38 | 61.11 | 56.41 |  | 51.43 | 56.41 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | similarity | 0 | test | 35 | 43.75 | 38.89 | 41.18 |  | 42.86 | 41.18 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | gpt | 0 | test | 35 | 57.14 | 66.67 | 61.54 |  | 57.14 | 61.54 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | ranker:xgboost | 0 | test | 35 | 52.63 | 55.56 | 54.05 |  | 51.43 | 54.05 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | ranker:ce_rmse | 0 | test | 35 | 55.56 | 55.56 | 55.56 |  | 54.29 | 55.56 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | ranker:ce_ranknet | 0 | test | 35 | 52.38 | 61.11 | 56.41 |  | 51.43 | 56.41 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | ranker:ce_listnet | 0 | test | 35 | 55.0 | 61.11 | 57.89 |  | 54.29 | 57.89 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | no_subgraph | 0 | test | 24 | 54.55 | 100.0 | 70.59 |  | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | no_subgraph_pair | 0 | test | 24 | 55.0 | 91.67 | 68.75 |  | 58.33 | 68.75 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | random:1 | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | random:2 | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | random:3 | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | similarity | 0 | test | 24 | 54.55 | 100.0 | 70.59 |  | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | gpt | 0 | test | 24 | 54.55 | 100.0 | 70.59 |  | 58.33 | 70.59 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | ranker:xgboost | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | ranker:ce_rmse | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | ranker:ce_ranknet | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | ranker:ce_listnet | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | ade | random(mean of 3 seeds) | 0 | test | 38 | 100.0 | 89.47 | 94.43 |  | 89.47 | 94.43 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | comagc | random(mean of 3 seeds) | 0 | test | 34 | 64.64 | 100.0 | 78.5 |  | 72.55 | 78.5 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | gene | random(mean of 3 seeds) | 0 | test | 35 | 51.59 | 59.26 | 55.15 |  | 50.48 | 55.15 | 0 |
| or_llama-3.1-8b-instruct_chat_oracle_top5_k1 | semeval | random(mean of 3 seeds) | 0 | test | 24 | 52.17 | 100.0 | 68.57 |  | 54.17 | 68.57 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | no_subgraph | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | random:1 | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | random:2 | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | random:3 | 0 | test | 34 | 68.18 | 88.24 | 76.92 |  | 73.53 | 76.92 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | similarity_recompute | 0 | test | 34 | 68.18 | 88.24 | 76.92 |  | 73.53 | 76.92 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | ranker:xgboost | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | ranker:ce_rmse | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | ranker:ce_ranknet | 0 | test | 34 | 75.0 | 88.24 | 81.08 |  | 79.41 | 81.08 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | ranker:ce_listnet | 0 | test | 34 | 68.18 | 88.24 | 76.92 |  | 73.53 | 76.92 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | gpt_rerank | 0 | test | 34 | 68.18 | 88.24 | 76.92 |  | 73.53 | 76.92 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | no_subgraph | 0 | test | 38 | 100.0 | 92.11 | 95.89 |  | 92.11 | 95.89 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | random:1 | 0 | test | 38 | 100.0 | 68.42 | 81.25 |  | 68.42 | 81.25 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | random:2 | 0 | test | 38 | 100.0 | 68.42 | 81.25 |  | 68.42 | 81.25 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | random:3 | 0 | test | 38 | 100.0 | 73.68 | 84.85 |  | 73.68 | 84.85 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | similarity_recompute | 0 | test | 38 | 100.0 | 71.05 | 83.08 |  | 71.05 | 83.08 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | ranker:xgboost | 0 | test | 38 | 100.0 | 65.79 | 79.37 |  | 65.79 | 79.37 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | ranker:ce_rmse | 0 | test | 38 | 100.0 | 76.32 | 86.57 |  | 76.32 | 86.57 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | ranker:ce_ranknet | 0 | test | 38 | 100.0 | 65.79 | 79.37 |  | 65.79 | 79.37 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | ranker:ce_listnet | 0 | test | 38 | 100.0 | 76.32 | 86.57 |  | 76.32 | 86.57 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | gpt_rerank | 0 | test | 38 | 100.0 | 73.68 | 84.85 |  | 73.68 | 84.85 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | no_subgraph | 0 | test | 35 | 50.0 | 33.33 | 40.0 |  | 48.57 | 40.0 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | random:1 | 0 | test | 35 | 45.45 | 27.78 | 34.48 |  | 45.71 | 34.48 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | random:2 | 0 | test | 35 | 45.45 | 27.78 | 34.48 |  | 45.71 | 34.48 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | random:3 | 0 | test | 35 | 45.45 | 27.78 | 34.48 |  | 45.71 | 34.48 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | similarity_recompute | 0 | test | 35 | 50.0 | 33.33 | 40.0 |  | 48.57 | 40.0 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | ranker:xgboost | 0 | test | 35 | 50.0 | 27.78 | 35.71 |  | 48.57 | 35.71 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | ranker:ce_rmse | 0 | test | 35 | 45.45 | 27.78 | 34.48 |  | 45.71 | 34.48 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | ranker:ce_ranknet | 0 | test | 35 | 54.55 | 33.33 | 41.38 |  | 51.43 | 41.38 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | ranker:ce_listnet | 0 | test | 35 | 54.55 | 33.33 | 41.38 |  | 51.43 | 41.38 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | gpt_rerank | 0 | test | 35 | 58.33 | 38.89 | 46.67 |  | 54.29 | 46.67 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | no_subgraph | 0 | test | 24 | 64.71 | 91.67 | 75.86 |  | 70.83 | 75.86 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | random:1 | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | random:2 | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | random:3 | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | similarity_recompute | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | ranker:xgboost | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | ranker:ce_rmse | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | ranker:ce_ranknet | 0 | test | 24 | 68.75 | 91.67 | 78.57 |  | 75.0 | 78.57 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | ranker:ce_listnet | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | gpt_rerank | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_all_k1 | ade | random(mean of 3 seeds) | 0 | test | 38 | 100.0 | 70.17 | 82.45 |  | 70.17 | 82.45 | 0 |
| or_ministral-8b-2512_chat_all_k1 | comagc | random(mean of 3 seeds) | 0 | test | 34 | 70.35 | 88.24 | 78.27 |  | 75.49 | 78.27 | 0 |
| or_ministral-8b-2512_chat_all_k1 | gene | random(mean of 3 seeds) | 0 | test | 35 | 45.45 | 27.78 | 34.48 |  | 45.71 | 34.48 | 0 |
| or_ministral-8b-2512_chat_all_k1 | semeval | random(mean of 3 seeds) | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | comagc | no_subgraph | 0 | full | 672 | 39.06 | 89.93 | 54.47 | [49, 60] | 68.9 | 54.47 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | comagc | random:1 | 0 | full | 672 | 37.22 | 84.89 | 51.75 | [46, 57] | 67.26 | 51.75 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | comagc | random:2 | 0 | full | 672 | 37.42 | 87.77 | 52.47 | [47, 58] | 67.11 | 52.47 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | comagc | random:3 | 0 | full | 672 | 37.27 | 86.33 | 52.06 | [46, 58] | 67.11 | 52.06 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | comagc | similarity_recompute | 0 | full | 672 | 35.61 | 86.33 | 50.42 | [45, 56] | 64.88 | 50.42 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | comagc | ranker:xgboost | 0 | full | 672 | 36.65 | 84.89 | 51.19 | [45, 57] | 66.52 | 51.19 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | ade | no_subgraph | 0 | full | 838 | 100.0 | 96.9 | 98.42 | [98, 99] | 96.9 | 98.42 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | ade | random:1 | 0 | full | 838 | 100.0 | 79.59 | 88.64 | [87, 90] | 79.59 | 88.64 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | ade | random:2 | 0 | full | 838 | 100.0 | 80.31 | 89.08 | [88, 91] | 80.31 | 89.08 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | ade | random:3 | 0 | full | 838 | 100.0 | 79.59 | 88.64 | [87, 90] | 79.59 | 88.64 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | ade | similarity_recompute | 0 | full | 838 | 100.0 | 81.03 | 89.52 | [88, 91] | 81.03 | 89.52 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | ade | ranker:xgboost | 0 | full | 838 | 100.0 | 80.43 | 89.15 | [88, 91] | 80.43 | 89.15 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | gene | no_subgraph | 0 | full | 552 | 29.59 | 49.15 | 36.94 | [30, 44] | 64.13 | 36.94 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | gene | random:1 | 0 | full | 552 | 30.25 | 41.53 | 35.0 | [28, 42] | 67.03 | 35.0 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | gene | random:2 | 0 | full | 552 | 30.06 | 41.53 | 34.88 | [28, 42] | 66.85 | 34.88 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | gene | random:3 | 0 | full | 552 | 31.68 | 43.22 | 36.56 | [30, 44] | 67.93 | 36.56 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | gene | similarity_recompute | 0 | full | 552 | 28.99 | 41.53 | 34.15 | [28, 41] | 65.76 | 34.15 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | gene | ranker:xgboost | 0 | full | 552 | 27.06 | 38.98 | 31.94 | [25, 39] | 64.49 | 31.94 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | semeval | no_subgraph | 0 | full | 347 | 26.04 | 83.33 | 39.68 | [28, 50] | 78.1 | 39.68 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | semeval | random:1 | 0 | full | 347 | 30.59 | 86.67 | 45.22 | [33, 55] | 81.84 | 45.22 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | semeval | random:2 | 0 | full | 347 | 30.34 | 90.0 | 45.38 | [33, 56] | 81.27 | 45.38 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | semeval | random:3 | 0 | full | 347 | 29.55 | 86.67 | 44.07 | [32, 54] | 80.98 | 44.07 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | semeval | similarity_recompute | 0 | full | 347 | 27.47 | 83.33 | 41.32 | [29, 52] | 79.54 | 41.32 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | semeval | ranker:xgboost | 0 | full | 347 | 30.23 | 86.67 | 44.83 | [32, 55] | 81.56 | 44.83 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | ade | random(mean of 3 seeds) | 0 | full | 838 | 100.0 | 79.83 | 88.79 |  | 79.83 | 88.79 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | comagc | random(mean of 3 seeds) | 0 | full | 672 | 37.3 | 86.33 | 52.09 |  | 67.16 | 52.09 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | gene | random(mean of 3 seeds) | 0 | full | 552 | 30.66 | 42.09 | 35.48 |  | 67.27 | 35.48 | 0 |
| or_ministral-8b-2512_chat_all_k1_full | semeval | random(mean of 3 seeds) | 0 | full | 347 | 30.16 | 87.78 | 44.89 |  | 81.36 | 44.89 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | no_subgraph | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | no_subgraph_pair | 0 | test | 34 | 53.85 | 82.35 | 65.12 |  | 55.88 | 65.12 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | random:1 | 0 | test | 34 | 70.0 | 82.35 | 75.68 |  | 73.53 | 75.68 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | random:2 | 0 | test | 34 | 75.0 | 88.24 | 81.08 |  | 79.41 | 81.08 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | random:3 | 0 | test | 34 | 68.18 | 88.24 | 76.92 |  | 73.53 | 76.92 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | similarity | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | gpt | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | ranker:xgboost | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | ranker:ce_rmse | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | ranker:ce_ranknet | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | ranker:ce_listnet | 0 | test | 34 | 71.43 | 88.24 | 78.95 |  | 76.47 | 78.95 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | no_subgraph | 0 | test | 38 | 100.0 | 92.11 | 95.89 |  | 92.11 | 95.89 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | no_subgraph_pair | 0 | test | 38 | 100.0 | 92.11 | 95.89 |  | 92.11 | 95.89 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | random:1 | 0 | test | 38 | 100.0 | 65.79 | 79.37 |  | 65.79 | 79.37 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | random:2 | 0 | test | 38 | 100.0 | 68.42 | 81.25 |  | 68.42 | 81.25 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | random:3 | 0 | test | 38 | 100.0 | 71.05 | 83.08 |  | 71.05 | 83.08 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | similarity | 0 | test | 38 | 100.0 | 63.16 | 77.42 |  | 63.16 | 77.42 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | gpt | 0 | test | 38 | 100.0 | 68.42 | 81.25 |  | 68.42 | 81.25 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | ranker:xgboost | 0 | test | 38 | 100.0 | 68.42 | 81.25 |  | 68.42 | 81.25 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | ranker:ce_rmse | 0 | test | 38 | 100.0 | 73.68 | 84.85 |  | 73.68 | 84.85 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | ranker:ce_ranknet | 0 | test | 38 | 100.0 | 65.79 | 79.37 |  | 65.79 | 79.37 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | ranker:ce_listnet | 0 | test | 38 | 100.0 | 73.68 | 84.85 |  | 73.68 | 84.85 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | no_subgraph | 0 | test | 35 | 50.0 | 33.33 | 40.0 |  | 48.57 | 40.0 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | no_subgraph_pair | 0 | test | 35 | 58.33 | 77.78 | 66.67 |  | 60.0 | 66.67 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | random:1 | 0 | test | 35 | 45.45 | 27.78 | 34.48 |  | 45.71 | 34.48 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | random:2 | 0 | test | 35 | 45.45 | 27.78 | 34.48 |  | 45.71 | 34.48 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | random:3 | 0 | test | 35 | 50.0 | 27.78 | 35.71 |  | 48.57 | 35.71 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | similarity | 0 | test | 35 | 53.85 | 38.89 | 45.16 |  | 51.43 | 45.16 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | gpt | 0 | test | 35 | 54.55 | 33.33 | 41.38 |  | 51.43 | 41.38 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | ranker:xgboost | 0 | test | 35 | 50.0 | 27.78 | 35.71 |  | 48.57 | 35.71 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | ranker:ce_rmse | 0 | test | 35 | 45.45 | 27.78 | 34.48 |  | 45.71 | 34.48 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | ranker:ce_ranknet | 0 | test | 35 | 54.55 | 33.33 | 41.38 |  | 51.43 | 41.38 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | ranker:ce_listnet | 0 | test | 35 | 54.55 | 33.33 | 41.38 |  | 51.43 | 41.38 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | no_subgraph | 0 | test | 24 | 64.71 | 91.67 | 75.86 |  | 70.83 | 75.86 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | no_subgraph_pair | 0 | test | 24 | 56.25 | 75.0 | 64.29 |  | 58.33 | 64.29 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | random:1 | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | random:2 | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | random:3 | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | similarity | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | gpt | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | ranker:xgboost | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | ranker:ce_rmse | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | ranker:ce_ranknet | 0 | test | 24 | 68.75 | 91.67 | 78.57 |  | 75.0 | 78.57 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | ranker:ce_listnet | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | ade | random(mean of 3 seeds) | 0 | test | 38 | 100.0 | 68.42 | 81.23 |  | 68.42 | 81.23 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | comagc | random(mean of 3 seeds) | 0 | test | 34 | 71.06 | 86.28 | 77.89 |  | 75.49 | 77.89 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | gene | random(mean of 3 seeds) | 0 | test | 35 | 46.97 | 27.78 | 34.89 |  | 46.66 | 34.89 | 0 |
| or_ministral-8b-2512_chat_oracle_top5_k1 | semeval | random(mean of 3 seeds) | 0 | test | 24 | 61.11 | 91.67 | 73.33 |  | 66.67 | 73.33 | 0 |
