# Stage 1 config audit

Generated from committed config JSONs. Missing assets are UNAVAILABLE, never a measured hash.

| seed | run | role | teacher LS | student LS | T | α | LR | epochs | augmentation | teacher file | actual SHA-256 | schedule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | T_LS0 | teacher | 0 | 0.1 (미사용) | 1 (미사용) | 0.5 (미사용) | 5e-05 | 30 | 224 resize + flip | 미사용 | 미사용 | seed 0 |
| 0 | T_LS01 | teacher | 0.1 | 0.1 (미사용) | 1 (미사용) | 0.5 (미사용) | 5e-05 | 30 | 224 resize + flip | 미사용 | 미사용 | seed 0 |
| 0 | R1_ce | student | 0 (미사용) | 0.1 | 1 (미사용) | 0.5 (미사용) | 0.0005 | 100 | 224 resize + flip | 미사용 | 미사용 | seed 0 |
| 0 | R2_full_LS01best_T1 | student | 0.1 | 0.1 | 1 | 0.5 | 0.0005 | 100 | 224 resize + flip | outputs/stage1_ls_gate/seed_0/T_LS01/best.pt | 9a435130896203b0192341a0eab7743801c176005f21150fb58d5247e0d87345 | seed 0 |
| 0 | LS01_T4 | student | 0.1 | 0.1 | 4 | 0.5 | 0.0005 | 100 | 224 resize + flip | outputs/stage1_ls_gate/seed_0/T_LS01/last.pt | 734e0284b4ebcc3e862d17188dde26944a6534336866c6e1fb5d4155f8604484 | seed 0 |
| 0 | LS0_T1 | student | 0 | 0.1 | 1 | 0.5 | 0.0005 | 100 | 224 resize + flip | outputs/stage1_ls_gate/seed_0/T_LS0/last.pt | 6e31acca311a19b307a722008ef6c7f65259445ec04217f7d86f8486faf293d6 | seed 0 |
| 0 | LS0_T4 | student | 0 | 0.1 | 4 | 0.5 | 0.0005 | 100 | 224 resize + flip | outputs/stage1_ls_gate/seed_0/T_LS0/last.pt | 6e31acca311a19b307a722008ef6c7f65259445ec04217f7d86f8486faf293d6 | seed 0 |
| 1 | T_LS0 | teacher | 0 | 0.1 (미사용) | 1 (미사용) | 0.5 (미사용) | 5e-05 | 30 | 224 resize + flip | 미사용 | 미사용 | seed 0 통과 시; KD는 선택된 T만 |
| 1 | R1_ce | student | 0 (미사용) | 0.1 | 1 (미사용) | 0.5 (미사용) | 0.0005 | 100 | 224 resize + flip | 미사용 | 미사용 | seed 0 통과 시; KD는 선택된 T만 |
| 1 | LS0_T1 | student | 0 | 0.1 | 1 | 0.5 | 0.0005 | 100 | 224 resize + flip | outputs/stage1_ls_gate/seed_1/T_LS0/last.pt | UNAVAILABLE | seed 0 통과 시; KD는 선택된 T만 |
| 1 | LS0_T4 | student | 0 | 0.1 | 4 | 0.5 | 0.0005 | 100 | 224 resize + flip | outputs/stage1_ls_gate/seed_1/T_LS0/last.pt | UNAVAILABLE | seed 0 통과 시; KD는 선택된 T만 |

Required manifest SHA-256: `ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e`.
Data: `data/coco_single`, prepared with unchanged `8ebf7af:coco_kd/prepare.py`.
Teachers: epoch 30 `last.pt`; R2 only: newly trained seed 0 `T_LS01/best.pt`.
R2 retains the historical 58.91 ±2.0 pp reference by user instruction; it does not replay the original experiment-2 teacher.
All runs: AdamW wd 0.05, warmup 5 + cosine to 1e-6, batch 32 × accumulation 4, clip 1.0, AMP, drop_path 0.1.
Student: DeiT-Ti scratch. Teacher: DeiT-S ImageNet. Probe disabled. workers 0, threads 2.
Teacher measurement seed 0, train gate only; validation is descriptive. No test evaluation.
