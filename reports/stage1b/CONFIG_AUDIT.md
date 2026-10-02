# Stage 1b config audit

| run | augmentation | T | alpha | LS | LR | epochs | teacher SHA-256 |
|---|---|---|---|---|---|---|---|
| T_LS0 | flip | 1 (미사용) | 0.5 (미사용) | 0.0 | 5e-05 | 30 | 미사용 |
| ce_flip | flip | 1 (미사용) | 0.5 (미사용) | 0.1 | 0.0005 | 100 | 미사용 |
| LS0_T4_flip | flip | 4.0 | 0.5 | 0.1 | 0.0005 | 100 | UNAVAILABLE |
| ce_rrc | rrc | 1 (미사용) | 0.5 (미사용) | 0.1 | 0.0005 | 100 | 미사용 |
| LS0_T1_rrc | rrc | 1.0 | 0.5 | 0.1 | 0.0005 | 100 | UNAVAILABLE |
| LS0_T4_rrc | rrc | 4.0 | 0.5 | 0.1 | 0.0005 | 100 | UNAVAILABLE |
| ce_rrc_mix | rrc_mix | 1 (미사용) | 0.5 (미사용) | 0.1 | 0.0005 | 100 | 미사용 |
| LS0_T4_rrc_mix | rrc_mix | 4.0 | 0.5 | 0.1 | 0.0005 | 100 | UNAVAILABLE |

Teacher: `outputs/stage1b_aug/seed_0/T_LS0/last.pt`; actual SHA-256 recorded and checked against its passing output-check record (no historical teacher hash).
Manifest required SHA-256: `ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e`.
Teacher: seed 0, ImageNet DeiT-S, LS=0, flip, 30 epochs, last.pt; train kl_to_ls >=0.02 required.
seed 0; scratch DeiT-Ti; batch 32 × accumulation 4; workers 0; threads 2; AMP; drop_path 0.1.
AdamW wd 0.05; warmup 5; cosine min 1e-6; grad clip 1.0; Probe off; test forbidden.
RRC: 224, scale (0.08, 1), ratio (3/4, 4/3), bicubic antialias=True; paired nearest mask; horizontal flip p=0.5.
Mix: batch mode, probability 1, mixup alpha 0.8 / cutmix alpha 1.0 selected 50/50, reverse-batch pairs, corrected area lambda.
CE soft targets contain LS=0.1 once; teacher sees the identical mixed image; mixed train accuracy is null.
Each KD PASS iff delta against same-augmentation CE >=1.5 pp; all seven runs finish; no seed 1.
