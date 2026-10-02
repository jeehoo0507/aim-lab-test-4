# Stage 1b — augmentation gate

Execution: **WAIT_RUNS**

Validation macro accuracy, mean of epochs 91–100. Test is never evaluated.
Stage 1 reference: `c5a826d`; existing reports are pinned and read only.

| Stage | run | augmentation | val mean % | delta vs same-augmentation CE (pp) | gate |
|---|---|---|---|---|---|
| 1 | R1_ce | flip | 60.7800 | +0.0000 | reference |
| 1 | R2_full_LS01best_T1 | flip | 58.9100 | -1.8700 | reference |
| 1 | LS01_T4 | flip | 57.8700 | -2.9100 | reference |
| 1 | LS0_T1 | flip | 59.9200 | -0.8600 | reference |
| 1 | LS0_T4 | flip | 60.2100 | -0.5700 | reference |
| 1b | ce_flip | flip | NOT_RUN | baseline | baseline |
| 1b | LS0_T4_flip | flip | NOT_RUN | NOT_RUN | NOT_RUN |
| 1b | ce_rrc | rrc | NOT_RUN | baseline | baseline |
| 1b | LS0_T1_rrc | rrc | NOT_RUN | NOT_RUN | NOT_RUN |
| 1b | LS0_T4_rrc | rrc | NOT_RUN | NOT_RUN | NOT_RUN |
| 1b | ce_rrc_mix | rrc_mix | NOT_RUN | baseline | baseline |
| 1b | LS0_T4_rrc_mix | rrc_mix | NOT_RUN | NOT_RUN | NOT_RUN |

KD PASS requires delta >=1.5 pp against its own augmentation CE. No automatic seed 1 or further training.
Train accuracy for rrc_mix is not recorded; the CE target is the LS=0.1 mixture.
Crop/mix stream hashes must match between each RRC / RRC-mix KD run and its CE for every epoch.

Server historical flip gate: **NOT_RUN**.
CPU parity is a separate same-platform test on synthetic data. Its checksum and metrics are not server reference values.

New T_LS0 output check: **NOT_RUN**.
Teacher actual SHA-256: `NOT_RUN`.

| split | accuracy | macro accuracy | kl_to_ls (nats) | non-target entropy T1 | non-target entropy T4 |
|---|---|---|---|---|---|

Teacher gate uses train kl_to_ls >=0.02 only; validation is report-only.

New-server flip diagnostics: initialization checksum is strict; historical epoch metrics are report-only.
| epoch | metric (raw units) | Stage 1 reference | new server | difference | within 1e-6 |
|---|---|---|---|---|---|

| flip reproduction | Stage 1 val mean % | new server val mean % | difference pp (report-only) |
|---|---|---|---|
| ce_flip | 60.7800 | NOT_RUN | NOT_RUN |
| LS0_T4_flip | 60.2100 | NOT_RUN | NOT_RUN |

Validation curves: NOT_RUN. No Stage 1b GPU results yet.

해석: 각 KD 결과는 같은 증강 CE와만 비교한다. 미실행 결과는 추정하지 않는다.
