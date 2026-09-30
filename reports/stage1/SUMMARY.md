# Stage 1 — KD gate

Decision: **WAIT_SEED0**

Only validation is used for student decisions. NOT_RUN means no measurement; it is not a failure or a pass.

R2 uses newly trained seed 0 T_LS01 best.pt. The historical 58.91 ±2.0 pp reference is retained by user instruction; the original experiment-2 teacher is not replayed.

## Table 1: teacher outputs

Train uses augmentation once with measurement seed 0; KL gates apply only to train. Entropy is normalized by log(9).

| seed | teacher | split | accuracy % | macro % | KL to LS (nats) | nontarget entropy T=1 | T=4 | train gate |
|---|---|---|---|---|---|---|---|---|
| 0 | T_LS0 | train | — | — | — | — | — | NOT_RUN |
| 0 | T_LS0 | val | — | — | — | — | — | NOT_RUN |
| 0 | T_LS01 | train | — | — | — | — | — | NOT_RUN |
| 0 | T_LS01 | val | — | — | — | — | — | NOT_RUN |
| 0 | T_LS01_best | train | — | — | — | — | — | NOT_RUN |
| 0 | T_LS01_best | val | — | — | — | — | — | NOT_RUN |
| 1 | T_LS0 | train | — | — | — | — | — | NOT_RUN |
| 1 | T_LS0 | val | — | — | — | — | — | NOT_RUN |

## Table 2: students

Means use epochs 91–100; differences use unrounded values.

| seed | run | val macro mean % | Δ CE (pp) | Δ regression reference (pp) | initial checksum |
|---|---|---|---|---|---|
| 0 | R1_ce | — | — | — | NOT_RUN |
| 0 | R2_full_LS01best_T1 | — | — | — | NOT_RUN |
| 0 | LS01_T4 | — | — | — | NOT_RUN |
| 0 | LS0_T1 | — | — | — | NOT_RUN |
| 0 | LS0_T4 | — | — | — | NOT_RUN |
| 1 | R1_ce | — | — | — | NOT_RUN |

## Table 3: decisions

| rule | status |
|---|---|
| local/server test suite (current source) | PASS |
| R1_ce: regression ±2.0 pp | NOT_RUN |
| R1_ce: initial checksum | NOT_RUN |
| R2_full_LS01best_T1: regression ±2.0 pp | NOT_RUN |
| R2_full_LS01best_T1: initial checksum | NOT_RUN |
| seed 0 T_LS0: train KL gate | NOT_RUN |
| seed 0 T_LS01: train KL gate | NOT_RUN |
| seed 1 T_LS0: train KL gate | NOT_RUN |
| manifest_sha256 | NOT_RUN |

Selected seed 0 candidate: `NOT_SELECTED`.

Validation curves: NOT_RUN (GPU experiment has not run).

해석: 위 결정은 사전 고정된 임계값만 적용한 결과이며, 미실행 항목의 성능은 추정하지 않는다.
