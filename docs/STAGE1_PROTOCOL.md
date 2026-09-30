# AIM Lab 실험 3 · Stage 1 — KD 게이트 확인 (COCO만)

> **이 문서가 이번 작업의 유일한 기준이다.**
>
> 1. 이 문서를 수정하지 말고 `docs/STAGE1_PROTOCOL.md`로 커밋한다.
> 2. 코드를 쓰기 전에 구현 계획(15줄 이내)과 모호한 점을 `QUESTIONS.md`에 적어 제출하고, 사용자 답을 받은 뒤 시작한다.
> 3. 여기 없는 설정은 추측으로 정하지 않는다. 결과를 보고 설정을 바꾸지 않는다. 바꿔야 할 것 같으면 멈추고 묻는다.

## 목적

실험 1·2에서는 teacher도 student용 `label_smoothing=0.1`로 학습됐다 (`coco_kd/train.py` L49의 `distillation_loss`를 teacher도 사용). teacher가 학습셋에서 LS 손실 하한까지 수렴해서(train loss 0.5019, 하한 0.5003), teacher 출력이 LS 라벨과 거의 같아졌고 그 결과 Full KD ≈ CE가 됐다.

이번 작업은 **teacher의 LS를 제거하면 COCO-10 scratch에서 KD가 CE보다 확실히 좋아지는지**만 확인한다. 마스킹 방법 비교는 하지 않는다.

## 작업 위치

- `jeehoo0507/aim-lab-test-2`의 main `8ebf7af`에서 새 브랜치 `stage1-ls-gate`를 만들어 작업한다.
- 기존 코드·데이터 로더·학습 루프를 그대로 쓰고, 아래 변경만 한다.
- 출력은 새 경로 `outputs/stage1_ls_gate/`에 쓴다. 기존 `outputs/experiment2/`는 읽기만 하고 수정하지 않는다.

## 코드 변경 (이것만)

1. **teacher 전용 label smoothing**&#x20;
   - config에 `teacher_label_smoothing`을 추가한다. 기본값 없이 JSON에 반드시 명시하게 하고, 없으면 에러를 낸다.
   - teacher 학습 경로(role == "teacher")는 이 값으로 CE를 계산한다. student 경로는 기존 `label_smoothing`(0.1)을 그대로 쓴다.
   - teacher 학습이 student KD 손실 함수에 이 값을 몰래 넘기는 구조가 아니라, teacher 전용 CE 호출이 되도록 명시적으로 분리한다.
2. **teacher 체크포인트 경로 지정:** student run이 쓸 teacher 파일을 config로 지정할 수 있게 한다(`teacher_checkpoint`). run 시작 시 SHA-256을 기록한다.
3. **temperature:** 기존 `temperature` 필드를 쓴다. 이번 run들의 config JSON에는 반드시 명시한다.
4. **teacher 출력 점검 스크립트** `scripts/check_teacher_outputs.py` (아래 정의).
5. **config 감사:** `scripts/plan_stage1.py`가 이번 run 전체의 역할별 표(teacher LS, student LS, T, α, LR, 에포치, 증강, teacher 파일 SHA-256)를 `reports/stage1/CONFIG_AUDIT.md`로 만든다.

## 고정 설정 (실험 2와 동일, 바꾸지 않는다)

- 데이터: 기존 COCO-10 준비 데이터. `data/coco_single/manifest.json`의 SHA-256이 `ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e`가 아니면 에러.
- 증강: 전체 이미지 224 resize + flip.
- student: DeiT-Ti scratch, 100 에포치, peak LR 5e-4, student LS 0.1, α 0.5.
- teacher: DeiT-S ImageNet 초기화, 30 에포치, LR 5e-5, drop_path 0.1.
- 공통: AdamW(wd 0.05), warmup 5 + cosine(최소 1e-6), micro-batch 32 × accumulation 4, grad clip 1.0, AMP.
- seed 0.

## 학습할 teacher (seed 0)

| teacher  | teacher LS | 나머지                  | 사용 체크포인트      |
| -------- | ---------- | -------------------- | ------------- |
| `T_LS0`  | 0.0        | 실험 2 teacher 레시피와 동일 | last (30 에포치) |
| `T_LS01` | 0.1        | 실험 2 teacher 레시피와 동일 | last (30 에포치) |

두 teacher는 LS 값만 다르다.

## student run (seed 0, 5개)

| run              | teacher                                                                                                                                | T | 목적                           |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------- | - | ---------------------------- |
| `R1_ce`          | 없음                                                                                                                                     | – | 회귀: 실험 2 CE 재현               |
| `R2_full_old_T1` | 실험 2 teacher `outputs/experiment2/seed_0/teacher/best.pt` (SHA-256 `4a18ddc11c98c662fd77ff9f8b5fc249e00d2c0052263e0e8d76f91088556646`) | 1 | 회귀: 실험 2 Full KD 재현          |
| `LS01_T4`        | `T_LS01`                                                                                                                               | 4 | 대조: LS teacher는 T를 올려도 안 되는지 |
| `LS0_T1`         | `T_LS0`                                                                                                                                | 1 | 후보                           |
| `LS0_T4`         | `T_LS0`                                                                                                                                | 4 | 후보                           |

- 모두 Full KD(teacher 196 patch 전부)다. 마스킹은 쓰지 않는다.
- 4\~5개 병렬로 돌려도 된다(기존 `--jobs` 방식).

## teacher 출력 점검 (teacher 학습 직후, student run 전)

`scripts/check_teacher_outputs.py`로 각 teacher를 학습셋 1회(학습 증강, 고정 seed)와 val에 대해 돌려서 다음을 잰다.

- teacher 정확도 (train, val)
- `kl_to_ls` = 샘플 평균 KL(p_T(T=1) ‖ 0.9·onehot + 0.01), 단위 nats
- `nontarget_entropy@T` (T ∈ {1, 4}): 정답 클래스를 빼고 재정규화한 분포의 엔트로피 / log 9

판정:

- `T_LS0`: `kl_to_ls` ≥ 0.02 이어야 한다. 아니면 멈춘다.
- `T_LS01`: `kl_to_ls` < 0.02 가 나와야 정상이다. 넘으면 점검 코드가 틀린 것이니 멈춘다.
- 두 teacher와 실험 2 teacher(`best.pt`)의 값을 한 표로 나란히 보고한다.

## 테스트 (GPU run 전에 모두 통과)

1. teacher LS = 0, student LS = 0.1인 config에서, teacher 학습 step 손실이 LS 없는 CE와 같고 student CE는 ε = 0.1을 쓴다.
2. `teacher_label_smoothing`이 없는 config는 에러가 난다.
3. KD 손실 = T²·KL(teacher‖student) batchmean이고, 같은 logits면 0이다.
4. T = 1이고 teacher 확률이 LS 라벨(ε = 0.1)과 같으면, student logits에 대한 KD gradient가 LS-CE gradient와 같다 (실험 1·2 실패 모드 문서화).
5. `check_teacher_outputs`의 `kl_to_ls`: LS 라벨 분포면 ≈ 0, one-hot이면 ≈ 0.094.
6. 기존 테스트 전부 통과.

## 판정 규칙 (validation만 사용, test는 보지 않는다)

지표: val macro accuracy의 91–100 에포치 평균.

1. **회귀:** 실험 2 seed 0 값은 CE 60.78%, Full KD 58.91%다.&#x20;
   - `R1_ce`, `R2_full_old_T1`이 각각 ±2.0%p 안이어야 한다.
   - `R1_ce`, `R2_full_old_T1`의 `initial_model_sha256`이 실험 2와 같아야 한다 (`543d4714549b4a1318384fd102ebcfc4b65a0ccf73da68d63e1507d15e5a4e8a`).
   - 하나라도 어긋나면 코드 변경이 다른 걸 건드린 것이니 멈춘다.
2. **게이트:** `LS0_T1`, `LS0_T4` 중 (KD − `R1_ce`)가 큰 쪽을 고른다.&#x20;
   - 차이 ≥ 1.5%p면 **seed 0 통과**다.
   - 통과하면 seed 1로 확인한다: `T_LS0` seed 1 teacher, 고른 KD 설정, CE를 각각 1회 돌린다.
   - seed 1 차이 ≥ 1.0%p면 **최종 통과**다.
3. **실패**(seed 0에서 1.5%p 미만, 또는 seed 1에서 1.0%p 미만)면 멈추고 보고한다. 다음 단계(RRC 증강 arm 또는 CUB)는 사용자가 정한다.

## 보고

`reports/stage1/SUMMARY.md`에 쓴다. 모든 숫자는 스크립트가 결과 파일에서 생성한다.

- 표 1: teacher 출력 점검 (3개 teacher).
- 표 2: run별 val macro 91–100 평균, CE 대비 차이, 회귀 기준 대비 차이, 초기 checksum 일치 여부.
- 표 3: 판정 항목별 PASS/FAIL과 규칙이 낸 결정.
- 그림: 5개 run의 val 학습 곡선.
- 해석은 표 아래 3줄 이내.

`reports/stage1/`만 push한다. 가중치와 데이터는 커밋하지 않는다.

## 커밋 규칙

- 메시지는 `feat:` / `fix:` / `results:` 중 하나로 시작한다.
- 모든 커밋에 `Agent: <에이전트/모델 이름>` 트레일러를 붙인다.

## 멈추고 물어야 하는 경우

- 테스트 실패, 데이터나 teacher 해시 불일치
- 회귀 run이 범위 밖이거나 초기 checksum이 다를 때
- teacher 출력 점검 실패
- 설정을 바꿔야 할 것 같을 때 (OOM이어도 조용히 배치를 줄이지 않는다)
- test를 봐야 할 것 같을 때
