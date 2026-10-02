# Stage 1b — 새 서버에서 클론부터 실행

기존 서버의 데이터·teacher 파일이 없는 Linux CUDA 서버용이다. 원하는 작업 디렉터리에서 시작한다. Git, curl, NVIDIA 드라이버는 서버에 있어야 한다. 설치되는 uv(기존 uv가 없을 때), Python, 패키지, 캐시, 데이터, 가중치는 모두 클론 내부에 둔다. 셸 설정 파일은 수정하지 않는다. `uv.lock`은 그대로 사용한다(torch 2.5.1+cu124).

## 1. 클론 → 데이터 준비 → 테스트 → 감사표

```bash
git clone --branch stage1-ls-gate https://github.com/jeehoo0507/aim-lab-test-4.git aim-lab-test-4 &&
cd aim-lab-test-4 &&
bash setup_stage1b.sh prepare &&
bash setup_stage1b.sh test &&
bash setup_stage1b.sh plan &&
cat reports/stage1b/CONFIG_AUDIT.md
```

`prepare`는 기존 `stage1-prepare`의 `scripts.plan_stage1.prepare_data()`를 그대로 호출한다. 원본 `coco_kd/prepare.py`와 manifest SHA-256 검사를 유지하며, 데이터는 `data/coco_single`에 저장한다. manifest 기대값은 `ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e`다. 기존 준비 함수가 쓰는 `outputs/stage1_ls_gate/preparation.json` 외에 새 실험 출력은 `outputs/stage1b_aug/`에 둔다. 과거 보고서와 학습 결과는 수정하지 않는다.

테스트는 현재 서버·소스 기준으로 통과해야 한다. 실패하면 다음 단계는 실행되지 않고 `outputs/stage1b_aug/STOP.json`에 기록한다. 감사표의 teacher SHA는 학습 전 `UNAVAILABLE`이며 학습·출력 점검 뒤 실제 값으로 갱신된다.

## 2. 감사표 확인 후 학습 실행

바로 위 명령을 실행한 **같은 클론 폴더**에서 아래를 실행한다. 새 터미널에서는 먼저 실제 클론 폴더로 이동한다.

```bash
if test -f setup_stage1b.sh && mkdir -p logs; then
  nohup bash setup_stage1b.sh run --jobs 7 > logs/stage1b_launcher.log 2>&1 < /dev/null &
  tail -f logs/stage1b_launcher.log
fi
```

`tail`은 로그 보기다. `Ctrl+C`로 로그 보기만 종료해도 백그라운드 학습은 계속된다. 같은 run을 중복 실행하지 않는다.

실행 순서:

1. 테스트 통과 기록·데이터 해시·고정 config·CUDA 환경 확인.
2. 새 `T_LS0` 학습: seed 0, ImageNet DeiT-S, LS=0, flip, 30에포치, 기존 Stage 1 teacher 레시피. 공통 학습 루프를 사용한다.
3. `outputs/stage1b_aug/seed_0/T_LS0/last.pt`의 실제 SHA 기록. 기존 teacher의 고정 SHA는 요구하지 않는다.
4. 기존 teacher 출력 측정 함수로 augmented train 1회(측정 seed 0)와 val 측정. **train kl_to_ls ≥0.02**만 통과 기준이며 val은 보고만 한다. 점검 후 체크포인트가 바뀌면 차단한다.
5. 새 flip 경로를 기존 100에포치 스케줄 그대로 2에포치 실행한다. 초기 checksum `543d4714549b4a1318384fd102ebcfc4b65a0ccf73da68d63e1507d15e5a4e8a`는 **엄격 검사**하며 불일치 시 첫 에포치 전에 STOP. 새 서버의 epoch 1·2 loss/val은 기존 기록과 차이 및 `1e-6` 이내 여부를 보고만 한다. 유한하지 않은 지표는 실패로 처리한다.
6. 아래 student **7개를 한 번에 병렬 실행**하고 SUMMARY 생성 후 종료한다. 자동 seed 1이나 `--stop` 옵션은 없다.

| run | 증강 | teacher | T |
|---|---|---|---|
| ce_flip | flip | 없음 | 미사용 |
| LS0_T4_flip | flip | 새 T_LS0 last.pt | 4 |
| ce_rrc | rrc | 없음 | 미사용 |
| LS0_T1_rrc | rrc | 새 T_LS0 last.pt | 1 |
| LS0_T4_rrc | rrc | 새 T_LS0 last.pt | 4 |
| ce_rrc_mix | rrc_mix | 없음 | 미사용 |
| LS0_T4_rrc_mix | rrc_mix | 새 T_LS0 last.pt | 4 |

모든 KD는 같은 새 teacher 파일을 사용한다. 오류·OOM·해시 불일치·teacher 점검 실패 시 STOP을 기록하고 진행하지 않는다. 병렬 student 중 하나가 실패하면 나머지도 종료한다. 배치·병렬 수·임계값을 자동으로 줄이거나 STOP을 자동 삭제하지 않는다.

## 로그와 보고서

```bash
tail -f logs/stage1b_launcher.log
```

teacher 학습 중에는 매 에포치 결과가 파일로 저장된다. 로그가 조용해도 이 파일과 GPU 사용량을 확인한다.

```bash
tail -n 60 outputs/stage1b_aug/seed_0/T_LS0/history.json
nvidia-smi
```

7개 student가 시작된 뒤 개별 로그:

```bash
tail -f outputs/stage1b_aug/_jobs/*.log
```

결과 재생성과 확인:

```bash
bash setup_stage1b.sh summary &&
cat reports/stage1b/SUMMARY.md
```

val macro accuracy의 91–100에포치 평균을 사용한다. 각 KD는 **같은 증강 CE 대비 ≥+1.5pp**이면 PASS다. flip 쌍은 과거 Stage 1 CE 60.78%, LS0_T4 60.21%와 새 값을 나란히 보고하며 과거 값과의 차이는 추가 중단 기준으로 쓰지 않는다. 7개 완료 후 KD가 모두 FAIL이어도 정상 종료한다. test는 읽거나 평가하지 않는다.

`SUMMARY.md`에는 teacher train/val 측정과 실제 SHA, 과거/새 결과, 2에포치 지표 차이, 7개 val 곡선이 들어간다. Mac CPU 동등성 테스트는 합성 데이터에서 원본 ↔ 새 flip의 초기 checksum·2에포치 loss/val이 정확히 같은지 확인하며 서버 기준으로 대체하지 않는다.

## 결과만 업로드

현재 클론 폴더에서 실행한다. 변경이 이미 커밋됐다면 커밋 단계를 건너뛴다. 데이터·가중치·outputs는 추가하지 않는다.

```bash
bash setup_stage1b.sh summary &&
git add -- reports/stage1b/ &&
if ! git diff --cached --quiet -- reports/stage1b/; then
  git commit --only -m "results: report Stage 1b augmentation gate" -m "Agent: Codex" -- reports/stage1b/
fi
```

위 명령이 성공했을 때만:

```bash
git pull --rebase origin stage1-ls-gate &&
git push origin HEAD:stage1-ls-gate
```

충돌·인증 오류는 중단하고 출력 내용을 공유한다.

## 유지되는 증강 설정

RRC: 224, scale=(0.08,1.0), ratio=(3/4,4/3), bicubic + antialias=True, horizontal flip. foreground mask에 동일 기하 변환을 nearest로 적용한다. val/probe는 기존 전체 이미지 resize다.

rrc_mix: batch 확률 1, mixup α=0.8 / cutmix α=1.0을 50%씩 선택, 역순 batch pairing, 실제 cutmix 면적으로 λ 보정. student CE target에는 LS=0.1을 한 번 적용하며 teacher는 같은 혼합 이미지를 받는다. 혼합 train accuracy는 기록하지 않는다. RRC/mix RNG를 모델 RNG와 분리하고 run 간 crop/mix 해시를 비교한다.

uv 자동 설치는 [공식 unmanaged 설치](https://docs.astral.sh/uv/reference/installer/#unmanaged-installations)를 사용한다. mixup/cutmix의 batch 선택·pairing·bbox 규칙은 [timm 고정 소스](https://github.com/huggingface/pytorch-image-models/blob/92dbd9e3f9b5d61c4d008223410781da983239fc/timm/data/mixup.py)를 따른다. 새 의존성은 추가하지 않는다.
