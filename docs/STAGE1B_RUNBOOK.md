# Stage 1b — RRC / mixup 서버 실행

Stage 1에서 만든 데이터와 `T_LS0/last.pt`를 재사용한다. teacher는 재학습하지 않는다. 다섯 student를 100에포치씩 병렬 실행하고 결과를 생성한 뒤 종료한다. `--stop` 옵션은 없다. 각 KD의 PASS/FAIL과 무관하게 다섯 run을 완료한다. test 평가는 금지한다.

## 1. 기존 서버 클론 갱신·테스트·감사표

현재 tako-server 경로 기준이다. 다른 곳에 클론했다면 첫 줄만 실제 경로로 바꾼다. 기존 학습이 종료되고 Stage 1 보고서가 커밋·업로드된 상태에서 실행한다.

```bash
cd ~/Documents/aim-lab-test-4 &&
git pull --ff-only origin stage1-ls-gate &&
bash setup_stage1b.sh test &&
bash setup_stage1b.sh plan &&
cat reports/stage1b/CONFIG_AUDIT.md
```

기존 `.venv`와 `uv.lock`을 사용한다. 새 의존성은 설치하지 않는다. `setup_stage1b.sh`는 기존과 같은 uv·Torch·XDG·임시 디렉터리 격리를 클론 내부에 적용하며 전역 환경과 셸 설정을 바꾸지 않는다.

다음 자산이 이미 있어야 한다. 새 클론만으로는 teacher 가중치가 생기지 않는다.

| 자산 | 경로 | 필수 SHA-256 |
|---|---|---|
| 데이터 manifest | `data/coco_single/manifest.json` | `ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e` |
| LS 0 teacher | `outputs/stage1_ls_gate/seed_0/T_LS0/last.pt` | `6e31acca311a19b307a722008ef6c7f65259445ec04217f7d86f8486faf293d6` |

자산 부재·해시 불일치·테스트 실패는 멈추고 확인한다. teacher 재학습·해시 변경·배치 축소는 자동으로 하지 않는다.

## 2. 학습 실행

감사표를 확인한 뒤 아래 전체를 한 번 실행한다.

```bash
if cd ~/Documents/aim-lab-test-4 && mkdir -p logs; then
  nohup bash setup_stage1b.sh run --jobs 5 > logs/stage1b_launcher.log 2>&1 < /dev/null &
  tail -f logs/stage1b_launcher.log
fi
```

먼저 **서버 flip 회귀 게이트**가 자동 실행된다. 기존 100에포치 레시피의 첫 2에포치만 새 코드의 flip 경로로 실행하며, 에포치 수나 LR 스케줄을 2에포치용으로 바꾸지 않는다. 이 검증이 모두 통과해야 다섯 student를 동시에 시작한다.

1. R1_ce 초기 checksum이 `543d4714549b4a1318384fd102ebcfc4b65a0ccf73da68d63e1507d15e5a4e8a`와 정확히 일치해야 한다.
2. epoch 1·2의 train loss 및 val macro가 보존된 `reports/stage1/seed_0/R1_ce/history.json`과 **절대 오차 1e-6** 이내여야 한다. 상대 오차는 허용하지 않으며 val macro는 0–1 단위로 비교한다.
3. 실패하면 `outputs/stage1b_aug/STOP.json`을 기록하고 **Stage 1b 다섯 run은 시작하지 않는다**. STOP을 임의로 지우지 않는다.

검증만 별도로 실행하려면 `bash setup_stage1b.sh verify-flip`을 사용한다. 같은 소스·설정·환경에서 통과 기록이 있으면 이후 `run`이 재사용한다. 모든 검증 출력은 `outputs/stage1b_aug/_flip_verification/`에 있으며 Stage 1 출력에는 쓰지 않는다.

| run | 증강 | 방법 | T | 비교 CE |
|---|---|---|---|---|
| ce_rrc | RRC + flip | CE | 미사용 | 기준 |
| LS0_T1_rrc | RRC + flip | Full KD | 1 | ce_rrc |
| LS0_T4_rrc | RRC + flip | Full KD | 4 | ce_rrc |
| ce_rrc_mix | RRC + flip + mix | CE | 미사용 | 기준 |
| LS0_T4_rrc_mix | RRC + flip + mix | Full KD | 4 | ce_rrc_mix |

다섯 run이 모두 완료되면 `STAGE1B_DONE: ...`에 각 KD의 PASS/FAIL이 표시되고 정상 종료한다. 각 KD는 같은 증강 CE 대비 val macro 91–100 에포치 평균이 1.5%p 이상 높을 때 PASS다. seed 1이나 다음 실험은 자동 실행하지 않는다. 완료된 run은 같은 소스·설정·자산 검사를 통과한 경우만 재사용한다.

## 3. 진행 로그

launcher 로그에서 현재 `tail -f` 중이면 `Ctrl+C`로 로그 보기만 끝낸 뒤 실행한다. 백그라운드 학습은 계속된다.

```bash
cd ~/Documents/aim-lab-test-4 &&
tail -n 20 -f outputs/stage1b_aug/_jobs/*.log
```

`_jobs` 로그는 서버 flip 게이트 통과 후 생성된다. 그 전에는 `logs/stage1b_launcher.log`에서 2에포치 검증 진행을 확인한다. 각 run 로그에는 에포치 진행과 validation 값이 기록된다.

## 4. 결과 보고서 업로드

실행이 종료된 뒤 아래 전체를 실행한다. 저장된 train/val 기록으로만 보고서를 생성하며 학습은 다시 시작하지 않는다. Stage 1b 보고서만 커밋한다.

```bash
cd ~/Documents/aim-lab-test-4 &&
bash setup_stage1b.sh summary &&
git add -- reports/stage1b/ &&
{
  if git diff --cached --quiet -- reports/stage1b/; then
    echo "보고서 변경 없음: 기존 커밋 업로드를 계속합니다."
  else
    git commit --only -m "results: report Stage 1b augmentation gate" -m "Agent: Codex" -- reports/stage1b/
  fi
} &&
git pull --rebase origin stage1-ls-gate &&
git push origin HEAD:stage1-ls-gate
```

Git 인증은 기존 서버 설정을 사용한다. 충돌·인증 오류가 나면 해당 단계에서 멈추고 오류를 공유한다. 강제 push나 기존 보고서 삭제를 하지 않는다.

[Stage 1b SUMMARY](../reports/stage1b/SUMMARY.md)에는 기존 Stage 1 flip 결과와 새 다섯 run의 비교표 및 새 다섯 val 곡선이 들어간다. 실제 학습 전 숫자는 `NOT_RUN`으로 표시한다. Stage 1 baseline은 커밋 `c5a826d`에서 읽어 파일별 SHA-256으로 고정했으며 수정하지 않는다.

## 고정 증강과 구현

- `train_augmentation`은 **Stage 1b JSON에서 필수**이며 기본값이 없다. 기존 Stage 1 JSON은 그대로 사용한다.
- RRC는 224×224, scale=(0.08,1.0), ratio=(3/4,4/3), 이미지 bicubic + antialias=True다. mask는 같은 crop·flip을 nearest로 적용한다. foreground가 사라져도 임의로 crop을 재추첨하지 않는다.
- HorizontalFlip 확률은 기존과 같은 0.5다. ImageNet normalize 및 val/probe의 전체 resize 224는 기존 그대로다. Probe 실행은 Stage 1과 같이 꺼져 있다.
- rrc_mix는 batch마다 mixup α=0.8 또는 cutmix α=1.0을 50%씩 선택하고 확률 1로 적용한다. 역순 batch pairing, CutMix의 실제 잘린 면적에 따른 lambda 보정을 사용한다.
- [timm의 batch Mixup 구현](https://github.com/huggingface/pytorch-image-models/blob/92dbd9e3f9b5d61c4d008223410781da983239fc/timm/data/mixup.py)을 고정 참조했다. timm 패키지를 추가하지 않아 기존 lockfile이 유지된다.
- CE target은 `lambda * LS(y_a) + (1-lambda) * LS(y_b)`이고 LS=0.1을 한 번만 적용한다. 혼합 batch의 train accuracy는 null이며 의미를 기록한다.
- teacher와 student는 동일하게 crop·flip·mix가 적용된 입력 텐서를 사용한다. teacher의 원본 이미지 출력들을 섞지 않는다.
- crop·mix 난수는 모델 난수와 분리한다. 동일 seed·epoch·sample/batch에 동일 crop·mix를 사용하고, CE/KD 짝의 에포치별 증강 해시가 같아야 보고서가 생성된다.
- 기존 `coco_kd/train.py`의 학습 루프와 optimizer·LR·평가 경로를 공유한다. 학습 루프를 별도로 복제하지 않는다.

## 로컬 CPU 검사와 서버 기준 분리

`test`는 기존 테스트와 Stage 1b 테스트를 수행한다. CPU flip 회귀 테스트는 실제 DeiT-Ti 모델과 합성 10클래스 데이터를 사용한다(배치 2, 에포치당 train batch 2, val 20개). 같은 플랫폼의 원본 코드와 새 flip 경로를 각각 2에포치 실행해 초기 checksum·loss·val 및 종료 가중치가 완전히 일치하는지 검사한다. 이는 코드 동등성 검사이며 실제 COCO 성능 측정이 아니다.

- `cpu_flip_verification.json`: `scope=same_platform_cpu`. Mac 결과는 Mac 로컬 비교 기준으로만 기록한다. 서버에서 CPU 테스트를 실행하면 그 서버의 CPU 비교 기록으로 갱신된다.
- `flip_verification.json`: `scope=server_a5000`. 실제 COCO·CUDA 실행에 대한 위의 역사적 기준 검증이다. CPU 통과 기록으로 대체할 수 없다.
- `tests.json`: 테스트를 실행한 플랫폼·Python·torch·소스 fingerprint가 포함된다. Mac의 통과 기록으로 서버 실행을 허용하지 않는다.

기존 Stage 1 config·보고서·원문 프로토콜·setup.sh·uv.lock은 보존한다. 공유 train.py가 확장되므로 과거 source fingerprint를 우회해서 Stage 1을 재개하지 않는다. 이번 작업에는 위의 Stage 1b 전용 명령만 사용한다.
