# AIM Lab 실험 3 · Stage 1 — KD 게이트

## 먼저 seed 0까지만 실행

서버에서 새 폴더에 클론하는 것부터 seed 0 실행까지의 전체 명령이다. 기존 `uv`가 PATH에 있어야 한다. 먼저 감사표 생성까지만 실행한다. 각 단계가 성공해야 다음 단계가 실행된다.

```bash
cd ~ &&
git clone --branch stage1-ls-gate https://github.com/jeehoo0507/aim-lab-test-4.git aim-lab-test-4 &&
cd aim-lab-test-4 &&
bash setup.sh install &&
bash setup.sh stage1-test &&
bash setup.sh stage1-prepare &&
bash setup.sh stage1-plan
```

`stage1-prepare`는 기준 커밋의 변경 없는 준비 코드로 클론 내부 `data/coco_single`에 COCO-10을 다운로드·준비한다. 완료 후 manifest SHA-256이 지정값과 다르면 중단한다.

여기서 멈춰 감사표의 설정과 데이터·teacher 경로를 확인한다. R2는 이번에 학습할 `T_LS01/best.pt`를 사용하며 과거 실험 2 teacher 파일은 필요 없다. 테스트 실패나 해시 문제가 있으면 학습을 시작하지 않는다.

```bash
cat reports/stage1/CONFIG_AUDIT.md
```

확인을 마친 뒤 아래 명령을 별도로 실행한다. `nohup`으로 SSH 연결이 끊겨도 학습 프로세스를 유지한다.

```bash
if cd ~/aim-lab-test-4 && mkdir -p logs; then
  nohup bash setup.sh stage1-run --jobs 5 --stop-after-seed0 > logs/stage1_seed0_launcher.log 2>&1 < /dev/null &
  tail -f logs/stage1_seed0_launcher.log
fi
```

위 블록이 **seed 0 학습 실행 명령**이다. 준비·테스트·감사표 확인을 마친 뒤 한 번만 실행한다. 새 터미널에서도 프로젝트 폴더로 이동하고 `logs/`를 만든 뒤 시작한다. 다른 위치에 클론했다면 `~/aim-lab-test-4`를 실제 경로로 바꾼다. `tail`은 로그 보기이며 `Ctrl+C`로 로그 보기를 끝내도 학습은 계속된다.

teacher 2개 병렬 → 출력 점검 → student 5개 병렬 → seed 0 판정·SUMMARY 생성까지 수행한다. 통과하면 `SEED0_DONE: RUN_SEED1, selected=T1` 또는 `selected=T4`를 출력하고 정상 종료한다. seed 1은 시작하지 않는다. 실패 판정은 기존대로 `STOP.json`에 기록한다.


## 결과 보고서를 GitHub에 올리기

학습이 끝났거나 `STOP_SEED0`로 종료됐다면 아래 전체를 실행한다. 현재 `tail -f` 화면은 `Ctrl+C`로 빠져나온다. `stage1-summary`는 저장된 결과로 표·곡선·보고용 JSON을 생성하며 학습이나 test 평가를 실행하지 않는다. `STOP.json`은 그대로 보존한다.

현재 tako-server의 `~/Documents/aim-lab-test-4` 기준이다. 다른 위치에 클론했다면 첫 줄의 경로를 바꾼다. 서버 GitHub 인증과 Git 작성자 정보가 설정되어 있어야 한다.

```bash
cd ~/Documents/aim-lab-test-4 &&
bash setup.sh stage1-summary &&
git add -- reports/stage1/ &&
git commit --only -m "results: report Stage 1 seed 0 gate" -m "Agent: Codex" -- reports/stage1/ &&
git pull --rebase origin stage1-ls-gate &&
git push origin HEAD:stage1-ls-gate
```

보고서를 먼저 커밋해 보존한 뒤 원격의 README 업데이트 등을 반영하고 push한다. `--only`로 이번 커밋에는 `reports/stage1/`만 포함한다. 데이터·가중치·`outputs/`는 추가하지 않는다. 충돌·인증 오류가 나면 그 단계에서 멈추며 강제 push나 보고서 복원·삭제를 하지 않는다. 오류 출력을 공유한다.

업로드 후 [GitHub SUMMARY](https://github.com/jeehoo0507/aim-lab-test-4/blob/stage1-ls-gate/reports/stage1/SUMMARY.md)에서 run별 성능과 CE 대비 차이를 확인할 수 있다.

### teacher 학습 진행 로그 확인

초기 가중치 다운로드가 100%에서 멈춘 것처럼 보여도 launcher 로그에는 에포치별 진행이 나오지 않는다. teacher 진행은 별도 로그에 기록된다. 현재 `tail -f` 화면에서 `Ctrl+C`로 로그 보기만 끝낸 뒤 아래를 실행한다. 백그라운드 학습은 계속되므로 학습 명령을 다시 실행하지 않는다.

아래는 현재 tako-server의 클론 위치인 `~/Documents/aim-lab-test-4` 기준이다. 홈 폴더에 클론했다면 첫 줄을 `cd ~/aim-lab-test-4 &&`로 바꾼다.

```bash
cd ~/Documents/aim-lab-test-4 &&
tail -n 30 -f \
  outputs/stage1_ls_gate/_jobs/seed_0_T_LS0.log \
  outputs/stage1_ls_gate/_jobs/seed_0_T_LS01.log
```

각 에포치가 끝나면 `T_LS0 1/30: ...` 같은 진행이 표시된다. 첫 에포치 도중에는 출력이 없을 수 있다. 계속 비어 있으면 새 터미널에서 `nvidia-smi`로 GPU 사용 상태를 확인하고 teacher 로그와 함께 확인한다.

### 로그가 비어 있거나 GPU 사용률이 불규칙할 때

첫 진행 로그는 학습 1에포치 → validation → 체크포인트·history 저장이 끝난 뒤 출력된다. `num_workers=0`이므로 CPU의 이미지 읽기·전처리를 기다리며 GPU 사용률이 떨어질 수 있지만, 사용률 변동만으로 정상 진행이라고 확정할 수는 없다.

새 터미널에서 아래 전체를 실행한다. 현재 tako-server 경로 기준이며 다른 곳에 클론했다면 첫 줄의 경로를 바꾼다. 파일과 프로세스 상태만 읽으므로 실행 중인 학습에 영향을 주지 않는다.

```bash
if cd ~/Documents/aim-lab-test-4; then
  ps -eo pid,etime,pcpu,stat,args | grep '[s]cripts.plan_stage1'
  ls -lh outputs/stage1_ls_gate/seed_0/T_LS*/{run_start.json,history.json}
  tail -n 20 logs/stage1_seed0_launcher.log \
    outputs/stage1_ls_gate/_jobs/seed_0_T_LS0.log \
    outputs/stage1_ls_gate/_jobs/seed_0_T_LS01.log
  if [ -f outputs/stage1_ls_gate/STOP.json ]; then
    cat outputs/stage1_ls_gate/STOP.json
  fi
  nvidia-smi
fi
```

- `run_start.json`이 있으면 해당 run의 모델 초기화까지 완료했다.
- `history.json`이 있으면 에포치 결과가 저장된 상태다. 파일이 있다는 것만으로 현재도 학습 중임을 보장하지는 않는다.
- 첫 에포치 완료 전에는 `history.json`이 없어서 `ls`에 파일 없음 메시지가 나올 수 있다. 프로세스의 경과 시간·CPU 사용률과 로그를 함께 확인한다.
- `STOP.json`이 있거나 오류가 보이면 기록을 보존한다. 학습 명령을 다시 실행하거나 STOP을 지우지 않는다.

원인을 확인할 때는 위 출력과 **가중치 다운로드 완료 후 기다린 시간**을 함께 전달한다.


### 나중에 seed 1까지 이어서 실행

나중에 seed 1까지 이어서 진행하려면 다음을 실행한다. 완료한 seed 0 run은 재사용한다.

```bash
if cd ~/aim-lab-test-4 && mkdir -p logs; then
  nohup bash setup.sh stage1-run --jobs 5 > logs/stage1_seed1_launcher.log 2>&1 < /dev/null &
  tail -f logs/stage1_seed1_launcher.log
fi
```

이 브랜치는 `aim-lab-test-2`의 `8ebf7af`를 기준으로 한 COCO-10 teacher LS 제거 실험이다. 서버 자산 부재에 따른 사용자 승인 변경은 [CHANGELOG](CHANGELOG.md)에 기록했다. 기존 클론의 이전 경로 오류에서 갱신하는 방법은 [실행 안내](docs/STAGE1_RUNBOOK.md#기존-클론의-경로-오류에서-갱신)에 있다.
[고정 프로토콜](docs/STAGE1_PROTOCOL.md) · [확정 답변](QUESTIONS.md) · [A5000 서버 실행 안내](docs/STAGE1_RUNBOOK.md) · [설정 감사](reports/stage1/CONFIG_AUDIT.md) · [실행 상태/결과](reports/stage1/SUMMARY.md)

실제 test 성능은 평가하지 않는다. 새 데이터·가중치·uv 설치물·캐시는 모두 이 클론 안에 저장한다.

아래는 기준 코드의 실험 2 안내다. 이번 Stage 1 실행에는 위의 전용 실행 안내를 사용한다.

# MaskedKD 실험 2 — COCO 자연 이미지 분류

Waterbirds 실험 1과 별도 저장소다. COCO 실험 2의 학습 결과와 후속 비교가 `reports/`에 있다.

검증 질문: 일반 자연 이미지에서도 student가 선택한 teacher 입력의 정보 제한이 학습 후반까지 남는가? 선택을 바꾸면 teacher 출력뿐 아니라 student 오류 교정·학습 속도·최종 분류 성능도 좋아지는가?

[최종 실험 설계](docs/EXPERIMENT_2.md) · [저장 자료와 해석](docs/ARTIFACTS.md)

후속 200에포치 비교: [신뢰도 진단에 따른 Random10→Low10/MaskedKD 전환 및 병렬 실행](docs/ADAPTIVE_200.md).

추가 비교: [Random 10→0 vs MaskedKD 실행 안내](docs/RANDOM_ANNEAL.md). `bash setup.sh anneal --jobs 4`로 기존 teacher를 재사용하여 seed 0의 두 초기화·두 방법만 별도 경로에서 학습·평가한다. 완료 후 `bash setup.sh anneal-export --push`로 결과를 공유한다.

빠른 후속 실험: [100-epoch 적응형 마스킹 파일럿](docs/ADAPTIVE_100_QUICK.md). 기존 실험 2의 세 기준 모델을 재사용하고 전환 방식 네 실행만 병렬로 학습한다.

## 구성

- COCO 2017: 이미지에 주석된 객체 종류가 하나인 10클래스. 같은 종류의 여러 객체는 허용.
- Teacher: ImageNet pretrained DeiT-Small → 분류 fine-tuning 30 epochs, seed별 한 개를 고정.
- Student: DeiT-Tiny, scratch / ImageNet pretrained를 별도 비교, 각각 100 epochs.
- 7방법 × 2초기화 × 3시드 = student 42개 + teacher 3개.
- Student는 196개 공간 패치를 모두 사용. Masked teacher는 98개 + CLS를 사용.
- 실제 COCO 데이터·GPU 학습은 서버에서 실행. 로컬 검증은 작은 합성 데이터와 CPU로 수행.

## 서버에서 실행

**모든 명령은 이 저장소를 clone한 폴더 안 터미널에서 실행한다.** 실험 1 폴더와 토큰/전역 Git 설정을 변경하지 않는다. `sudo`와 `activate`는 필요 없다. 기존 `uv`가 PATH에 있어야 한다.

```bash
git clone https://github.com/jeehoo0507/aim-lab-test-2.git
cd aim-lab-test-2
bash setup.sh check
```

`check`: lockfile 기반 Python 3.11 환경 설치 → 자동 테스트 → 작은 전체 흐름 smoke → GPU/저장 경로 출력. **실제 데이터 다운로드와 본 학습을 시작하지 않는다.** Linux x86_64는 torch 2.5.1 CUDA 12.4, macOS는 PyPI 패키지를 사용한다. NVIDIA driver 호환 여부는 서버 `check`에서 확인한다. CPU 점검은 `bash setup.sh check --device cpu`.

```bash
# 1. COCO 주석과 필요한 이미지 후보만 다운로드·품질 검사·고정 분할
bash setup.sh prepare

# 2. 실제 데이터로 student 조건 1~7개 병렬 속도·VRAM·총 소요 시간 측정
bash setup.sh benchmark

# 3. seed 0: teacher + 두 초기화 각각 CE/Full KD, validation만 확인
bash setup.sh pilot
```

Pilot은 본 실험과 같은 출력 폴더를 사용한다. **설정이 그대로라면 본 실행에서 완료한 teacher·CE·Full KD는 재학습하지 않는다.** 학습률·에포치·전처리를 바꾸면 새로운 output root/config를 사용한다. 기존 결과 위에 덮어쓰지 않는다.

Pilot 검토 후 동일 설정으로 전체 학습하는 한 줄:

```bash
bash setup.sh run
```

동시 실행은 기본 1개다. `benchmark`는 teacher와 7개 방법을 각각 **warmup 2 optimizer updates + 측정 8 updates**(update당 microbatch 4개)로 실행한다. 실제 데이터 로딩, accumulation, 대표 validation·매 epoch probe·상세 probe·checkpoint 쓰기까지 측정한다. 무작위 초기화된 임시 모델을 사용하며 실험 가중치를 덮어쓰지 않는다.

단독 실행에서 teacher와 7개 방법의 비용을 측정한 뒤, VRAM 여유가 충분한 범위에서 대표 student 조건을 2개부터 최대 7개까지 동시에 실행한다. **Teacher 3개를 먼저 만들고 42개 student 조건을 공용 작업 큐로 처리하는 실제 스케줄**의 완료 시간을 계산해, 10% 이상 단축되는 가장 빠른 작업 수를 권장한다. 여유 메모리는 프로세스당 추가 0.75GiB와 GPU 전체의 10%(최소 2GiB)를 보수적으로 확보한다. 상한을 낮추려면 예를 들어 `bash setup.sh benchmark --max-jobs 4`를 사용한다. 벤치와 2개 이상 병렬 학습은 이미 학습 프로세스 자체가 병렬이므로 중첩 DataLoader 프로세스를 만들지 않고 각 프로세스가 직접 데이터를 읽는다(`num_workers=0`). 단독 학습은 설정값 2를 유지한다.

`outputs/experiment2/benchmark.json`에 방법별 처리량·VRAM, 단독/병렬 예상 시간, 권장 작업 수, checkpoint/probe 용량을 저장한다. **전체 42개 student + teacher 3개를 처음부터 실행할 때의 추정**이며, 이미 완료한 pilot을 뺀 잔여 시간은 아니다. 원시 예측에 1.5배 여유를 둔 계획 범위를 함께 출력하며 통계적 신뢰구간은 아니다. 설치·다운로드·최종 test/export는 별도다.

벤치가 끝나면 공유용 결과 `reports/benchmark/latest.json`도 생성된다. 서버에서 아래 파일 하나만 GitHub에 올리면 다른 컴퓨터에서 시간·VRAM·권장 병렬 수를 분석할 수 있다.

```bash
git add reports/benchmark/latest.json
git commit -m "data: add A5000 benchmark"
git push origin main
```

Pilot 검토 후 권장 병렬 수를 적용하려면:

```bash
bash setup.sh run --jobs auto
```

SSH/터미널 종료 후에도 계속 실행하려면, 같은 실행을 중복으로 켜지 말고 아래 명령으로 시작한다.

```bash
mkdir -p logs
nohup bash setup.sh run --jobs auto > logs/launcher.log 2>&1 < /dev/null &
tail -f logs/launcher.log
```

`auto`는 24시간 이내, 같은 데이터·설정·코드·GPU의 완료된 벤치 결과만 사용한다. pilot 후 오래 지났으면 benchmark를 다시 실행한다. 시작 시 VRAM이 줄었으면 권장 작업 수를 자동으로 낮춘다. 먼저 seed별 teacher를 최대 3개 병렬로 완료하고, 이후 서로 독립인 7방법×2초기화×3시드의 42개 student 작업을 공용 큐에서 권장 개수만큼 병렬 실행한다. 기본 자동 병렬 상한은 **7개**이며 `--jobs 1`부터 `--jobs 7`까지 직접 지정할 수도 있다.

병렬 세부 로그는 `logs/seed_*`에 기록된다. 한 epoch 도중 중단되면 그 epoch는 다시 실행하며, 마지막 완료 epoch부터 재개한다. 다른 사람의 GPU 프로세스에는 접근하지 않는다. VRAM 여유가 충분해도 연산량은 경합하므로 2배 속도를 보장하지 않는다. 벤치 시점의 공유 서버 부하가 바뀌면 예상 시간도 달라진다.

학습을 계속하면서 중간 결과를 공유하려면 **별도 터미널**에서 다음을 실행한다:

```bash
.venv/bin/python -m coco_kd.interim --push
```

`reports/interim/<timestamp>/`에 모든 run의 검증 성능·학습 곡선·진행 epoch를 저장한다. 7개 방법이 전부 완료된 seed/초기화 묶음은 기존 probe에서 배경 선택률·전경 누락률·teacher KL·오류 교정 분석과 그림도 만든다. 방법마다 다른 시드 수를 평균내지 않도록 미완료 묶음은 개별 history에만 포함한다. GPU·가중치·진행 중 probe에는 접근하지 않으며 기존 출력 폴더를 변경하지 않는다. CPU 분석과 디스크 읽기 부하는 소량 발생한다. `incomplete` 표시는 완료 파일이 없다는 뜻이며 프로세스 생존 여부를 보장하지 않는다. Test 평가를 수행하지 않으므로 최종 성능 판정과 구분한다. `--push`를 빼면 로컬 보고서만 만든다.

학습/설정 비교가 끝난 뒤 **고정한 best·last checkpoint만 test 평가**하고 내보낸다:

```bash
bash setup.sh evaluate
bash setup.sh export --push
```

`export --push`는 새 `reports/<timestamp>/`만 commit/push한다. 데이터·가중치·가상환경·기존 다른 변경은 올리지 않는다. 인증이나 push가 실패해도 보고서는 남는다. 모델 학습 중에는 export하지 않고 완료 후 실행한다. 이 단계에서 Git 인증을 이미 설정한 서버라면 별도 설정은 필요 없다.

## 저장 경로·공간

기본 저장 위치는 clone 폴더 아래 `data/`, `outputs/`, `.cache/`, `logs/`다. `configs/experiment2.json`의 `data_root`, `output_root`를 별도 SSD/HDD 경로로 지정할 수 있다. `preflight.json`과 내보낸 `storage.json`에 실제 경로·symlink 해석·파일시스템·마운트·남은 공간을 기록한다.

| 자료 | 기본 저장 |
| --- | --- |
| Loss/LR/학습 시간/검증 성능, validation 전체 이미지 logits | 매 epoch |
| 고정 200장 student attention·교체 전/후 indices·teacher full/raw/actual logits | epoch 0 + 매 epoch |
| 같은 checkpoint에서 6가지 마스크, 랜덤 5회 반복 | 0·10·25·50·75·100 + best |
| 모델 가중치 | epoch 50 + best·final |
| 중단 재개용 weights/optimizer/scaler/history/best weights | last, 매 epoch 원자적 교체 |
| 후반 개입 실험 준비 | MaskedKD만 epoch 50의 전체 재개 상태 별도 보존 |
| Test 이미지별 logits/labels/ID | best·last, 명시적 evaluate 후 |

완료 상태의 전체 checkpoint 예상은 약 **3.8GB**다. Student는 epoch50·best·final을, teacher는 best·final을 남기고 MaskedKD만 후반 분기용 epoch50 optimizer 상태를 추가 보존한다. 실행 중인 run의 `last.pt`는 중단 재개를 위해 optimizer까지 가지지만 완료 시 final weights로 자동 축소된다.

`outputs/experiment2`의 권장 계획값은 **5GB**, 용량 경고 기준은 **10GB**다. `benchmark`가 실제 probe·validation 파일 크기까지 투영해 보고하며 10GB를 넘으면 경고만 한다. 용량 때문에 9GB에서 학습을 강제 정지하지 않는다. 이 계산은 **실험 출력 폴더 기준**이며 `data/`, `.venv/`, `.cache/`는 포함하지 않는다. 데이터는 전체 COCO archive 대신 후보 이미지만 받는다. 실제 디스크 여유가 2GiB 미만일 때만 파일 손상을 피하기 위해 안전 정지한다.

가중치가 없는 중간 epoch도 저장된 probe/validation 지표는 다시 분석할 수 있다. **그 epoch의 새로운 이미지·다른 mask 실험·gradient 분석을 임의로 다시 실행하는 것은 불가능**하다. epoch50 snapshot에서 새 평가를 하거나, 정확한 후반 분기에는 MaskedKD의 `resume_050.pt`를 사용한다. 진행 중 중단은 `last.pt`에서 재개한다.

## 구현 출처

모델의 last-layer CLS attention과 token gathering은 [공식 MaskedKD](https://github.com/effl-lab/MaskedKD)를 따르는 torch-only adaptation이다. `vendor/MaskedKD/UPSTREAM.json`에 원본 revision과 출처를 보존한다. 테스트는 원본 클래스와 logits·attention·gradient를 비교한다. 이 COCO subset·초기화·100epoch 프로토콜은 **원 논문 ImageNet 실험의 완전 재현이 아니다.**

이전 코드의 배치 크기 기반 LR 배율을 제거했다. config의 `scratch_lr=5e-4`, `pretrained_lr=5e-5`가 실제 optimizer peak LR이며 매 epoch 기록된다.
