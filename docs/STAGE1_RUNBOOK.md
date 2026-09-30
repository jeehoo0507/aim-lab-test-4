# Stage 1 서버 실행

기준은 변경하지 않은 `STAGE1_PROTOCOL.md`와 `QUESTIONS.md`의 사용자 확정 답변·후속 변경 지시다. 서버 자산 부재에 따른 최신 변경은 `CHANGELOG.md`에 기록했다.
이번 Codex 작업 범위는 로컬 구현·테스트와 이 실행 안내까지이며, 실제 A5000 학습은 사용자가 실행한다.

## 새 서버 클론 및 실행

기존 `aim-lab-test-2` 폴더와 다른 새 폴더에서 실행한다. 기존 uv 실행 파일이 PATH에 있어야 한다.

```bash
cd ~ &&
git clone --branch stage1-ls-gate https://github.com/jeehoo0507/aim-lab-test-4.git aim-lab-test-4 &&
cd aim-lab-test-4 &&
bash setup.sh install &&
bash setup.sh stage1-test &&
bash setup.sh stage1-prepare &&
bash setup.sh stage1-plan
```

`stage1-prepare`는 `8ebf7af`의 변경 없는 `coco_kd/prepare.py`를 기본 인자 그대로 사용한다. 소스 SHA-256을 먼저 확인하고, 준비 완료 직후 manifest SHA-256이 `ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e`와 같은지 검사한다. 다르면 `STOP_PREPARE`를 기록하고 중단하며 기준 해시를 자동 변경하지 않는다. 준비 기록은 `outputs/stage1_ls_gate/preparation.json`에 저장한다.

여기서 멈춰 감사표를 확인한다. 테스트가 전부 통과하고 데이터 해시와 설정·teacher 경로를 확인한 뒤에만 학습을 시작한다. 실패하면 로그를 보존하고 사용자에게 보고한다.

```bash
cat reports/stage1/CONFIG_AUDIT.md
```

확인을 마친 뒤 아래 명령을 별도로 실행한다. SSH 연결 종료 후에도 학습이 유지되도록 `nohup`을 사용한다.

```bash
if cd ~/aim-lab-test-4 && mkdir -p logs; then
  nohup bash setup.sh stage1-run --jobs 5 --stop-after-seed0 > logs/stage1_seed0_launcher.log 2>&1 < /dev/null &
  tail -f logs/stage1_seed0_launcher.log
fi
```

위 블록이 **seed 0 학습 실행 명령**이다. 준비·테스트·감사표 확인을 마친 뒤 한 번만 실행한다. 새 터미널에서도 프로젝트 폴더로 이동하고 `logs/`를 만든 뒤 시작한다. 다른 위치에 클론했다면 `~/aim-lab-test-4`를 실제 경로로 바꾼다. `tail`은 로그 보기이며 `Ctrl+C`로 로그 보기를 끝내도 학습은 계속된다.

`stage1-test`는 전체 기존 테스트와 Stage 1 테스트를 CPU에서 실행한다. 합성 테스트 데이터는 클론 안의 임시 폴더에만 만든다. 실제 COCO test 이미지·예측·지표는 테스트 실행 중 읽지 않는다. 별도 `stage1-prepare`는 기존 준비 절차대로 전체 split을 구성·무결성 검사하지만 모델의 test 성능을 평가하지 않는다. 소스·config·lockfile·테스트가 바뀌면 테스트 통과 기록이 무효화되며 GPU 실행을 거부한다. 서버에서 다시 테스트해야 하므로 로컬 통과 기록만으로는 실행되지 않는다.


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


## 기존 클론의 경로 오류에서 갱신

이 절차는 이전 `/home/kebap/...` 파일 부재로 **학습 시작 전에 종료한 경우**에 사용한다. 실행 중인 작업이 없어야 한다. 기존 출력과 서버에서 생성된 보고서를 백업한 뒤 코드를 갱신한다. 다른 학습 실패·해시 불일치의 STOP을 우회하는 용도로 쓰지 않는다.

```bash
cd ~/aim-lab-test-4 &&
stage1_backup=".cache/before-local-data-$(date +%Y%m%d_%H%M%S)" &&
mkdir -p "$stage1_backup" &&
cp -a reports/stage1 "$stage1_backup/reports-stage1" &&
git restore --source=HEAD -- reports/stage1 &&
git pull --ff-only origin stage1-ls-gate &&
mv outputs/stage1_ls_gate "$stage1_backup/outputs-stage1-ls-gate" &&
bash setup.sh stage1-test &&
bash setup.sh stage1-prepare &&
bash setup.sh stage1-plan
```

이전 `STOP.json`과 로그는 백업에 보존된다. 준비·감사표 확인 후 위의 `nohup ... --stop-after-seed0` 명령을 별도로 실행한다. 소스와 teacher 정의가 바뀌었으므로 이전 코드로 완료한 run을 새 run에 혼합하지 않는다.

## seed 0까지만 실행한 뒤 이어서 진행

```bash
if cd ~/aim-lab-test-4 && mkdir -p logs; then
  nohup bash setup.sh stage1-run --jobs 5 --stop-after-seed0 > logs/stage1_seed0_launcher.log 2>&1 < /dev/null &
  tail -f logs/stage1_seed0_launcher.log
fi
```

seed 0의 `decide_seed0` 판정과 SUMMARY 생성까지 수행한다. `RUN_SEED1`이면 `SEED0_DONE: RUN_SEED1, selected=T1` 또는 `selected=T4`를 출력하고 exit 0으로 종료하며, seed 1을 시작하거나 `STOP.json`을 만들지 않는다. `STOP_*` 판정이면 기존대로 STOP 기록을 남기고 실패 종료한다.

나중에 같은 소스·config에서 옵션을 빼고 실행한다.

```bash
if cd ~/aim-lab-test-4 && mkdir -p logs; then
  nohup bash setup.sh stage1-run --jobs 5 > logs/stage1_seed1_launcher.log 2>&1 < /dev/null &
  tail -f logs/stage1_seed1_launcher.log
fi
```

완료한 seed 0 학습 run은 기존 완료 검증을 거쳐 재사용한다. 자산·teacher 출력·seed 0 판정을 다시 확인한 뒤 선택된 seed 1으로 진행한다. 옵션은 실행 제어에만 쓰며 config·판정 규칙·임계값에 저장하거나 영향을 주지 않는다.

## 고정 실행 순서

1. 준비 manifest SHA-256, 고정 JSON, 테스트 통과 기록을 확인한다. 과거 실험 2 teacher는 요구하지 않는다.
2. ImageNet teacher 초기 가중치를 클론 내부에 한 번 캐시한 뒤 seed 0 teacher 두 개를 병렬 실행한다.
3. epoch 30 `last.pt` 두 개와 이번에 학습한 `T_LS01/best.pt`를 train(증강 1회, 측정 seed 0)·val에서 점검한다. 기존대로 두 last teacher에만 train KL 기준을 적용하고 best 행은 보고만 한다.
4. teacher 학습·점검 후 seed 0 student 다섯 개를 `--jobs 5`로 동시에 실행한다. `R2_full_LS01best_T1`은 `T_LS01/best.pt`를 T=1로 사용한다. R1 회귀 기준 60.78 ±2.0%p, R2 기준 58.91 ±2.0%p와 초기 checksum 기준은 유지한다. 회귀 실패면 나머지 프로세스도 중단한다. 초기 checksum은 모델 생성 직후에도 검사한다.
5. seed 0 후보의 KD−CE 차이가 1.5%p 이상인 경우에만 seed 1의 T_LS0 teacher를 학습·점검하고, 선택된 온도 KD와 CE를 각각 한 번 실행한다. 완전 동률이면 T=1이다.
6. seed 1 차이가 1.0%p 이상이면 FINAL_PASS. 그 외에는 중단하고 보고한다. RRC, CUB, test 평가로 넘어가지 않는다.

모든 student의 지표는 validation macro accuracy epoch 91–100 평균이다. 반올림 전 값으로 판정한다.
seed 1 JSON에는 T=1과 T=4 후보가 모두 있지만 실행기는 seed 0에서 선택된 하나만 실행한다.
teacher·CE config에서 사용하지 않는 온도는 1, CE의 미사용 teacher LS는 0으로 명시하며 감사표에 미사용으로 표시한다.

## 실험 1·2 실패 모드와 손실 검증

10클래스에서 student LS 라벨은 `q = 0.9·onehot + 0.01`이다. T=1에서 teacher 확률이 q와 같다면
`KL(q || softmax(z)) = CE(q, softmax(z)) − H(q)`이며, 두 손실의 student logits gradient는 모두 `(softmax(z) − q) / batch_size`이다.
따라서 `(1−α)·LS-CE + α·KD`의 gradient도 LS-CE와 같아진다. 손실 값의 상수 차이는 학습 신호를 추가하지 않는다.
새 테스트는 실제 teacher 학습 step의 LS 없는 CE, student의 LS=0.1, KD의 T²·batchmean·방향, 이 gradient 동치, LS 분포 KL≈0 및 one-hot KL=−log(0.91)≈0.09431을 검증한다.

## 클론 내부 자산

| 자산 | 클론 기준 경로 |
|---|---|
| 준비 COCO-10 | `data/coco_single` |
| R2 teacher | `outputs/stage1_ls_gate/seed_0/T_LS01/best.pt` |
| LS01_T4 teacher | `outputs/stage1_ls_gate/seed_0/T_LS01/last.pt` |
| LS0 후보 teacher | `outputs/stage1_ls_gate/seed_0/T_LS0/last.pt` |

새 데이터 준비와 R2 teacher 대체는 서버에 기존 자산이 없다는 사용자의 명시적 변경 지시에 따른다. R2는 원래 실험 2 teacher의 동일 파일 재현이 아니며, 비교 기준 수치만 사용자 지시대로 유지한다. 과거 teacher의 고정 SHA 검사는 제거했지만 새 teacher의 실제 SHA 기록과 변경 검사는 유지한다. 준비 데이터 해시가 다르거나 학습·점검에 실패하면 멈춘다.

## 출력과 점검

```text
outputs/stage1_ls_gate/
  tests.json, tests.log, preparation.json, preflight.json, environment.json
  _jobs/seed_0_<run>.log
  seed_0/T_LS0/last.pt
  seed_0/T_LS01/last.pt
  seed_0/teacher_checks.json
  seed_0/{R1_ce,R2_full_LS01best_T1,LS01_T4,LS0_T1,LS0_T4}/
  seed_1/T_LS0/last.pt                       # seed 0 통과 때만
  seed_1/{R1_ce,선택된_KD}/                  # seed 0 통과 때만
reports/stage1/
  CONFIG_AUDIT.md, SUMMARY.md, decision.json
  val_curves.png                            # 측정 결과가 생긴 뒤
  tests.json, seed_<n>/...                   # train/val 결과 파일만
```

각 run의 `run_start.json`, `result.json`, checkpoint에 teacher SHA-256과 초기 모델 checksum이 기록된다. CE에는 teacher 경로·SHA가 null이다. 새 teacher 폴더가 서로 달라 충돌하지 않는다.

```bash
bash setup.sh stage1-summary
```

이 명령은 결과 파일에서 세 표와 최대 다섯 개 seed 0 val 곡선을 재생성한다. 학습 전에는 미측정 칸을 `NOT_RUN`으로 표시하며 성능이나 그래프를 만들어 내지 않는다. `stage1-plan`도 실제 파일이 없는 teacher 해시는 `UNAVAILABLE`로 표시한다.

오류·OOM·해시 불일치·판정 실패 시 `STOP.json`과 로그를 보존한다. 자동 배치 축소·재시도는 없다. 원인과 설정을 사용자와 확인하기 전에는 STOP 기록을 지우거나 다시 실행하지 않는다. 일반적인 프로세스 중단 후 재개는 동일 config/source와 저장된 `last.pt`가 일치할 때 기존 학습 루프가 처리한다.

## uv 격리 및 제거

`setup.sh`는 상속받은 외부 캐시 경로를 덮어쓰고 다음을 클론 안으로 지정한다.

| 항목 | 저장 위치 |
|---|---|
| 가상환경 | `.venv/` |
| uv cache / Python / tools | `.cache/uv`, `.cache/python`, `.cache/uv-tools` |
| Python/tool 실행 파일 디렉터리 | `.cache/python-bin`, `.cache/uv-tool-bin` |
| Torch / XDG / 임시 파일 | `.cache/torch`, `.cache/xdg`, `.cache/tmp` |
| Matplotlib / Hugging Face / pip | `.cache/matplotlib`, `.cache/huggingface`, `.cache/pip` |
| Python bytecode / CUDA / Triton | `.cache/pycache`, `.cache/cuda`, `.cache/triton` |

Python은 `uv python install 3.11 --no-bin`, 의존성은 기존 lockfile로 `uv sync --frozen --managed-python --python 3.11`을 사용한다. 전역 Python·패키지·셸 설정은 수정하지 않는다. 명령은 항상 `bash setup.sh ...`로 실행해 이 환경을 전달한다.

프로세스를 종료한 뒤 새 클론 폴더를 삭제하면 이번 작업에서 준비한 데이터·가중치·환경·캐시·결과가 함께 제거된다. 기존 시스템 uv와 클론 밖의 파일은 삭제 대상이 아니다.

## 결과 업로드

코드·설정·문서는 `aim-lab-test-4`의 `stage1-ls-gate`에 둔다. 서버에서 새로 생성한 실험 산출물은 다음 범위만 올린다.

```bash
git add reports/stage1/
git commit -m "results: report Stage 1 KD gate" -m "Agent: Codex"
git push origin stage1-ls-gate
```

가중치·데이터·환경·캐시는 `.gitignore` 대상이며 커밋하지 않는다.
