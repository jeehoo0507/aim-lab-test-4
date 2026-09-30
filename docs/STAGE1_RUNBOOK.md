# Stage 1 서버 실행

기준은 변경하지 않은 `STAGE1_PROTOCOL.md`와 `QUESTIONS.md`의 사용자 확정 답변이다.
이번 Codex 작업 범위는 로컬 구현·테스트와 이 실행 안내까지이며, 실제 A5000 학습은 사용자가 실행한다.

## 새 서버 클론 및 실행

기존 `aim-lab-test-2` 폴더와 다른 새 폴더에서 실행한다. 기존 uv 실행 파일이 PATH에 있어야 한다.

```bash
git clone --branch stage1-ls-gate https://github.com/jeehoo0507/aim-lab-test-4.git aim-lab-test-4
cd aim-lab-test-4
bash setup.sh install
bash setup.sh stage1-test
```

테스트가 전부 통과했을 때만 다음 명령을 실행한다. 실패하면 로그를 보존하고 사용자에게 보고한다.

```bash
bash setup.sh stage1-plan
bash setup.sh stage1-run --jobs 5
```

터미널 종료 후에도 실행하려면 마지막 명령 대신 다음을 사용한다.

```bash
nohup bash setup.sh stage1-run --jobs 5 > logs/stage1_launcher.log 2>&1 < /dev/null &
tail -f logs/stage1_launcher.log
```

`stage1-test`는 전체 기존 테스트와 Stage 1 테스트를 CPU에서 실행한다. 합성 테스트 데이터는 클론 안의 임시 폴더에만 만든다. 실제 COCO test 이미지·예측·지표는 읽지 않는다. 소스·config·lockfile·테스트가 바뀌면 테스트 통과 기록이 무효화되며 GPU 실행을 거부한다. 서버에서 다시 테스트해야 하므로 로컬 통과 기록만으로는 실행되지 않는다.

## seed 0까지만 실행한 뒤 이어서 진행

```bash
bash setup.sh stage1-run --jobs 5 --stop-after-seed0
```

seed 0의 `decide_seed0` 판정과 SUMMARY 생성까지 수행한다. `RUN_SEED1`이면 `SEED0_DONE: RUN_SEED1, selected=T1` 또는 `selected=T4`를 출력하고 exit 0으로 종료하며, seed 1을 시작하거나 `STOP.json`을 만들지 않는다. `STOP_*` 판정이면 기존대로 STOP 기록을 남기고 실패 종료한다.

나중에 같은 소스·config에서 옵션을 빼고 실행한다.

```bash
bash setup.sh stage1-run --jobs 5
```

완료한 seed 0 학습 run은 기존 완료 검증을 거쳐 재사용한다. 자산·teacher 출력·seed 0 판정을 다시 확인한 뒤 선택된 seed 1으로 진행한다. 옵션은 실행 제어에만 쓰며 config·판정 규칙·임계값에 저장하거나 영향을 주지 않는다.

## 고정 실행 순서

1. 준비 manifest와 기존 teacher SHA-256, 고정 JSON, 테스트 통과 기록을 확인한다.
2. ImageNet teacher 초기 가중치를 클론 내부에 한 번 캐시한 뒤 seed 0 teacher 두 개를 병렬 실행한다.
3. epoch 30 `last.pt` 두 개와 기존 `best.pt`를 train(증강 1회, 측정 seed 0)·val에서 점검한다. train KL 기준을 통과해야 student가 시작된다.
4. seed 0 student 다섯 개를 `--jobs 5`로 동시에 실행한다. 회귀 run 완료 시 범위·초기 checksum을 확인하며 실패하면 나머지 프로세스도 중단한다. 초기 checksum은 모델 생성 직후에도 검사한다.
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

## 읽기 전용 외부 자산

| 자산 | config 절대경로 |
|---|---|
| 준비 COCO-10 | `/home/kebap/Desktop/workspace/34/aim-lab-test-2/data/coco_single` |
| 실험 2 teacher | `/home/kebap/Desktop/workspace/34/aim-lab-test-2/outputs/experiment2/seed_0/teacher/best.pt` |

파일이 없거나 해시가 다르면 중단한다. 새로 준비하거나 경로·배치·레시피를 임의로 바꾸지 않는다. 기존 자산에 파일을 쓰거나 복사·링크로 새 클론에 배치하지 않는다.

## 출력과 점검

```text
outputs/stage1_ls_gate/
  tests.json, tests.log, preflight.json, environment.json
  _jobs/seed_0_<run>.log
  seed_0/T_LS0/last.pt
  seed_0/T_LS01/last.pt
  seed_0/teacher_checks.json
  seed_0/{R1_ce,R2_full_old_T1,LS01_T4,LS0_T1,LS0_T4}/
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

프로세스를 종료한 뒤 새 클론 폴더를 삭제하면 이번 작업에서 설치·다운로드한 환경과 캐시, 새 결과가 함께 제거된다. 기존 시스템 uv 및 클론 밖의 읽기 전용 COCO/실험 2 teacher는 남는다.

## 결과 업로드

코드·설정·문서는 `aim-lab-test-4`의 `stage1-ls-gate`에 둔다. 서버에서 새로 생성한 실험 산출물은 다음 범위만 올린다.

```bash
git add reports/stage1/
git commit -m "results: report Stage 1 KD gate" -m "Agent: Codex"
git push origin stage1-ls-gate
```

가중치·데이터·환경·캐시는 `.gitignore` 대상이며 커밋하지 않는다.
