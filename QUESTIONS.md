# Stage 1 구현 계획 및 사전 질문

상태: 사용자 Q1–Q6 답변 수신. 로컬 구현·테스트 및 서버 실행 명령 작성이 승인되었다. GPU 실행은 사용자가 수행한다.
기준 문서: `docs/STAGE1_PROTOCOL.md`에 사용자가 마지막으로 보낸 프로토콜 원문을 보존한다.

## 최초 제출 시점의 추가 지시와 확인 사항 (최신 후속 변경은 문서 끝 참조)

- 코드 기준: `jeehoo0507/aim-lab-test-2`의 `8ebf7afe09c9cc701236a2b972b2f12b04c9fbed`, 작업 브랜치 `stage1-ls-gate`.
- 사용자가 별도로 지정한 업로드 대상: `https://github.com/jeehoo0507/aim-lab-test-4` (`origin`). 원본 저장소는 `upstream`으로 보존한다.
- 로컬 클론: `/Users/choejihu/Documents/ChatGPT/aim lab test 4/aim-lab-test-2`.
- uv로 설치하며, 이번 작업에서 설치·다운로드하는 Python, 의존성, 캐시, 임시 파일은 모두 클론 내부에 두어 폴더 삭제로 제거되게 한다. 전역 환경과 셸 설정은 수정하지 않는다.
- 호스트는 macOS arm64이며 기존 uv 0.11.24가 있다. 기존 uv 자체는 이번 작업의 설치물이 아니다.
- 클론에 `data/coco_single/` 및 `outputs/experiment2/`가 없다. 데이터나 가중치를 새로 만들거나 다른 파일로 대체하지 않는다.
- `pyproject.toml`과 `uv.lock`이 이미 있으며 Python 3.11, torch 2.5.1, torchvision 0.20.1 등이 고정되어 있다.
- 아래 최초 제안보다 이 문서 마지막의 사용자 확정 답변이 우선한다.

## 구현 계획 (12줄)

1. 아래 질문의 답변을 기록하고 프로토콜 원문은 변경하지 않는다.
2. 기존 lockfile을 유지하며 uv Python·가상환경·도구·캐시·임시 경로와 Torch 등 라이브러리 캐시를 클론 내부로 고정한다.
3. `teacher_label_smoothing`을 필수로 검증하고 teacher CE 호출과 기존 student CE/KD 호출을 명시적으로 분리한다.
4. `teacher_checkpoint` 경로와 실행 시작 SHA-256 기록을 추가하고 CE는 teacher 없이 실행한다.
5. 고정 레시피를 명시한 두 teacher 및 다섯 student JSON과 역할별 config 감사표 생성기를 만든다.
6. 준비 데이터와 기존 teacher의 지정 해시를 검증하고 출력 경로를 `outputs/stage1_ls_gate/` 아래로 제한한다.
7. teacher의 train·val 정확도, KL, 정규화한 비정답 엔트로피를 측정하고 결과 파일과 비교표를 생성한다.
8. 요구된 손실·gradient·KL 단위 테스트와 기존 테스트를 실행하며 실패하면 멈춘다.
9. seed 0 teacher 두 개를 학습하고 epoch 30 last 체크포인트 출력 점검을 통과한 뒤 student를 실행한다.
10. 회귀 범위·초기 checksum과 seed 0 게이트를 검증하고 통과할 때만 seed 1 teacher·선택 KD·CE를 실행한다.
11. 결과 파일에서만 SUMMARY의 세 표와 다섯 val 곡선을 생성하고 해석은 3줄 이내로 쓴다.
12. 합의한 범위만 새 저장소에 push하며 모든 커밋에 지정 접두사와 `Agent: Codex` 트레일러를 붙인다.

## 최초 질문 (기록 보존)

### Q1. 새 저장소에 올릴 범위

`aim-lab-test-4`는 조회 시 공개된 ref가 없었다. 프로토콜은 원문 커밋을 요구하면서 `reports/stage1/`만 push하라고 한다. Git은 파일별이 아니라 커밋과 조상 이력을 push하므로 아래 중 어느 방식인지 확인이 필요하다.

- A: 새 저장소에 기준 코드와 Stage 1 코드·설정·문서도 올리고, 실험 산출물은 `reports/stage1/`만 올린다. 데이터·가중치는 제외한다.
- B: 코드·설정·프로토콜 커밋은 로컬에만 보존하고, 새 저장소에는 별도 이력으로 `reports/stage1/`만 올린다.

### Q2. GPU 실행 위치 및 기존 자산

GPU 실험을 실행할 서버/접속 방법, 준비 COCO-10 디렉터리, 실험 2 teacher `best.pt`의 실제 위치를 알려 달라. 또는 이번 작업은 로컬 구현·테스트와 서버 실행 명령 제공까지만인지 알려 달라. 외부에 있는 원본 자산은 읽기만 하고 클론 삭제 시 원본까지 지우지 않는다. 기존 실험 환경 기록이 있다면 동일 초기 checksum 재현을 위해 그 경로도 필요하다.

### Q3. teacher 점검의 판정 대상과 seed

제안: KL 임계값은 **학습 증강을 적용한 train 전체 1회 평균**에만 적용하고 val 값은 보고만 한다. 각 teacher의 측정 시작마다 seed 0으로 초기화하여 동일한 train 증강을 사용하고, 정확도는 overall과 macro를 모두 보고한다. seed 1 teacher도 같은 측정 seed 0으로 점검하고 train KL ≥ 0.02 조건을 적용한다. 이 정의로 진행할지 확인해 달라.

### Q4. JSON 필수 필드 중 해당 역할에서 사용하지 않는 값

제안: CE와 teacher run의 미사용 `temperature`는 1.0으로 명시하고 감사표에서는 미사용으로 표시한다. CE의 미사용 `teacher_label_smoothing`은 0.0, `teacher_checkpoint`는 null로 명시한다. KD config의 teacher LS는 실제 teacher와 같게 명시한다(기존 teacher 0.1). teacher config의 student LS와 α는 0.1과 0.5로 유지하되 teacher 손실에는 사용하지 않는다. 이렇게 명시해도 되는지 확인해 달라.

### Q5. 기존 자동 진단과 필수 필드 전환에 필요한 최소 변경

기존 `Probe`는 CE/Full KD에서도 masked teacher 예측과 여러 마스킹 방식의 진단을 수행한다. 제안: Stage 1에서는 이 probe를 실행하지 않고 train/val만 평가한다. 또한 필수 teacher LS 때문에 기존 테스트와 기존 실행용 config의 입력에 `teacher_label_smoothing=0.1`을 명시해야 한다. 기존 테스트의 assertion은 유지하고 fixture/호출부 및 실행용 config만 최소 수정해도 되는지 확인해 달라. 과거 보고서·결과 config는 수정하지 않는다.

### Q6. 동률과 병렬 실행

제안: seed 0 후보 차이가 반올림 전 수치로 완전히 같으면 T=1을 선택한다. student는 `--jobs 4`로 실행하고, 기존 병렬 방식대로 DataLoader worker 0 및 프로세스당 thread 2를 사용한다. teacher 두 개는 같은 병렬 실행 방식으로 실행한다. OOM이나 자원 부족 시 자동으로 설정을 낮추지 않고 멈춘다. 이 사전 규칙으로 진행할지 확인해 달라.

## 사용자 확정 답변 (2026-09-30)

- Q1: A. 새 저장소에 코드·설정·문서 전체 및 `reports/stage1/`를 올린다. 데이터·가중치는 제외한다.
- Q2: 로컬 구현·테스트 및 서버 실행 명령까지만 수행한다. A5000 GPU 실행은 사용자가 한다. 기존 서버와 같은 `uv.lock`(torch 2.5.1 cu124)을 유지한다.
- 서버 데이터: `/home/kebap/Desktop/workspace/34/aim-lab-test-2/data/coco_single`.
- 서버 기존 teacher: `/home/kebap/Desktop/workspace/34/aim-lab-test-2/outputs/experiment2/seed_0/teacher/best.pt`.
- 위 자산은 config의 절대경로로 읽기 전용 참조한다. 클론 밖에 있으며 클론 삭제 대상이 아니다.
- Q3: 제안 승인. KL 판정은 train 전체 1회 평균에만 적용하고 측정 seed는 항상 0이다. val은 보고만 하며 seed 1 teacher에도 같은 기준을 적용한다.
- Q4: 제안 승인. 감사표의 미사용 필드는 반드시 “미사용”으로 표시한다.
- Q5: Stage 1 Probe를 끈다. 기존 테스트 assertion을 유지하며 fixture/호출부·실행 config만 최소 수정한다. 과거 보고서는 수정하지 않는다.
- Q6: 동률이면 T=1. teacher 2개 병렬, student 5개는 `--jobs 5`로 동시에 실행한다. workers 0, threads 2. OOM이면 멈춘다.
- 추가: T_LS0/T_LS01은 별도 출력 폴더와 epoch 30 `last.pt`를 사용한다. R2만 기존 `best.pt`를 사용한다.
- 추가: `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_TOOL_DIR`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TMPDIR`를 모두 `setup.sh`에서 클론 내부 경로로 export한다. 셸 설정 파일은 수정하지 않는다.

## 후속 변경 지시 — tako-server 자산 부재 (2026-09-30)

- 위의 외부 데이터·teacher 경로 지정은 폐기한다. tako-server에는 실험 2 자산이 없다.
- `stage1-prepare`로 클론 내부 `data/coco_single`에 데이터를 새로 준비한다. `8ebf7af:coco_kd/prepare.py`는 변경하지 않으며 준비 후 기존 manifest SHA-256 기준을 유지한다. 불일치하면 중단한다.
- 과거 teacher의 고정 SHA-256 검사는 제거한다. `R2_full_old_T1`을 `R2_full_LS01best_T1`로 바꾸고 seed 0의 새 `T_LS01/best.pt`를 T=1로 사용한다.
- R2는 T_LS01 학습 이후 실행한다. teacher 출력 점검의 세 번째 행은 새 `T_LS01_best`로 대체한다. 기존 두 last teacher의 KL 판정과 best 행의 보고 전용 성격은 유지한다.
- R1은 동일하며 R2의 회귀 기준 58.91 ±2.0%p, 초기 checksum, 다른 판정·임계값·설정은 유지한다. 원문 프로토콜은 기록으로 보존한다.

## Stage 1b 사전 계획 — RRC / mixup (2026-10-03)

상태: 최신 Stage 1 결과 커밋 `c5a826d` 확인. 아래 두 항목 답변 전 구현 코드를 작성하지 않는다. 기존 요청의 사전 계획 제출·사용자 답변 후 구현 규칙을 따른다. 로컬 구현·테스트 및 서버 명령 작성 범위는 유지한다.

### 구현 계획 (12줄)

1. Stage 1 소스·config·테스트·프로토콜·보고서·outputs 및 uv.lock을 보존하고 Stage 1b 전용 파일을 추가한다.
2. Stage 1b config에서 train_augmentation을 기본값 없이 JSON 필수로 검증하고 flip/rrc/rrc_mix만 허용한다.
3. 고정 다섯 seed 0 config를 생성하고 Stage 1 student 레시피(100 epochs, workers 0, threads 2, Probe off)를 유지한다.
4. RRC 크기·비율·flip을 이미지와 foreground mask에 동일 적용하고 mask는 nearest, val/probe는 기존 resize로 유지한다.
5. 증강 RNG를 모델 RNG와 분리하여 동일 seed·epoch·샘플에 동일 crop, 동일 batch에 동일 mix 종류·lambda·짝·CutMix 박스를 사용한다.
6. rrc_mix는 timm batch 방식의 역순 짝, Beta(0.8,0.8)/Beta(1,1), 50% 선택, 적용 확률 1, CutMix 실제 면적 lambda 보정을 사용한다.
7. 이미 smoothing된 혼합 target에 soft-label CE를 적용하고 teacher/student에 동일 혼합 텐서를 전달하며 mixed train accuracy는 기록하지 않는다.
8. 기존 T_LS0/last.pt를 읽기만 하고 업로드된 SHA-256 및 manifest 해시를 실행 전·각 run에서 검증하고 기록한다.
9. 독립 runner로 다섯 student를 동시에 실행하고 OOM/학습 오류 시 중단하며 학습 완료 후 세 KD 조건을 각 증강 CE와 비교한다.
10. 증강 일치·run 간 재현성·soft CE·teacher 입력 일치·flip 동작 불변·판정 테스트와 기존 테스트를 수행한다.
11. reports/stage1b/에 기존 flip 결과와 새 결과 비교표, run별 PASS/FAIL, 새 다섯 val 곡선을 결과 JSON에서 생성한다.
12. 클론 내부 uv 격리를 유지하는 실행 안내와 결과 업로드 명령을 제공하며 다섯 run 종료 후 끝낸다(seed 1/test/자동 후속 실행 없음).

### B1. 기존 Stage 1 보존과 새 필수 config 범위

제안 A: 기존 Config와 setup.sh를 변경하지 않고 Stage 1b 전용 config 클래스·학습 모듈·`setup_stage1b.sh`를 추가한다. `train_augmentation`은 Stage 1b JSON에만 필수이며 Stage 1은 기존 JSON 그대로 동작한다. 기존 모델·optimizer·LR·평가 함수는 읽기 전용 재사용한다. 기존 lockfile에 timm이 없으므로 새 의존성 없이 timm batch Mixup 동작을 참조 구현하고 출처를 기록한다.

대안 B: 공통 config·setup.sh의 최소 확장을 허용한다. 이 경우 "기존 Stage 1 코드 수정 금지"의 예외 범위를 먼저 확정해야 한다.

### B2. RRC 이미지 보간

- A (제안): 기존 이미지 resize와 같은 bicubic + antialias=True. crop 정책만 바꾼다.
- B: torchvision RandomResizedCrop 기본인 bilinear + antialias=True.
- 두 경우 모두 foreground mask는 동일 crop·flip + nearest, val/probe는 기존 bicubic 전체 resize다.

### 확인한 고정 자산과 판정

- 데이터: `data/coco_single`; manifest SHA-256 `ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e`.
- Teacher: `outputs/stage1_ls_gate/seed_0/T_LS0/last.pt`; SHA-256 `6e31acca311a19b307a722008ef6c7f65259445ec04217f7d86f8486faf293d6` (Stage 1 업로드 기록에서 확인; 로컬 가중치는 없음).
- 출력: `outputs/stage1b_aug/`; 보고서: `reports/stage1b/`.
- 각 KD run은 같은 증강 CE 대비 val macro 91–100 평균 차이가 1.5pp 이상이면 PASS. CE 행은 기준이며 PASS/FAIL 대상이 아니다. Stage 1의 회귀 수치·STOP을 Stage 1b 중단 기준으로 재사용하지 않는다.
- Full KD이며 teacher는 재학습하지 않는다. RRC로 foreground가 잘려도 임의 crop 재추첨·필터링을 하지 않는다.
- timm 참조: https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py (구현 시 고정 revision으로 출처 기록).

### Stage 1b 사용자 확정 답변 (2026-10-03)

- B1: A 승인, 단 **config·런처·출력만 분리하고 학습 루프는 복제하지 않는다**. 기존 `coco_kd/train.py`에 증강 옵션을 추가하는 최소 변경은 승인되었다. Stage 1 전용 config·런처·보고서·데이터·가중치·프로토콜은 보존한다.
- `train_augmentation="flip"`에서 Stage 1 R1_ce와 초기 checksum 및 첫 2에포치 loss/validation이 일치하는 회귀 테스트를 수행한다. 원본 학습 코드는 기준 Git 커밋에서 읽어 검증하며 별도 학습 루프 구현을 유지하지 않는다.
- B2: A 승인. RRC 이미지는 bicubic + antialias=True, mask는 nearest다.
- 서버 실행 전에 원본·새 flip 경로를 각각 2에포치 비교한다. 검증 출력도 `outputs/stage1b_aug/`에만 쓰고 기존 Stage 1 실행을 재개하지 않는다.

### Stage 1b 구현 중 검증 중단 (2026-10-03)

- `bash setup_stage1b.sh test`: 70 passed 뒤 flip 회귀 테스트 1개 실패로 중단했다. 나머지 테스트는 아직 실행되지 않았다.
- 실패 지점은 새 경로가 아니라 Git의 `c5a826d:coco_kd/train.py` 원본 `_train` 초기 checksum 검사다. macOS CPU에서 원본이 생성한 값은 `1284dcfa406db55021620afb0d3bcbd923dc2597a3d122db018bf1633ae9a074`이며 서버 기준 `543d4714549b4a1318384fd102ebcfc4b65a0ccf73da68d63e1507d15e5a4e8a`와 달랐다. 두 경로의 첫 2에포치 비교까지 도달하지 않았다.
- 플랫폼별 초기화 차이가 의심되나 아직 확정하지 않았다. 결과·기준 해시·레시피는 수정하지 않았다. 구현 작업은 로컬 미커밋 상태이며 GPU 실행은 하지 않았다.
- 확인 요청: 로컬 검증은 동일 macOS 환경의 원본 Stage 1과 새 flip 경로 사이 checksum·첫 2에포치 loss/val 일치를 검사하고, GPU 서버 검증은 기존 `543d…` checksum 및 업로드된 Stage 1 R1_ce 첫 2에포치 수치와의 일치까지 계속 강제하는 방식으로 분리해도 되는가? 승인 후 원인을 추가 확인하고 검증 코드를 수정한다.

### 플랫폼별 검증 분리 승인 (2026-10-03)

- 로컬(Mac): 동일 Mac의 원본 Stage 1 코드와 새 flip 경로 사이 초기 checksum 및 첫 2에포치 train loss·val macro의 완전 일치를 검사한다. 플랫폼 기준 해시는 사용하지 않는다.
- 서버(A5000): 새 flip R1_ce 경로만 2에포치 실행한다. 초기 checksum은 기존 `543d4714549b4a1318384fd102ebcfc4b65a0ccf73da68d63e1507d15e5a4e8a`와 정확히 일치해야 한다. epoch 1·2의 train loss와 val macro는 보존된 Stage 1 history와 절대 오차 1e-6 이내여야 한다.
- 서버 게이트 실패 시 Stage 1b 다섯 run은 시작하지 않고 STOP을 기록한다. 기준 해시·학습 설정은 변경하지 않는다.
- CPU 동등성 기록(`scope=same_platform_cpu`)과 서버 게이트 기록(`scope=server_a5000`)을 별도 파일에 저장하며 CPU 통과 기록은 서버 실행 허가에 사용하지 않는다. 앞의 서버 원본·새 코드 각각 2에포치 실행 제안은 이 승인 내용으로 대체한다.
