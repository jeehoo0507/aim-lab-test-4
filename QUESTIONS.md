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
