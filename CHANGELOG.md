# Changelog

## 2026-09-30 — tako-server 자산 부재에 따른 준비·R2 변경

- 사유: 서버에 실험 2 데이터와 teacher가 없으며 과거 teacher 파일은 재현할 수 없다는 사용자 지시.
- `stage1-prepare`를 추가하고 `DATA` 및 모든 Stage 1 config의 `data_root`를 클론 내부 `data/coco_single`로 변경했다. uv 격리는 유지한다.
- `8ebf7af`의 `coco_kd/prepare.py`는 바이트 단위로 보존한다. 소스 확인 후 원래 함수를 기본 인자 그대로 호출하고, 준비 직후 manifest SHA-256 `ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e`를 검증한다. 불일치하면 `STOP_PREPARE`로 종료한다.
- 과거 teacher의 `4a18ddc1…` 고정 해시 검사를 제거했다. `R2_full_old_T1`을 `R2_full_LS01best_T1`로 바꾸고 새 seed 0 `T_LS01/best.pt`와 T=1을 사용한다. 실제 teacher SHA 기록·변경 검사는 유지한다.
- 출력 점검의 `experiment2_best` 행을 `T_LS01_best`로 대체했다. last 두 개의 KL 게이트는 유지하고 best 행은 보고만 한다. R2를 포함한 student는 teacher 학습·점검 후 시작한다.
- R1은 변경하지 않는다. R2의 역사적 회귀 기준 58.91 ±2.0%p, 초기 checksum, 나머지 설정·판정·seed 0 종료 옵션은 유지한다. R2가 원래 teacher와 동일한 파일 재현은 아님을 감사표·보고서에 명시한다.
- 준비 해시 성공·실패, 준비 소스 변경 거부, best/last 경로·에포치 검증, 학습 순서 테스트를 갱신했다. 원문 프로토콜과 과거 보고서는 변경하지 않는다.

## 2026-09-30 — Stage 1 seed 0 종료 옵션

- `bash setup.sh stage1-run --jobs 5 --stop-after-seed0` 추가.
- seed 0 판정과 SUMMARY 생성 후 `RUN_SEED1`이면 `SEED0_DONE: RUN_SEED1, selected=T1` 또는 `selected=T4`를 출력하고 정상 종료한다. seed 1과 `STOP.json`은 생성하지 않는다.
- `STOP_*` 판정의 STOP 기록·실패 종료 동작은 유지한다.
- 같은 소스·config에서 옵션 없이 재실행하면 완료한 seed 0 run을 재사용하고 확인 절차를 거쳐 seed 1으로 진행한다.
- CLI 정상 종료·실패 판정·T1/T4 선택·후속 실행·완료 run 무재학습 테스트를 추가했다. 판정 규칙·임계값·config는 변경하지 않았다.
