# Changelog

## 2026-09-30 — Stage 1 seed 0 종료 옵션

- `bash setup.sh stage1-run --jobs 5 --stop-after-seed0` 추가.
- seed 0 판정과 SUMMARY 생성 후 `RUN_SEED1`이면 `SEED0_DONE: RUN_SEED1, selected=T1` 또는 `selected=T4`를 출력하고 정상 종료한다. seed 1과 `STOP.json`은 생성하지 않는다.
- `STOP_*` 판정의 STOP 기록·실패 종료 동작은 유지한다.
- 같은 소스·config에서 옵션 없이 재실행하면 완료한 seed 0 run을 재사용하고 확인 절차를 거쳐 seed 1으로 진행한다.
- CLI 정상 종료·실패 판정·T1/T4 선택·후속 실행·완료 run 무재학습 테스트를 추가했다. 판정 규칙·임계값·config는 변경하지 않았다.
