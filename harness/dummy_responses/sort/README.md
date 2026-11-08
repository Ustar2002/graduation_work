# sort 더미 응답 (docs/harness_design.md §17)

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다. `<language>/<k>.md` 가
`sample_idx mod 파일 수` 번째 샘플의 응답이 된다(세 언어 모두 정답 1개씩). 각 파일은 짧은 설명 + 펜스 코드 블록 하나다.

| language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|
| numpy | 0 | `out[...] = np.sort(x)` — 기준선 `tasks/sort/baseline/baseline.py` 와 같은 코드(단일 스레드) | 없음 — bench 까지 정상, `speedup_vs_baseline` ≈ 1 | 1 |
| c_seq | 0 | `memcpy(out, x)` 뒤 `qsort`(비교자 `(x > y) - (x < y)` — `a - b` 오버플로 없음) | 없음 (large 10⁷ 에 호출당 약 1.3 s, 1스레드) | 1 |
| c_openmp | 0 | 8비트 × 4패스 병렬 LSD 기수 정렬: 스레드별 연속 청크의 digit 히스토그램 → `single` 에서 bucket·thread 순 쓰기 오프셋 → 안정 분산; 키는 부호 비트를 뒤집어 unsigned 순서 = signed 순서, 마지막 패스에서 되돌려 `out` 에 착지 | 없음, `speedup_vs_baseline` > 1 (병렬 정렬이 단일 스레드 `np.sort` 보다 빠른 것은 정상) | 1 |

## 변형별 주의

- **정확 일치**: `tolerance.yaml` 은 rtol = atol = 0(정수 출력, 설계 §17). 오름차순 배열은 유일하므로 안정성·알고리즘과 무관하게 모든 정답이 같은 출력을 낸다.
  틀린 원소의 배율은 inf 라 `numeric_instability` 는 나올 수 없고(`wrong_output/mismatch` 만), 센티널 `-2048` 은 음이 아닌 입력의 정렬 결과가 될 수 없어
  `output_not_written`·`partial_write`·`accumulated_into_c` 판정이 그대로 작동한다.
- **입력 제자리 정렬은 `wrong_output/inputs_mutated`**: 워커가 호출 뒤 `x` 를 원본과 비교한다(`tests/test_tier2_sort.py` 의 심은 버그 `x.sort(); out[...] = x`).
  내림차순은 `wrong_output/mismatch`, 마지막 원소 미기록은 `wrong_output/partial_write:1/7`(tiny).
- **중복값**: 입력은 `[0, 2³¹−1]` 균등이라 생일 문제대로 rect(100,003)에서 수 개, small(2²⁰)에서 수백 개, large(10⁷)에서 약 2만 3천 개의 중복이 있다(실측 2026-11-08).
  c_openmp 더미는 단위 테스트에서 스레드 4·3·1, 중복이 많은 입력(값 10가지 × 1000개), n = 1, 극값(0·INT32_MAX)만 있는 입력으로도 pass 를 확인했다.
- **기준선이 python(`np.sort`)**: `gate_blas: false` 라 BLAS 정체 게이트 없이 완료·`final_check` pass 만 본다. 실측(2026-11-08, large 10⁷, worker bench 중앙값):
  기준선 1스레드(core 2) 58.2 ms(CV 1.6 %), 4스레드 설정 57.7 ms(정렬은 단일 스레드라 같음); 더미 c_openmp/0 37.9 ms(4스레드); 더미 c_seq/0(`qsort`) 1,269 ms.
- C 변형은 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o kernel.so kernel.c -lm -Wl,--no-undefined`
  로 경고 0 으로 컴파일된다(c_openmp/0 은 `omp.h` 를 쓰므로 `-fopenmp` 필수 — c_seq 로 넣으면 `omp_get_num_threads` 미정의로 `compile_error` 가 된다).

## 실측 (2026-11-08, `dummy-tier2`, large n = 10⁷, warmup 3 + timed 10 의 중앙값, governor `performance`)

| 커널 | 스레드·코어 | median ms (샘플 0 / 1) | cv | speedup_vs_baseline (대응) |
|---|---|---|---|---|
| 기준선 `np.sort` run 시작 | 4T 설정(단일 스레드 구현), cpu 0,2,4,6 | 63.3 (대응 기준선 64–90) | 0.022 | — |
| 기준선 1스레드 | 1T, cpu 2 | 70.5 | 0.035 | — |
| numpy/0 (= 기준선 코드) | 4T 설정 | 68.1 / 78.9 | 0.016 / 0.153 | 1.03 / 0.83 |
| c_seq/0 (`qsort`) | 1T, cpu 2 | 1,406 / 1,400 | 0.133 / 0.039 | 0.05 / 0.05 |
| c_openmp/0 (병렬 LSD 기수 정렬) | 4T | 99.5 / 46.0 | 0.028 / 0.067 | 0.90 / 1.48 |

모두 check(tiny·rect·small)·bench `final_check` pass, 실행 간 불일치 0. 샘플 0 의 c_openmp 행은 대응 기준선도 89.75 ms(`baseline_drift=+30.1%` 플래그)로 느린 구간이었다
(통합 세션의 동시 작업); 샘플 1 의 46 ms·속도비 1.48 이 위 구현자 측정(37.9 ms)에 가깝다. 4T 설정 행의 `cpu_mhz_median` 이 1.6 GHz 로 낮은 것은 `np.sort` 가 단일 스레드라
고정 코어 4개 중 3개가 유휴 주파수로 샘플되기 때문이다(정보용 열).
