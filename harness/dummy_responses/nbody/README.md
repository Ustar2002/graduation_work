# nbody 더미 응답 (docs/harness_design.md §17)

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다. `<language>/<k>.md` 가
`sample_idx mod 파일 수` 번째 샘플의 응답이 된다(세 언어 모두 정답 1개씩). 각 파일은 짧은 설명 + 펜스 코드 블록 하나다.

| language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|
| numpy | 0 | float64 로 변환해 256행씩 `(chunk, n, 3)` 브로드캐스트로 모든 쌍을 직접 합산, 블록마다 `acc[i0:i1] = …` (참조 구현 `tasks/nbody/gen_tests.py` 와 같은 식) | 없음 — bench 까지 정상 (max_tol_ratio ≈ float32 반올림 수준) | 1 |
| c_seq | 0 | i·j 2중 루프, `r2 = dx²+dy²+dz²+1e-4f`, `inv_r = 1.0f/sqrtf(r2)`, `w = mass[j]·inv_r³`, AoS 그대로 | 없음 | 1 |
| c_openmp | 0 | 위 c_seq 커널에 `#pragma omp parallel for schedule(static)` (i 병렬) — 기준선 `tasks/nbody/baseline/baseline.c` 와 같은 코드 | 없음, `speedup_vs_baseline` ≈ 1 | 1 |

## 변형별 주의

- **허용오차(`tolerance.yaml`)**: `rtol = 1e-3`, `atol_rms_rel = 1e-3`(테스트마다 atol = 1e-3 × 그 형태 참조 가속도 성분 RMS; large 18.1, tiny 0.0032 — 고정 atol 20 은 tiny 를 공허하게 만들어 2026-11-08 에 바꿈), `numeric_pattern_tol_ratio = 20`, **센티널 `-1e6`**(정당 범위 밖이면서 float32 ulp 가 atol 보다 훨씬 작아 누적 판정이 성립; 처음의 `-1e30` 은 덧셈을 흡수했다).
  참조는 float64 로 모든 쌍을 직접 합산한 값(근사 없음; j = i 항은 정확히 0).
- **직진 float32 커널의 계약(§17 (a))**: `1.0f/sqrtf`·AoS·순차 누적 커널의 `max_tol_ratio` 실측(2026-11-08) tiny 2.8e-8 / rect 4.9e-5 / small 4.3e-4 / large 4.9e-3 — 모두 0.2 이하.
  **보정 없는 `_mm256_rsqrt_ps`(§17 (b))**: rect 0.069 / small 0.13 / large 0.17 로 여전히 pass(허용오차의 약 1/6); 뉴턴 1회 보정은 large 6.4e-4.
  즉 모델이 `rsqrt` 근사를 써도 정확성에서 떨어지지 않으며, 허용오차를 넘기되 배율 20 이내면 `numeric_instability`/`imprecise` 다.
- **SoA 출력은 `arg_order`/`transposed_output`**(의도된 분류): `acc[c*n + i]` 로 쓰면 2-D 출력 (n, 3) 의 공통 전치 패턴 `out.reshape(3, n).T ≈ ref` 에 걸린다.
  `tests/test_tier2_nbody.py` 가 python·C 커널로 rect(n=301)에서 단언한다(더미에는 이 변형이 없다).
- **j = i 항**: 분자 0·분모 EPS2^(3/2) > 0 이라 정확히 0 — 포함해도 건너뛰어도 같다(참조는 포함).
- C 변형은 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o kernel.so kernel.c -lm -Wl,--no-undefined`
  로 경고 0 으로 컴파일된다. j 루프는 float 누적의 재결합이 필요해 gcc 가 벡터화하지 않는다(`-ffast-math` 없음; build.log 의 `-fopt-info-vec-missed` 참고).

## 실측 (2026-11-08, `dummy-tier2`, large n = 16,384, warmup 3 + timed 10 의 중앙값, governor `performance`)

| 커널 | 스레드·코어 | median ms (샘플 0 / 1) | cv | GFLOPS (20·n²) | max_tol_ratio (large) |
|---|---|---|---|---|---|
| 기준선 `tasks/nbody/baseline/baseline.c` run 시작 | 4T, cpu 0,2,4,6 | 229.2 (대응 기준선 213–232) | 0.047 | 23.4 | 4.9e-3 |
| 기준선 1스레드 | 1T, cpu 2 | 643.7 | 0.034 | 8.3 | 4.9e-3 |
| c_openmp/0 (= 기준선 코드) | 4T | 213.7 / 229.2 | 0.011 / 0.025 | 25.1 | 4.9e-3 |
| c_seq/0 | 1T, cpu 2 | 608.7 / 606.8 | 0.036 / 0.037 | 8.8 | 4.9e-3 |
| numpy/0 (float64 브로드캐스트) | 4T 환경 | 5,822 / 6,167 | 0.013 / 0.059 | 0.92 | 3.7e-5 |

check(tiny·rect·small)의 max_tol_ratio: C 커널 2.76e-8 / 4.87e-5 / 4.34e-4, numpy 1.1e-8 / 2.7e-6 / 1.0e-5 — 모두 pass, 실행 간 불일치 0(비트 불일치도 0).
속도비(대응 기준선 대비): c_openmp 1.00–1.01, c_seq 0.35–0.38, numpy 0.04. numpy 의 호출당 5.8–6.2 s 는 13회 호출 76–80 s 로 bench 예산(300 s × 0.8) 안이라
반복 축소가 없었다; 순수 Python 2중 루프(n² = 2.7×10⁸ 반복)는 첫 호출이 예산을 넘겨 `bench / timeout`(pass=1, 시간 NA)로 기록될 것이다(설계 §8 적응 규칙).
