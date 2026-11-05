0. 역할과 원칙
너는 이 저장소의 구현 파트너다. 프로젝트는 경희대 컴퓨터공학부 졸업프로젝트(지도교수: 허선영, 컴파일러 & 시스템 소프트웨어 연구실)이고, 주제는 **"프로그래밍 언어 선택과 언어 특성이 LLM 생성 HPC 코드의 정확성·성능·안전성에 미치는 영향 — 실증 분석(예비 연구)"**이다.
원칙:

1. 범위를 늘리지 않는다. 지도교수 지시: 간단한 태스크부터 결과를 확보한다. 새 언어 설계·파인튜닝·RL·보드 포팅은 하지 않는다.
2. 끝까지 돌아가는 수직 슬라이스 먼저. 마일스톤 1(GEMM × NumPy·C × 3샘플)이 CSV 한 줄을 뱉을 때까지 다른 언어·과제를 만들지 않는다.
3. 측정 규약은 코드로 강제한다. 사람이 기억해서 지키는 규약은 없는 것과 같다.
4. 모든 결과는 재현 가능해야 한다. 프롬프트·모델·온도·시드·컴파일 플래그·환경 정보를 결과 행마다 기록한다.
5. 결정이 필요하면 `docs/decisions.md`에 날짜와 함께 적고 진행한다. 되돌릴 수 없는 결정(비용이 드는 대량 API 호출, 측정 규약 변경)은 먼저 묻는다.

1. 연구 설계 요약 (구현이 지켜야 할 것)
핵심 가설. LLM 코드의 성능 격차는 추상화 수준이 아니라 "성능 결정(루프 순서·타일링·벡터화·병렬화·메모리 배치)을 LLM이 직접 내려야 하는가, 컴파일러·라이브러리에 넘길 수 있는가"로 설명된다.
독립 변수.

* 언어/조건 (성능 결정 부담 순): `numpy` → `numba`(타입 전부 명시, @njit) → `halide_auto`(알고리즘만, 오토스케줄러) → `halide_manual`(알고리즘+LLM 스케줄) → `c_openmp` → `c_seq`(-O3) → `rust`(+rayon, 후순위) → `opencl`/`c_neon`(후순위) → 확장 `llvm_ir`, `aarch64_asm`.
* 마이크로 조건 (같은 하네스에 플래그로): `args=positional|keyword`, `grammar_hint=on|off`(저자원 언어용 문법 안내 프롬프트), `feedback=compiler|compiler+lint`(수정 루프에 린트 메시지 포함 여부).

종속 변수 (결과 행의 열).

* 정확성: `pass`(허용오차 내 일치), `turns_to_pass`(0~3, 실패 시 NA), `error_type`(아래 분류표), `pass_at_1`, `pass_at_5`는 분석 단계에서 집계.
* 성능: `time_median_ms`, `speedup_vs_baseline`, `time_cv`(샘플 내 반복 변동), 선택 `roofline_pct`.
* 안전성: `asan_findings`, `ubsan_findings`, `lint_findings_total`, 범주별 린트 수.
* 비용: `tokens_in`, `tokens_out`, `tokens_per_pass`(분석 단계).
* 성능 결정 요약: `perf_summary`(JSON 문자열, 아래 §5).

오류 분류표 (실험 전에 고정, 변경 금지). `compile_error` · `runtime_crash` · `timeout` · `wrong_output` · `numeric_instability`(NaN/Inf 또는 허용오차 초과이면서 패턴이 수치적) · `arg_order`(인자 순서·차원 뒤바뀜으로 판정되는 wrong_output: 전치/차원 교환 시 일치) · `memory_error`(ASan 탐지) · `race_condition`(반복 실행 간 결과 불일치 또는 TSan 탐지) · `signature_mismatch`(요구한 함수 시그니처를 지키지 않음).
측정 규약. 입력은 메모리 상주(파일 I/O 없음) · `OMP_NUM_THREADS`와 `taskset` 코어 집합 고정(설정 파일) · CPU 거버너 `performance` 확인(아니면 경고하고 결과에 플래그) · 워밍업 3회 후 10회 반복의 중앙값 · 컴파일 플래그 통일(`-O3 -march=native` 또는 설정값) · 샘플 분포 전체 기록(최선값만 보고하지 않음) · 실행 전 시스템 부하 체크(load average 기록).
모델. 기본 프론티어 API 모델 1종(환경변수로 지정), temperature 고정(기본 0.2), 과제당 `n=5` 샘플, 수정 루프 최대 3턴. 오픈 모델은 2차(OpenAI 호환 엔드포인트로 교체 가능하게).
2. 저장소 구조

```
llm-lang-hpc/
  CLAUDE.md                 # 이 문서
  config/
    experiment.yaml         # 모델·온도·n·언어·조건·과제·스레드·코어·플래그
    machine.yaml            # CPU 모델, 코어 수, 대역폭 실측(roofline용), 거버너
  tasks/
    gemm/                   # 과제 1
      spec.md               # 자연어 명세(한/영), 수학 정의, 입출력 계약, 크기
      signatures.yaml       # 언어별 함수 시그니처 (positional / keyword 두 버전)
      reference.py          # NumPy 참조 구현 (정답)
      gen_tests.py          # 시드 고정 테스트 입력 생성 → tests/*.npz (small, large)
      baseline/             # 성능 기준선: 라이브러리(OpenBLAS) 또는 손 구현 C+OpenMP
      tolerance.yaml        # 허용오차 (rtol/atol, dtype별)
    stencil2d/              # 과제 2 (2D Jacobi, 5-point)
    spmv_csr/               # 과제 3
  prompts/
    system.md               # 공통 시스템 프롬프트
    lang/<language>.md      # 언어별 지시(시그니처 삽입 자리, 금지 사항)
    hints/<language>.md     # grammar_hint=on 일 때 붙는 짧은 문법 안내(≤100줄)
    repair.md               # 수정 루프 프롬프트(오류 메시지 삽입 자리)
  harness/
    llm_client.py           # Anthropic SDK 기본, OpenAI 호환 폴백; 토큰 수·지연 기록
    generate.py             # 과제×언어×조건×n 생성, 원문 저장(results/raw/…)
    extract.py              # 응답에서 코드 블록 추출, 시그니처 검증
    build/<language>.py     # 빌드: 플래그, 타임아웃, 진단·리마크 수집
    run.py                  # 실행·정확성 검사·시간 측정 (ctypes 또는 서브프로세스)
    repair.py               # 수정 루프 (compiler | compiler+lint 피드백)
    safety.py               # ASan/UBSan 재빌드 실행, 린트 플러그인 호출
    perf_summary.py         # 성능 결정 요약 추출 (§5)
    protocol.py             # 측정 규약 강제: 거버너·코어·스레드·부하 체크
    schema.py               # 결과 행 스키마(dataclass) + CSV/Parquet 쓰기
  plugins/
    lint/                   # 인지프 린트 플러그인 자리 (§6 인터페이스)
  analysis/
    aggregate.py            # pass@k, speedup, 분포, 부트스트랩 CI
    plots.py                # 산점도(정확성 vs 속도비), 오류 분류 막대, 언어 축 그래프
  dashboard/                # 후순위: 생성→실행→요약 라이브 화면
  results/                  # raw/, built/, runs.csv (git에는 csv만)
  docs/
    decisions.md            # 결정 기록
    protocol.md             # 측정 규약 상세
    error_taxonomy.md       # 오류 분류표
```

3. 마일스톤 (순서대로, 각 단계 끝에 결과를 보여주고 멈춘다)
M1 — 수직 슬라이스 (최우선). GEMM 하나, 언어 `numpy`·`c_seq`·`c_openmp`, 모델 1개, `n=3`. 생성→추출→빌드→실행→정확성→시간→CSV까지. 결과 `results/runs.csv`에 행이 쌓이고 `analysis/aggregate.py`가 pass율·중앙값 속도비를 출력하면 완료. 이 단계에서 측정 규약(`protocol.py`)과 오류 분류를 함께 완성한다.
M2 — 과제 확장(2026-11-05 갱신). 커널 12개를 3단계로 넣는다(§4 표). M2a = Tier 1(`gemm`, `jacobi2d`, `spmv_csr`, `dot`), M2b = Tier 2, M2c = Tier 3. 각 과제에 참조·테스트·기준선·허용오차. `n=5`. 앞 단계 결과를 보고 다음 단계를 넣는다(원칙 1).
M3 — 언어 확장. `numba`(타입 명시 강제: 프롬프트에 서명 타입 요구, 빌드 단계에서 `@njit` 시그니처 검사), `halide_auto`/`halide_manual`(pip `halide`; 오토스케줄러 플러그인 로드 확인; 실패 시 `docs/decisions.md`에 기록하고 CPU 타깃만). `rust`는 툴체인 확인 후.
M4 — 마이크로 조건. `args=keyword`(Python/Numba 키워드 전용 시그니처, C는 매개변수 구조체), `grammar_hint`, `feedback=compiler+lint`(§6 플러그인 연결), 수정 루프.
M5 — 성능 결정 요약 + 안전성. `perf_summary.py`, `safety.py`(ASan/UBSan), 린트 플러그인 호출. 결과 열 채움.
M6 — 분석·그래프·대시보드. 산점도(x: 언어를 성능 결정 부담 순으로, y: pass@1과 속도비), 오류 분류 막대, 조건별 효과(대응 비교 + 부트스트랩 CI). 대시보드는 마지막.
M7 — 확장(시간 남을 때만). `llvm_ir`/`aarch64_asm` 직접 생성 vs C `-O0`/`-O3`. 보드(RUBIK Pi 3)에서 같은 하네스 재실행.
4. 과제 명세 (M1·M2)
공통: float32, 행우선(row-major), 입력 버퍼는 읽기 전용, 출력 버퍼는 호출자가 할당. 크기는 `small`(정확성, 수 ms)·`large`(성능, 수백 ms~수 s) 두 가지. 허용오차는 과제별 `tolerance.yaml`(GEMM: rtol 1e-3 — 누적 순서 차이 허용; 스텐실: 1e-5; SpMV: 1e-4).

* gemm: `C[M×N] = A[M×K] · B[K×N]`. small 256³, large 1024³ (머신에 맞춰 조정). 기준선: NumPy(OpenBLAS) `A @ B`. C 시그니처 `void gemm(const float* A, const float* B, float* C, int M, int N, int K)`; 키워드 버전은 Python에서 `gemm(*, A, B, M, N, K)`.
* stencil2d: 5-point Jacobi 1회 반복, 경계는 그대로 복사. `out[y][x] = 0.25*(in[y-1][x]+in[y+1][x]+in[y][x-1]+in[y][x+1])`. small 512², large 4096². 기준선: 손 구현 C+OpenMP(`baseline/`에 포함, 벡터화 확인).
* spmv_csr: `y = A·x`, A는 CSR(`row_ptr`, `col_idx`, `vals`). small 10⁴행/밀도 1e-3, large 10⁶행. 기준선: `scipy.sparse` 또는 손 구현 C+OpenMP.

커널 세트(2026-11-05 확정, 교수님 피드백 반영). 선정 기준: 하드웨어 고정 · 모든 언어에서 구현 가능 · 명세만으로 생성 가능한 크기(함수 1~3개) · 정답 자동 검증 · 성능 특성 다양 · 난이도 사다리.

| Tier | 커널 | 병목 | LLM이 내려야 하는 성능 결정 |
|---|---|---|---|
| 1 | `gemm` (1024³) | compute | 루프 순서·타일링·벡터화 |
| 1 | `jacobi2d` (4096², T=20) | memory | 배열 교환, 경계 처리, 병렬 for |
| 1 | `spmv_csr` (10⁶행) | 불규칙 memory | 행 분할, 간접 참조 |
| 1 | `dot` (2²⁵) | bandwidth | reduction 병렬화, 누적 정밀도 |
| 2 | histogram · prefix scan · n-body O(n²) · sort | atomics / 알고리즘 재구성 / compute / 알고리즘 선택 | private histogram, 2-pass 스캔, SoA·rsqrt, radix/merge |
| 3 | LU 또는 Cholesky · Floyd–Warshall · BFS · FFT | 의존성·불규칙 | 병렬 가능한 루프 판별, 프론티어 병렬화, bit-reversal |

Tier 1의 구체 계약(시그니처·크기·허용오차)은 `tasks/<task>/`와 `docs/decisions.md`가 기준이다. 위 `stencil2d` 항목은 `jacobi2d`(T 스텝)로 대체됐다. Tier 2·3의 정수 출력(histogram, sort, BFS)과 Halide로 표현하기 어려운 커널의 처리는 해당 단계 시작 때 정한다.

각 `spec.md`는 LLM에 그대로 들어가는 문장이므로 모호함 없이 쓰고, 언어 중립적으로 쓴다(특정 라이브러리 사용 지시 금지 — 라이브러리 사용 여부 자체가 관찰 대상).
5. 성능 결정 요약 (`perf_summary.py`) — 출력 스키마
생성 커널 하나당 JSON 하나. 사람이 "이 코드가 성능 결정을 어떻게 내렸나"를 10초 안에 보는 용도이자, 가설 검증(성능 결정을 누가 내렸나)의 데이터.
json

```json
{
  "language": "c_openmp",
  "loops": [{"depth": 3, "order": ["i","j","k"], "bounds": ["M","N","K"]}],
  "parallel": {"kind": "openmp", "regions": 1, "collapse": false, "reduction": false},
  "vectorized": {"status": "yes|no|partial|unknown", "source": "compiler_remark", "loops": ["j"]},
  "tiling": {"detected": false, "tile_sizes": []},
  "memory": {"temporaries": 0, "layout_transposed": false},
  "library_calls": ["cblas_sgemm"],
  "decisions_by_llm": ["loop_order", "parallel_region"],
  "decisions_by_compiler": ["vectorization"],
  "notes": "컴파일러 리마크: loop not vectorized: unsafe dependent memory operations"
}
```

추출 소스: clang `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize`(C/Rust), OpenMP pragma 파싱, Python AST(배열 원소 for-루프 탐지, NumPy 호출 목록), Numba `inspect_types`/`parallel_diagnostics`, Halide `print_loop_nest()`·스케줄 문자열. 휴리스틱으로 못 정하면 `"unknown"`으로 두고 거짓 확신을 만들지 않는다.
6. 안전성 요약 — 린트 플러그인 인터페이스 (인지프 과목 결과물이 꽂히는 자리)
하네스는 린트의 내부를 모른다. 계약만 지키면 어떤 구현이든 된다.
호출: `plugins/lint/run_lint.py --language <lang> --file <path> --task <task> --signature <json> --out <json>`
출력 JSON:
json

```json
{
  "tool": "mllint-v0",
  "findings": [
    {"rule": "numeric/accumulate-in-float32", "severity": "warning", "line": 12,
     "message": "float32 누적으로 큰 K에서 오차 증가", "category": "numeric"}
  ],
  "summary": {"memory": 0, "numeric": 1, "concurrency": 0, "shape_dtype": 0, "api_misuse": 0, "performance": 2}
}
```

범주 정의(변경 시 `docs/error_taxonomy.md` 갱신): `memory`(경계·초기화·별칭), `numeric`(NaN/Inf 위험, 정수 오버플로, 누적 정밀도, 부동소수 비교), `concurrency`(병렬 영역 내 공유 쓰기, 리덕션 누락, 비결정성), `shape_dtype`(차원·브로드캐스트·dtype 암묵 변환), `api_misuse`(시그니처 위반, 입력 변형, 전역 상태), `performance`(배열 원소 Python 루프, 불필요한 복사, 비연속 접근 — 안전성은 아니지만 같은 도구로 뽑는 성능 냄새).
린트 타임라인(2026-11-05): M1·M2는 플러그인 없이 진행(빈 결과). M3·M4에서 `feedback=compiler+lint` 조건을 만들 때 기성 린터(clang-tidy/cppcheck, ruff/pyflakes)를 이 계약으로 감싼 베이스라인을 먼저 쓴다. 인지프 ML lint는 필요해지는 시점에 구현해 같은 CLI로 교체한다(마감 연동 없음). 하네스가 쌓는 실패 커널과 오류 라벨은 린트의 학습·평가 데이터로 내보낸다.
하네스에서의 쓰임: (1) `lint_findings_*` 열, (2) `feedback=compiler+lint` 조건에서 수정 프롬프트에 `findings[].message`를 넣는다. 플러그인이 없으면 빈 결과로 진행하고 경고만 남긴다(실험이 막히면 안 된다).
7. 시작할 때 사용자에게 확인할 것 (한 번에 물어라)

1. LLM 제공자와 키 환경변수 이름(기본 `ANTHROPIC_API_KEY`), 사용할 모델명.
2. 실험 머신 정보: CPU, 코어 수, 메모리 — `config/machine.yaml` 채움. `cpupower`/`taskset` 사용 가능 여부(sudo).
3. M1 언어 세 개(`numpy`, `c_seq`, `c_openmp`)로 시작해도 되는지.
4. 저장소 이름과 원격(GitHub) 여부.

8. 첫 작업 지시
위 구조대로 저장소를 만들고 M1만 구현한다. 순서: `config/` → `tasks/gemm/` → `harness/protocol.py` → `harness/llm_client.py`(키 없으면 더미 응답 모드로 파이프라인 검증) → `generate.py`/`extract.py` → `build/c.py`, `build/python.py` → `run.py` → `schema.py` → `analysis/aggregate.py`. 더미 모드로 전체 파이프라인이 CSV를 만들면 멈추고 보여준 뒤, 실제 API 호출(3샘플 × 3언어 = 9회)은 확인을 받고 실행한다.
