# LLM 생성 HPC 코드 평가 하네스 — 빌드 글루 (docs/harness_design.md §14, docs/harness_design_v3.md)
#
#   make venv                      python3 -m venv .venv && pip install -r requirements.txt
#   make rust-setup                Rust 템플릿의 고정 크레이트(rayon)를 내려받는다(온라인 1회; 이후 rust 빌드는 --offline)
#   make tests                     tasks/*/gen_tests.py (과제 12개의 시드 고정 테스트 입력 npz + manifest.json; 전체 약 1.6 GB)
#   make probe                     harness.protocol --probe --write  (config/machines/<호스트>.yaml 생성·갱신)
#   make unittest                  .venv/bin/python -m unittest discover -s tests
#   make devcheck                  tools/devcheck.py dummies (모든 0.md 더미를 워커 check 로 점검; 측정 아님)
#   make devcheck-baselines        tools/devcheck.py baseline --task <과제> (과제마다)
#   make dummy [RUN_ID=…]          generate --provider dummy → run → aggregate --csv results/runs_dummy.csv
#   make dummy-all / dummy-m3      언어 일곱 개 / 새 언어 네 개의 더미 파이프라인 (n=1, 과제 전부) + --coverage
#   make repair-dummy              gemm 더미로 수정 루프와 피드백 조건 대응 비교 → aggregate --by-turn --conditions
#   make real RUN_ID=… [STRICT=1]  generate(확인 프롬프트) → run [--strict-protocol] → aggregate
#   make aggregate [RUN_ID=…] [CSV=…]   RUN_ID 가 없으면 --latest
#   make plots [RUN_ID=…] [CSV=…]       analysis.plots (기본 --latest, results/plots/)
#   make aa [TASKS=a,b]            A/A 점검: 기준선 대 기준선 교차 측정(측정 장치 치우침, fsync 유무 비교)
#   make unpack-raw                1차 실험(노트북) 응답 원문을 results/raw 로 푼다(새 장비에서 같은 코드를 다시 잴 때)
#   make report                    중간보고서 PDF (report/interim/build.py; 먼저 `.venv/bin/pip install -r report/requirements.txt`)
#   CONFIG=path                    모든 harness 명령에 --config path (예: 새 장비용 설정)
#
# 모든 python 실행은 .venv/bin/python 이다(시스템 python3 는 `make venv` 에서 venv 를 만들 때만).
# RUN_ID 기본값은 타임스탬프(dummy-YYYYmmdd-HHMMSS / YYYYmmdd-HHMMSS-real 등)이며 한 타깃 안의 generate·run·aggregate 가 같은 값을 쓴다.
# 선택 필터(비우면 각 CLI 의 기본값 = config/experiment.yaml): TASKS=a,b  LANGUAGES=x,y  N=<샘플 수>  SKIP_BENCH=1

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
# venv 를 만들 시스템 인터프리터 (3.14 권장: `make venv SYSPY=python3.14`)
SYSPY  ?= python3
# rustup 은 PATH 를 바꾸지 않고 설치했을 수 있다(docs/desktop_setup.md): PATH 에 없으면 CARGO=~/.cargo/bin/cargo 로 준다.
# (하네스의 빌드는 PATH → $CARGO_HOME/bin → ~/.cargo/bin 순으로 스스로 찾는다.)
CARGO  ?= cargo

TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
COMMA := ,

# STRICT=1 이면 harness.run 에 --strict-protocol (run 시작의 정적 규약 플래그 — 거버너·EPP·taskset·BLAS — 가 있으면 중단, exit 4;
# bench 직전 동적 플래그·baseline_drift 는 strict 여도 행에 기록만 한다)
STRICT ?= 0
# CONFIG=path 이면 harness 명령들에 --config path (새 장비용 설정 등)
CONFIG ?=
CFG_FLAG := $(if $(CONFIG),--config $(CONFIG),)
STRICT_FLAG := $(if $(filter 1 yes true,$(STRICT)),--strict-protocol,)

# SKIP_BENCH=1 이면 harness.run·harness.repair 에 --skip-bench (check 까지만; bench 생략)
SKIP_BENCH ?= 0
SKIP_BENCH_FLAG := $(if $(filter 1 yes true,$(SKIP_BENCH)),--skip-bench,)

# generate 필터. 셋 다 비면 generate 명령은 예전과 같다(run.tasks·run.languages·run.n_samples). TASKS 는 aa·devcheck 도 쓴다.
TASKS     ?=
LANGUAGES ?=
N         ?=
GEN_FILTER := $(if $(TASKS),--tasks $(TASKS),) $(if $(LANGUAGES),--languages $(LANGUAGES),) $(if $(N),--n $(N),)

# 언어 목록 (성능 결정 부담 순은 analysis.aggregate 가 정한다). mojo 는 아직 하네스에 없다(docs/decisions.md 2026-11-23).
ALL_LANGUAGES    := numpy,c_seq,c_openmp,numba,halide_auto,halide_manual,rust
NEW_LANGUAGES    := numba,halide_auto,halide_manual,rust
# repair-dummy: 수정 턴 더미(gemm c_seq/1.turn1.md, numpy/2.turn1.md)가 있는 언어
REPAIR_LANGUAGES := numpy,c_seq
REPAIR_TURNS     ?= 3

# 과제 목록: tasks/<과제>/gen_tests.py 가 있는 디렉터리 (devcheck-baselines 는 TASKS=a,b 로 좁힌다)
ALL_TASKS := $(sort $(patsubst tasks/%/gen_tests.py,%,$(wildcard tasks/*/gen_tests.py)))
DC_TASKS  := $(if $(TASKS),$(subst $(COMMA), ,$(TASKS)),$(ALL_TASKS))
# devcheck 의 C 툴체인: 비우면 devcheck 기본값(auto: Linux 에 설정의 gcc 가 있으면 gcc, 아니면 zig 직렬). (CC 는 make 내장 변수라 쓰지 않는다)
DEVCHECK_CC ?=
DC_CC := $(if $(DEVCHECK_CC),--cc $(DEVCHECK_CC),)

# 타깃별 RUN_ID 기본값(명령줄 RUN_ID=… 가 우선한다)
dummy:        RUN_ID ?= dummy-$(TIMESTAMP)
dummy-all:    RUN_ID ?= dummy-all-$(TIMESTAMP)
dummy-m3:     RUN_ID ?= dummy-m3-$(TIMESTAMP)
repair-dummy: RUN_ID ?= dummy-repair-$(TIMESTAMP)
real:         RUN_ID ?= $(TIMESTAMP)-real

# dummy-all 은 LANGUAGES 로 바꿀 수 있고, dummy-m3 는 새 언어 네 개로 고정이다. 샘플 수 기본 1(N=… 으로 바꾼다).
dummy-all: DUMMY_LANGUAGES = $(or $(LANGUAGES),$(ALL_LANGUAGES))
dummy-m3:  DUMMY_LANGUAGES = $(NEW_LANGUAGES)

# aggregate·plots: RUN_ID 를 명령줄로 주면 그 run, ALL_RUNS=1 이면 --all-runs, 아니면 --latest. CSV=… 로 파일을 바꿀 수 있다.
ALL_RUNS ?= 0
ifeq ($(origin RUN_ID),command line)
AGG_SELECT := --run-id $(RUN_ID)
else ifneq ($(filter 1 yes true,$(ALL_RUNS)),)
AGG_SELECT := --all-runs
else
AGG_SELECT := --latest
endif
CSV ?=
AGG_CSV := $(if $(CSV),--csv $(CSV),)
# aggregate 에 더할 옵션(예: AGG_FLAGS="--ci --coverage --by-turn --conditions --halide"). plots 의 출력 디렉터리·형식은 OUT=…, FORMAT=png|pdf|svg
AGG_FLAGS ?=
OUT    ?=
FORMAT ?=

.PHONY: help venv rust-setup tests probe unittest devcheck devcheck-baselines dummy dummy-all dummy-m3 repair-dummy real aggregate plots report aa unpack-raw

help:
	@echo "make venv                       .venv 생성 + requirements.txt 설치 (SYSPY=$(SYSPY))"
	@echo "make rust-setup                 cargo fetch --locked (Rust 템플릿의 고정 크레이트; 온라인 1회)"
	@echo "make tests                      tasks/*/tests/*.npz + manifest.json 생성 (과제 12개, 약 1.6 GB)"
	@echo "make probe                      config/machines/<호스트>.yaml 생성·갱신 (harness.protocol --probe --write)"
	@echo "make unittest                   tests/ 단위 테스트"
	@echo "make devcheck [TASKS=…] [LANGUAGES=…] [DEVCHECK_CC=gcc|zig|zig-omp]   모든 0.md 더미 점검 표"
	@echo "make devcheck-baselines [TASKS=…] [DEVCHECK_CC=…]                      과제별 성능 기준선 점검"
	@echo "make dummy [RUN_ID=…] [TASKS=…] [LANGUAGES=…] [N=…] [SKIP_BENCH=1]    더미 파이프라인 → results/runs_dummy.csv"
	@echo "make dummy-all [LANGUAGES=…] [TASKS=…] [N=1] [SKIP_BENCH=1]           언어 일곱 개 × 과제 전부 더미 (+ --coverage)"
	@echo "make dummy-m3 [TASKS=…] [N=1] [SKIP_BENCH=1]                          numba·halide_auto·halide_manual·rust 더미"
	@echo "make repair-dummy [LANGUAGES=numpy,c_seq] [REPAIR_TURNS=3] [SKIP_BENCH=1]   gemm 더미 수정 루프 + 피드백 대응 비교"
	@echo "make real RUN_ID=… [STRICT=1] [TASKS=…] [LANGUAGES=…] [N=…]         실제 LLM 호출(확인 프롬프트) → results/runs.csv"
	@echo "make aggregate [RUN_ID=…|ALL_RUNS=1] [CSV=…] [AGG_FLAGS=…]           결과 집계 (기본 --latest, results/runs.csv)"
	@echo "make plots [RUN_ID=…|ALL_RUNS=1] [CSV=…] [OUT=…] [FORMAT=…]          그래프 (기본 --latest → results/plots/)"
	@echo "make aa [TASKS=a,b]                                                  A/A 점검(기준선 대 기준선; 새 장비에서 측정 전에)"
	@echo "make unpack-raw                                                      1차 실험(노트북) 응답 원문을 results/raw 로 푼다"
	@echo "make report                                                          중간보고서 PDF (results/runs.csv → report/interim/interim_report.pdf)"
	@echo "CONFIG=path                                                          모든 harness 명령에 --config path"

venv:
	$(SYSPY) -m venv .venv && $(PIP) install -r requirements.txt

rust-setup:
	$(CARGO) fetch --locked --manifest-path harness/build/rust_template/Cargo.toml

tests:
	@for t in tasks/*/gen_tests.py; do echo "$(PYTHON) $$t"; $(PYTHON) $$t || exit 1; done

probe:
	$(PYTHON) -m harness.protocol --probe --write $(CFG_FLAG)

unittest:
	$(PYTHON) -m unittest discover -s tests

devcheck:
	$(PYTHON) tools/devcheck.py dummies $(DC_CC) $(if $(TASKS),--tasks $(TASKS),) $(if $(LANGUAGES),--languages $(LANGUAGES),) $(CFG_FLAG)

# 한 과제가 실패해도 나머지를 점검하고, 하나라도 실패(exit 1)·도구 문제(exit 2)가 있으면 끝에 exit 1
devcheck-baselines:
	@rc=0; for t in $(DC_TASKS); do \
		echo "$(PYTHON) tools/devcheck.py baseline --task $$t $(DC_CC) $(CFG_FLAG)"; \
		$(PYTHON) tools/devcheck.py baseline --task $$t $(DC_CC) $(CFG_FLAG) || rc=1; \
	done; exit $$rc

dummy:
	$(PYTHON) -m harness.generate --provider dummy --run-id $(RUN_ID) $(GEN_FILTER) $(CFG_FLAG)
	$(PYTHON) -m harness.run --run-id $(RUN_ID) $(SKIP_BENCH_FLAG) $(CFG_FLAG)
	$(PYTHON) -m analysis.aggregate --csv results/runs_dummy.csv --run-id $(RUN_ID)

# rust 가 들어가므로 `make rust-setup` 을 먼저 한 번 돌려 둔다(없으면 harness.run 이 행을 쓰기 전에 exit 7)
dummy-all dummy-m3:
	$(PYTHON) -m harness.generate --provider dummy --run-id $(RUN_ID) --languages $(DUMMY_LANGUAGES) --n $(or $(N),1) $(if $(TASKS),--tasks $(TASKS),) $(CFG_FLAG)
	$(PYTHON) -m harness.run --run-id $(RUN_ID) $(SKIP_BENCH_FLAG) $(CFG_FLAG)
	$(PYTHON) -m analysis.aggregate --csv results/runs_dummy.csv --run-id $(RUN_ID) --coverage

# 턴 0(feedback=compiler) → 수정 턴 → 같은 턴 0 을 compiler+lint 조건으로 복사(API 호출 없음) → 처리 → 수정 턴 → 집계.
# harness.repair 는 provider 를 설정·환경변수에서 읽으므로 LLM_PROVIDER=dummy 로 턴 0 의 provider 와 맞춘다.
repair-dummy:
	$(PYTHON) -m harness.generate --provider dummy --run-id $(RUN_ID) --tasks gemm --languages $(or $(LANGUAGES),$(REPAIR_LANGUAGES)) --n 5 --feedback compiler $(CFG_FLAG)
	$(PYTHON) -m harness.run --run-id $(RUN_ID) $(SKIP_BENCH_FLAG) $(CFG_FLAG)
	LLM_PROVIDER=dummy $(PYTHON) -m harness.repair --run-id $(RUN_ID) --feedback compiler --max-turns $(REPAIR_TURNS) $(SKIP_BENCH_FLAG) $(CFG_FLAG)
	$(PYTHON) -m harness.repair --run-id $(RUN_ID) --seed-from-feedback compiler --feedback compiler+lint $(CFG_FLAG)
	$(PYTHON) -m harness.run --run-id $(RUN_ID) $(SKIP_BENCH_FLAG) $(CFG_FLAG)
	LLM_PROVIDER=dummy $(PYTHON) -m harness.repair --run-id $(RUN_ID) --feedback compiler+lint --max-turns $(REPAIR_TURNS) $(SKIP_BENCH_FLAG) $(CFG_FLAG)
	$(PYTHON) -m analysis.aggregate --csv results/runs_dummy.csv --run-id $(RUN_ID) --by-turn --conditions

real:
	$(PYTHON) -m harness.generate --run-id $(RUN_ID) $(GEN_FILTER) $(CFG_FLAG)
	$(PYTHON) -m harness.run --run-id $(RUN_ID) $(STRICT_FLAG) $(CFG_FLAG)
	$(PYTHON) -m analysis.aggregate --run-id $(RUN_ID)

aggregate:
	$(PYTHON) -m analysis.aggregate $(AGG_CSV) $(AGG_SELECT) $(AGG_FLAGS)

plots:
	$(PYTHON) -m analysis.plots $(AGG_CSV) $(AGG_SELECT) $(if $(OUT),--out $(OUT),) $(if $(FORMAT),--format $(FORMAT),)

# 중간보고서: runs.csv → 수치·그림 → Typst 조판. 본문 수치가 데이터와 다르면 빌드가 실패한다.
report:
	$(PYTHON) report/interim/build.py

# A/A 점검: 과제의 기준선을 자기 자신과 교차 측정한다(docs/protocol.md §10.2). TASKS 가 없으면 run.tasks 전체.
aa:
	$(PYTHON) -m harness.aa_check $(if $(TASKS),--tasks $(TASKS),) $(CFG_FLAG)

# 1차 실험(노트북, 2026-11-05~09) 응답 원문 묶음. results/raw 는 git 에 올리지 않으므로 새 장비로 옮길 때 이것을 푼다.
unpack-raw:
	mkdir -p results/raw && tar -C results/raw -xzf results/archive/raw_1st_experiment_laptop.tar.gz && ls results/raw
