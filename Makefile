# LLM 생성 HPC 코드 평가 하네스 — M1 빌드 글루 (docs/harness_design.md §14)
#
#   make venv                      python3 -m venv .venv && pip install -r requirements.txt
#   make tests                     tasks/*/gen_tests.py (시드 고정 테스트 입력 npz + manifest.json; 8개 과제 전체 약 1.3 GB, 약 15 s)
#   make probe                     harness.protocol --probe --write  (config/machine.yaml 생성·갱신)
#   make unittest                  .venv/bin/python -m unittest discover -s tests
#   make dummy [RUN_ID=…]          generate --provider dummy → run → aggregate --csv results/runs_dummy.csv
#   make real RUN_ID=… [STRICT=1]  generate(확인 프롬프트) → run [--strict-protocol] → aggregate
#   make aggregate [RUN_ID=…] [CSV=…]   RUN_ID 가 없으면 --latest
#
# 모든 python 실행은 .venv/bin/python 이다(시스템 python3 는 `make venv` 에서 venv 를 만들 때만).
# RUN_ID 기본값은 타임스탬프(dummy-YYYYmmdd-HHMMSS / YYYYmmdd-HHMMSS-real)이며 generate·run·aggregate 가 같은 값을 쓴다.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
# venv 를 만들 시스템 인터프리터 (3.14 권장: `make venv SYSPY=python3.14`)
SYSPY  ?= python3

TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)

# STRICT=1 이면 harness.run 에 --strict-protocol (run 시작의 정적 규약 플래그 — 거버너·EPP·taskset·BLAS — 가 있으면 중단, exit 4;
# bench 직전 동적 플래그·baseline_drift 는 strict 여도 행에 기록만 한다)
STRICT ?= 0
STRICT_FLAG := $(if $(filter 1 yes true,$(STRICT)),--strict-protocol,)

# 타깃별 RUN_ID 기본값(명령줄 RUN_ID=… 가 우선한다)
dummy: RUN_ID ?= dummy-$(TIMESTAMP)
real:  RUN_ID ?= $(TIMESTAMP)-real

# aggregate: RUN_ID 를 명령줄로 주면 그 run, 아니면 --latest. CSV=… 로 파일을 바꿀 수 있다.
ifeq ($(origin RUN_ID),command line)
AGG_SELECT := --run-id $(RUN_ID)
else
AGG_SELECT := --latest
endif
CSV ?=
AGG_CSV := $(if $(CSV),--csv $(CSV),)

.PHONY: help venv tests probe unittest dummy real aggregate

help:
	@echo "make venv                       .venv 생성 + requirements.txt 설치 (SYSPY=$(SYSPY))"
	@echo "make tests                      tasks/*/tests/*.npz + manifest.json 생성 (8개 과제, 약 1.3 GB)"
	@echo "make probe                      config/machine.yaml 생성·갱신 (harness.protocol --probe --write)"
	@echo "make unittest                   tests/ 단위 테스트"
	@echo "make dummy [RUN_ID=…]           더미 응답으로 전체 파이프라인 → results/runs_dummy.csv"
	@echo "make real RUN_ID=… [STRICT=1]   실제 LLM 호출(확인 프롬프트) → results/runs.csv"
	@echo "make aggregate [RUN_ID=…] [CSV=…]   결과 집계 (기본 --latest, results/runs.csv)"

venv:
	$(SYSPY) -m venv .venv && $(PIP) install -r requirements.txt

tests:
	@for t in tasks/*/gen_tests.py; do echo "$(PYTHON) $$t"; $(PYTHON) $$t || exit 1; done

probe:
	$(PYTHON) -m harness.protocol --probe --write

unittest:
	$(PYTHON) -m unittest discover -s tests

dummy:
	$(PYTHON) -m harness.generate --provider dummy --run-id $(RUN_ID)
	$(PYTHON) -m harness.run --run-id $(RUN_ID)
	$(PYTHON) -m analysis.aggregate --csv results/runs_dummy.csv --run-id $(RUN_ID)

real:
	$(PYTHON) -m harness.generate --run-id $(RUN_ID)
	$(PYTHON) -m harness.run --run-id $(RUN_ID) $(STRICT_FLAG)
	$(PYTHON) -m analysis.aggregate --run-id $(RUN_ID)

aggregate:
	$(PYTHON) -m analysis.aggregate $(AGG_CSV) $(AGG_SELECT)
