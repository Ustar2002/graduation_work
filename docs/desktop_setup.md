# 새 장비(리눅스 데스크톱)로 옮기기

2026-11-11 면담 결정에 따라 최종 측정은 집의 리눅스 데스크톱에서 한다. 노트북과 데스크톱이 한 저장소를 함께 쓰도록, 장비마다 다른 것은 따로 둔다.

| 장비마다 다른 것 | 위치 |
|---|---|
| 머신 프로필(CPU 토폴로지·캐시·BLAS·컴파일러) | `config/machines/<hostname>.yaml` (`make probe` 가 만든다) |
| 측정 코어·스레드 수·결과 CSV | 장비용 설정 파일 `config/experiment_desktop.yaml` (아래 7단계) |
| API 키 | 저장소 루트의 `.env` (git 에 올리지 않는다) |
| 테스트 입력·빌드 산출물·응답 원문 | `tasks/*/tests/`, `results/built/`, `results/raw/` (git 에 올리지 않는다) |

데스크톱에서 Claude Code 를 실행하면 아래 단계를 대신 진행할 수 있다(`sudo` 가 필요한 단계만 직접 입력).

## 1. 기본 패키지 (sudo)

```bash
sudo apt install build-essential git python3-venv python3-dev linux-tools-common linux-tools-generic
```

`gcc`(OpenMP 포함), `taskset`, `cpupower` 가 필요하다. Python 은 3.12 이상이면 되고, 노트북과 같은 3.14 를 권장한다.

## 2. 저장소와 Python 환경

```bash
git clone https://github.com/Ustar2002/graduation_work.git ~/dev/graduate_project
cd ~/dev/graduate_project
make venv SYSPY=python3.14
.venv/bin/pip install -r report/requirements.txt
```

`make venv` 는 `requirements.txt` 의 고정 버전(NumPy 2.3.5 + OpenBLAS, numba 0.68.1, halide 22.1.0, mojo 1.2.0 등)을 설치한다. 시스템 numpy 는 참조 BLAS 라 쓰지 않는다.

## 3. Rust (노트북과 같은 판)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.99.1 --no-modify-path
~/.cargo/bin/rustc --version
```

## 4. API 키

노트북의 `.env` 내용을 직접 옮겨 적는다(채팅이나 git 에 붙이지 않는다).

```bash
printf 'ANTHROPIC_API_KEY=%s\n' '<키>' > .env && chmod 600 .env
```

## 5. 거버너 (sudo, 부팅할 때마다)

```bash
sudo cpupower frequency-set -g performance
```

## 6. 머신 프로필과 측정 코어

```bash
make probe
.venv/bin/python -m harness.protocol --suggest-cores 4
```

`--suggest-cores N` 은 물리 성능 코어마다 CPU 하나(SMT 형제와 효율 코어 제외)를 골라 준다. 코어가 많은 장비에서 몇 개를 쓸지는 지도교수와 정한다. 1차 실험 코드를 다시 잴 때는 노트북과 같은 4개를 쓴다.

## 7. 장비용 설정 파일

```bash
cp config/experiment.yaml config/experiment_desktop.yaml
```

`config/experiment_desktop.yaml` 에서 다음만 바꾼다.

- `measurement.cores`: 6단계의 결과(예: `"0,1,2,3"`)
- `measurement.threads`: 코어 수
- `measurement.single_core`: 그 코어 가운데 하나(정확성 검사의 단일 코어 고정)
- `paths.results_csv`: `results/runs_desktop.csv` (노트북 결과 `results/runs.csv` 와 섞지 않는다)

이후 모든 명령에 `CONFIG=config/experiment_desktop.yaml` 을 붙인다.

## 8. 테스트 입력과 단위 테스트

```bash
make tests
make unittest
```

## 9. A/A 점검 → `bench_loop_fsync` 결정

```bash
make aa CONFIG=config/experiment_desktop.yaml
```

기준선을 자기 자신과 교차 측정한다(속도비 1 이 정상). `nofsync` 의 `첫/둘째` 비가 `fsync` 보다 1 에 가깝고 평균 비가 1 ± 0.02 안이면 `measurement.bench_loop_fsync: false` 로 바꾸고 `docs/decisions.md` 에 결과를 적는다(2026-11-12 사용자 승인 조건).

## 10. 1차 실험 코드를 데스크톱에서 다시 재기 (API 비용 없음)

노트북에서 생성한 같은 코드를 다시 재서, 면담에서 지적한 차이가 노트북 환경 때문이었는지 확인한다.

```bash
make unpack-raw
for r in 20261108-tier1 20261108-tier2 20261109-tier3; do .venv/bin/python -m harness.run --run-id $r --force --config config/experiment_desktop.yaml; done
.venv/bin/python -m analysis.aggregate --csv results/runs_desktop.csv --all-runs
```

## 11. 2차 실험 (언어 확장 후)

M3 통합이 끝난 뒤, 모든 조건의 코드를 이 장비의 프롬프트(하드웨어 정보)로 다시 생성한다. API 호출 전에 비용을 확인한다.

```bash
make real RUN_ID=<새 run id> CONFIG=config/experiment_desktop.yaml
```
