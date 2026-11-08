"""저장소 경로 상수와 경로 헬퍼.

- ``ROOT``: 저장소 루트(이 파일의 상위 디렉터리).
- ``PYTHON_EXE``: 모든 Python 서브프로세스(워커·프로브)에 쓰는 인터프리터.
  ``ROOT/.venv/bin/python`` 이 없으면 ``sys.executable`` 로 대체하고 stderr에 경고한다.
- 결과 디렉터리 레이아웃(설계 §5, §8)을 만드는 헬퍼와 원자적 파일 쓰기 헬퍼.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = ROOT / "config"
DEFAULT_EXPERIMENT_YAML: Path = CONFIG_DIR / "experiment.yaml"
DEFAULT_MACHINE_YAML: Path = CONFIG_DIR / "machine.yaml"
PROMPTS_DIR: Path = ROOT / "prompts"
TASKS_DIR: Path = ROOT / "tasks"
RESULTS_DIR: Path = ROOT / "results"


def _resolve_python_exe() -> Path:
    """격리 venv의 인터프리터를 찾는다. 없으면 현재 인터프리터로 대체하고 경고한다."""
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    print(
        f"[paths] 경고: {venv_python} 가 없어 sys.executable({sys.executable})을 사용한다. "
        "격리 venv(.venv)가 아니면 NumPy BLAS가 참조 구현일 수 있다.",
        file=sys.stderr,
    )
    return Path(sys.executable).resolve()


PYTHON_EXE: Path = _resolve_python_exe()


def resolve(path: str | os.PathLike[str]) -> Path:
    """상대 경로를 ROOT 기준 절대 경로로 바꾼다(절대 경로는 그대로)."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def rel_to_root(path: str | os.PathLike[str]) -> str:
    """ROOT 안의 경로면 ROOT 기준 상대 문자열, 아니면 절대 문자열을 돌려준다(행 기록용)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """디렉터리를 (상위 포함) 만들고 Path를 돌려준다."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cond_tag(args: str, grammar_hint: str, feedback: str) -> str:
    """조건 태그: ``args-{args}_hint-{grammar_hint}_fb-{feedback}`` (설계 §5)."""
    return f"args-{args}_hint-{grammar_hint}_fb-{feedback}"


def raw_sample_path(
    raw_dir: str | os.PathLike[str], run_id: str, task: str, language: str, tag: str, sample_idx: int
) -> Path:
    """생성 원문 경로: ``<raw_dir>/<run_id>/<task>/<language>/<cond_tag>/sample_<i>.json``."""
    return resolve(raw_dir) / run_id / task / language / tag / f"sample_{sample_idx}.json"


def built_sample_dir(
    built_dir: str | os.PathLike[str], run_id: str, task: str, language: str, tag: str, sample_idx: int
) -> Path:
    """빌드·실행 산출물 디렉터리: ``<built_dir>/<run_id>/<task>/<language>/<cond_tag>/sample_<i>/``."""
    return resolve(built_dir) / run_id / task / language / tag / f"sample_{sample_idx}"


def baseline_dir(built_dir: str | os.PathLike[str], run_id: str, task: str) -> Path:
    """기준선 결과 디렉터리: ``<built_dir>/<run_id>/<task>/baseline/``."""
    return resolve(built_dir) / run_id / task / "baseline"


def _default_file_mode() -> int:
    """umask 를 반영한 일반 파일 모드(mkstemp 는 0600 으로 만들므로 보정한다)."""
    old = os.umask(0)
    os.umask(old)
    return 0o666 & ~old


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """같은 디렉터리의 임시 파일에 쓴 뒤 ``os.replace`` 로 바꿔치기한다(원자적)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, _default_file_mode())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str, encoding: str = "utf-8") -> None:
    """문자열을 원자적으로 쓴다."""
    atomic_write_bytes(path, text.encode(encoding))


def json_safe(obj: Any) -> Any:
    """NaN/±Inf 를 None 으로 바꾼 사본(표준 JSON 에는 Infinity/NaN 토큰이 없다; 2026-11-08). dict/list/tuple 을 재귀 처리한다."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def atomic_write_json(path: str | os.PathLike[str], obj: Any, *, indent: int | None = 2) -> None:
    """JSON을 원자적으로 쓴다(UTF-8, 키 순서 유지, 끝에 개행). 비유한 실수는 null 로 쓴다(allow_nan=False 로 재발을 막는다)."""
    atomic_write_text(path, json.dumps(json_safe(obj), ensure_ascii=False, indent=indent, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """경로 상수를 출력한다(진단용)."""
    parser = argparse.ArgumentParser(prog="python -m harness.paths", description="하네스 경로 상수 출력")
    parser.add_argument("--json", action="store_true", help="JSON으로 출력")
    ns = parser.parse_args(argv)
    info = {
        "ROOT": str(ROOT),
        "PYTHON_EXE": str(PYTHON_EXE),
        "DEFAULT_EXPERIMENT_YAML": str(DEFAULT_EXPERIMENT_YAML),
        "DEFAULT_MACHINE_YAML": str(DEFAULT_MACHINE_YAML),
        "RESULTS_DIR": str(RESULTS_DIR),
    }
    if ns.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        for k, v in info.items():
            print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
