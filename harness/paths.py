"""저장소 경로 상수와 경로 헬퍼.

- ``ROOT``: 저장소 루트(이 파일의 상위 디렉터리).
- ``PYTHON_EXE``: 모든 Python 서브프로세스(워커·프로브)에 쓰는 인터프리터.
  ``ROOT/.venv/bin/python`` 이 없으면 ``sys.executable`` 로 대체하고 stderr에 경고한다
  (비-POSIX 개발 머신에는 그 경로가 있을 수 없으므로 경고 없이 ``sys.executable`` 을 쓴다; 설계 v3 §0).
- 결과 디렉터리 레이아웃(설계 §5, §8; 수정 턴은 v3 §9.4)을 만드는 헬퍼와 원자적 파일 쓰기 헬퍼.
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import re
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
    if os.name != "posix":
        # 개발용 Windows: .venv/bin/python 은 원래 없다(측정은 Linux 에서만). 현재 인터프리터를 그대로 쓴다.
        return Path(sys.executable).resolve()
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
    """ROOT 안의 경로면 ROOT 기준 상대 문자열, 아니면 절대 문자열을 돌려준다(행 기록용).

    상대 문자열은 어느 플랫폼에서나 ``/`` 로 구분한다(``as_posix``; POSIX 에서는 ``str()`` 과 같은 문자열이다) —
    개발용 Windows 에서 만든 행의 ``raw_path``·``built_path`` 가 ``\\`` 로 기록되면 Linux 에서 경로로 쓸 수 없다.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
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


# 수정 루프(설계 v3 §9.4): 턴 0 은 기존 경로 그대로, 턴 t ≥ 1 은 같은 샘플 아래에 따로 둔다.
# raw 파일 이름: ``sample_<i>.json``(턴 0) | ``sample_<i>.turn<t>.json``(t ≥ 1). 그룹 1 = sample_idx, 그룹 2 = turn(없으면 0).
# ``sample_<i>.prev<N>.json``(generate 의 보관본)은 맞지 않는다.
RAW_TURN_RE = r"^sample_(\d+)(?:\.turn(\d+))?\.json$"
_RAW_TURN_PATTERN = re.compile(RAW_TURN_RE)


def _check_turn(turn: Any) -> int:
    """``turn`` 을 0 이상의 파이썬 ``int`` 로 돌려준다.

    ``int`` 뿐 아니라 정수형 전반(``__index__`` 를 가진 값 — ``numpy.int64`` 등 pandas/NumPy 로 읽은 행의 값)을 받는다.
    ``bool``·실수(``1.0``)·문자열·``None``·음수는 ``ValueError`` 다(조용히 다른 경로를 만들지 않는다).
    """
    error = ValueError(f"turn 은 0 이상의 정수여야 한다: {turn!r}")
    if isinstance(turn, bool):
        raise error
    try:
        value = operator.index(turn)
    except TypeError:
        raise error from None
    if value < 0:
        raise error
    return value


def raw_turn_path(
    raw_dir: str | os.PathLike[str], run_id: str, task: str, language: str, tag: str, sample_idx: int, turn: int
) -> Path:
    """턴별 생성 원문 경로: 턴 0 → ``…/sample_<i>.json``(= ``raw_sample_path``), t ≥ 1 → ``…/sample_<i>.turn<t>.json``."""
    base = raw_sample_path(raw_dir, run_id, task, language, tag, sample_idx)
    turn = _check_turn(turn)
    if turn == 0:
        return base
    return base.with_name(f"sample_{sample_idx}.turn{turn}.json")


def built_turn_dir(
    built_dir: str | os.PathLike[str], run_id: str, task: str, language: str, tag: str, sample_idx: int, turn: int
) -> Path:
    """턴별 산출물 디렉터리: 턴 0 → ``…/sample_<i>/``(= ``built_sample_dir``), t ≥ 1 → ``…/sample_<i>/turn_<t>/``."""
    base = built_sample_dir(built_dir, run_id, task, language, tag, sample_idx)
    turn = _check_turn(turn)
    if turn == 0:
        return base
    return base / f"turn_{turn}"


def parse_raw_turn_name(name: str) -> tuple[int, int] | None:
    """raw 파일 이름 → ``(sample_idx, turn)``. ``RAW_TURN_RE`` 에 맞지 않으면 None."""
    m = _RAW_TURN_PATTERN.match(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


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


def atomic_write_json(path: str | os.PathLike[str], obj: Any, *, indent: int | None = 2) -> None:
    """JSON을 원자적으로 쓴다(UTF-8, 키 순서 유지, 끝에 개행)."""
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent) + "\n")


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
    # 경로에는 로캘 코드페이지로 인코딩할 수 없는 글자가 있을 수 있다 → 찍다가 죽지 않게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
