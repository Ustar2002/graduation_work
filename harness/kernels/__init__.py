"""커널 백엔드 (설계 v3 §5): LLM 이 쓴 커널을 **로드하고 호출하는 방법**을 백엔드별로 감춘다. ``harness.worker`` 만 쓴다.

    load = kernels.open_backend("halide", options)   # 하네스 쪽 준비만 — LLM 코드는 아직 실행하지 않는다
    handle = load(artifact, spec)                     # LLM 모듈 임포트·심벌 조회 → KernelHandle
    build_ms = handle.prepare(dims, arrays)           # 테스트(형태) 하나에 대한 준비
    ret = handle.invoke()                             # 커널 호출 한 번 — 워커가 재는 구간에는 이것만 들어간다
    info = handle.inspect(dims)                       # ``--inspect``: 컴파일 결과 관찰값

    backend   모듈          산출물                    호출(§5.2)
    c         cabi          공유 라이브러리(C·Rust)   positional ``fn(*args)`` / keyword ``fn(byref(ArgsStruct))``
    python    pyfunc        kernel.py                 ``fn(*args)`` / ``fn(**kwargs)``
    numba     pyfunc        kernel.py                 python 과 같다(+ 명시 서명으로 즉시 컴파일됐는지 검사)
    halide    halide_pipe   kernel.py(빌드 함수)      형태마다 파이프라인 빌드·JIT → ``pipeline.realize(출력 버퍼)``

단계를 둘로 나눈 이유: ``open_backend`` 는 하네스가 고른 도구만 건드린다(numba·halide 임포트, numba 스레딩 계층 시작,
오토스케줄러 플러그인 로드와 시험 적용). 여기서 나는 예외나 프로세스 종료는 LLM 코드 탓이 아니므로 워커가 ``worker_error`` 로
기록한다 — 없는 도구·틀린 설정이 모델 오류 행이 되면 안 된다. LLM 코드는 ``load`` 부터 실행된다.

실패 규칙:

- ``KernelError(classification, detail[, exception])``: LLM 커널 쪽 원인으로 로드·준비를 못 했다. import 모드에서는
  ``load_ok=false`` 와 그 ``classification`` (``compile_error`` | ``signature_mismatch``), check/bench 에서 형태별 준비가
  실패하면 그 테스트의 ``runtime_crash`` 로 기록된다(``exception`` 은 커널이 올린 예외와 같은 꼴의 기록).
- 그 밖의 예외는 하네스·도구 문제다(워커의 ``worker_error``).
- 커널 호출(``invoke``)이 올리는 예외는 잡지 않는다 — 워커가 ``runtime_crash`` 로 기록한다.
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from harness.taskspec import TaskSpec

BACKENDS: tuple[str, ...] = ("c", "python", "numba", "halide")
ARGS_MODES: tuple[str, ...] = ("positional", "keyword")
HALIDE_MODES: tuple[str, ...] = ("auto", "manual")
DEFAULT_AUTOSCHEDULER = "Mullapudi2016"

ONE_LINE_CHARS = 500
MESSAGE_CHARS = 1000
TRACEBACK_TAIL_CHARS = 2000


# ----------------------------------------------------------------------------- 예외 기록


def safe_str(e: BaseException) -> str:
    """``str(e)``. 커널이 올린 예외 객체의 ``__str__`` 자체가 예외를 던지면(LLM 코드) ``<unprintable Type>`` —
    예외 핸들러가 다시 죽어 worker_error(rc=3, run 중단)가 되지 않도록 한다."""
    try:
        return str(e)
    except BaseException:  # __str__ 이 SystemExit 등을 올려도 워커는 살아야 한다
        return f"<unprintable {type(e).__name__}>"


def safe_format_exc() -> str:
    """``traceback.format_exc()``; 포맷 중 예외(커널 예외 객체의 속성이 던지는 경우)면 자리표시 문자열."""
    try:
        return traceback.format_exc()
    except BaseException:
        return "<traceback unavailable>"


def single_line(text: str) -> str:
    """여러 줄 문자열을 한 줄로: 줄마다 앞뒤 공백을 떼고 빈 줄을 뺀 뒤 공백 하나로 잇는다."""
    return " ".join(part for part in (line.strip() for line in text.splitlines()) if part)


def one_line(e: BaseException, *, join_lines: bool = False) -> str:
    """예외를 한 줄 ``Type: message`` 로(500자). 메시지가 여러 줄이면 첫 줄만 쓴다. ``join_lines`` 면 모든 줄을 공백으로 잇는다 —
    Numba·Halide 의 오류는 첫 줄이 머리말이고 원인은 그 뒤 줄에 있다."""
    message = safe_str(e).strip()
    if join_lines:
        text = single_line(message)
    else:
        lines = message.splitlines()
        text = lines[0] if lines else ""
    return f"{type(e).__name__}: {text}".rstrip(": ")[:ONE_LINE_CHARS]


def exception_record(e: BaseException) -> dict[str, Any]:
    """결과 문서의 예외 기록 ``{type, message, traceback}``. ``except`` 블록 안에서 부른다(지금 처리 중인 예외의 트레이스백을 쓴다)."""
    return {"type": type(e).__name__, "message": safe_str(e)[:MESSAGE_CHARS], "traceback": safe_format_exc()[-TRACEBACK_TAIL_CHARS:]}


def crash_detail(record: Mapping[str, Any]) -> str:
    """예외 기록을 그 호출의 ``runtime_crash`` 설명 한 줄 ``Type: message`` 로(500자). 여러 줄 메시지는 한 줄로 잇는다:
    Halide 의 메시지는 줄바꿈으로 끝나고 Numba 의 메시지는 원인이 뒷줄에 있다(기록의 ``message`` 는 원문 그대로다)."""
    return f"{record['type']}: {single_line(str(record['message']))}".rstrip(": ")[:ONE_LINE_CHARS]


class KernelError(Exception):
    """LLM 커널을 로드하거나 형태에 맞게 준비하지 못했다(모듈 docstring 의 실패 규칙).

    ``classification``: import 모드의 판정(``compile_error`` | ``signature_mismatch``). ``detail``: 한 줄 설명(500자까지; 행의
    ``error_detail`` 과 수정 프롬프트에 들어가므로 영어로, 하네스 내부 용어 없이 쓴다). ``exception``: LLM 코드나 그 코드를
    처리하던 컴파일 단계가 올린 예외의 기록(없으면 None).
    """

    def __init__(self, classification: str, detail: str, exception: dict[str, Any] | None = None) -> None:
        detail = detail[:ONE_LINE_CHARS]
        super().__init__(f"{classification}: {detail}")
        self.classification = classification
        self.detail = detail
        self.exception = exception

    def record(self) -> dict[str, Any]:
        """check/bench 의 테스트 기록에 넣는 예외 기록. 예외 없이 난 실패(반환값 검사 등)는 판정 이름을 타입 자리에 쓴다."""
        return self.exception or {"type": self.classification, "message": self.detail, "traceback": ""}


# ----------------------------------------------------------------------------- 옵션


@dataclass(frozen=True)
class KernelOptions:
    """워커 명령줄의 백엔드 옵션(설계 v3 §5.1). ``threads`` 는 오케스트레이터가 환경변수로 건 측정 스레드 수와 같은 값이다."""

    args_mode: str = "positional"
    threads: int | None = None
    halide_mode: str | None = None
    halide_autoscheduler: str = DEFAULT_AUTOSCHEDULER
    halide_params: Mapping[str, Any] = field(default_factory=dict)


def autoscheduler_params(options: KernelOptions) -> dict[str, str]:
    """오토스케줄러에 넘기는 매개변수(§5.3): ``{"parallelism": str(threads), **halide_params}`` — 값은 전부 문자열."""
    if options.threads is None:
        raise ValueError("halide_mode=auto 에는 threads 가 필요하다(오토스케줄러의 parallelism)")
    return {"parallelism": str(options.threads), **{str(k): str(v) for k, v in options.halide_params.items()}}


def backend_detail(backend: str, options: KernelOptions) -> dict[str, Any]:
    """결과 문서의 ``backend_detail``: 이 실행이 받은 백엔드 설정(커널 로드가 실패해도 기록된다)."""
    detail: dict[str, Any] = {}
    if options.threads is not None:
        detail["threads"] = options.threads
    if backend == "halide":
        auto = options.halide_mode == "auto"
        detail["halide_mode"] = options.halide_mode
        detail["autoscheduler"] = options.halide_autoscheduler if auto else None
        if auto:
            detail["autoscheduler_params"] = autoscheduler_params(options)
    return detail


# ----------------------------------------------------------------------------- 핸들


class KernelHandle(ABC):
    """로드된 커널 하나. 워커는 테스트(형태)마다 ``prepare`` 를 한 번, 그 뒤 ``invoke`` 를 여러 번 부른다."""

    per_shape_build: bool = False  # ``prepare`` 가 형태마다 LLM 코드를 다시 실행해 커널을 만드는가(halide)
    has_return_value: bool = False  # ``invoke`` 의 반환값이 커널의 반환값인가(python 계열: 워커가 타입 이름을 기록한다)

    @abstractmethod
    def prepare(self, dims: Mapping[str, int], arrays: Mapping[str, np.ndarray] | None = None) -> float | None:
        """형태 하나에 대한 준비. ``dims`` 는 dim 인자 값, ``arrays`` 는 인자 이름 → 배열 뷰(과제 형태의 N-D, C-연속, 쓰기 가능;
        워커가 호출 사이에 같은 메모리를 복원·재사용한다). ``arrays`` 가 None 이면 데이터 없이 형태별 빌드만 한다(import 모드).

        반환: 형태별 빌드 시간(ms). 형태별 빌드가 없는 백엔드는 None. 측정 구간 밖이다."""

    @abstractmethod
    def invoke(self) -> Any:
        """마지막 ``prepare`` 로 묶은 인자로 커널을 한 번 호출한다. 커널이 올린 예외는 그대로 올라간다."""

    def inspect(self, dims: Mapping[str, int] | None = None) -> dict[str, Any]:
        """``--inspect``: 컴파일 결과의 관찰값(§5.1). ``dims`` 가 있으면 그 크기(성능 형태)의 커널을 기술한다.
        수집 중의 실패는 ``inspect_error`` 키에 한 줄로 남기고, 그때까지 모은 값은 그대로 돌려준다(예외를 올리지 않는다)."""
        info: dict[str, Any] = {}
        try:
            self._collect(info, dims)
        except Exception as e:  # noqa: BLE001 - 관찰값을 못 모았다고 워커가 실패하면 안 된다
            info["inspect_error"] = one_line(e, join_lines=True)
        return info

    def _collect(self, info: dict[str, Any], dims: Mapping[str, int] | None) -> None:
        """``info`` 를 채운다(백엔드별). 기본: 관찰할 것이 없다."""


Loader = Callable[[Path, TaskSpec], KernelHandle]


def open_backend(backend: str, options: KernelOptions) -> Loader:
    """백엔드를 준비하고 ``load(artifact, spec) -> KernelHandle`` 을 돌려준다. **LLM 코드는 실행하지 않는다.**

    하네스가 고른 도구를 쓸 수 없으면 예외가 그대로 올라간다(도구·설정 문제). 계약상 예외 하나: halide 오토스케줄러
    플러그인을 찾지 못하거나 로드하지 못하면 ``KernelError("compile_error", "halide_autoscheduler_unavailable:<name>")``.
    """
    if options.args_mode not in ARGS_MODES:
        raise ValueError(f"알 수 없는 args_mode: {options.args_mode!r} (허용: {ARGS_MODES})")
    if backend == "c":
        from harness.kernels import cabi

        return cabi.make_loader(options)
    if backend in ("python", "numba"):
        from harness.kernels import pyfunc

        return pyfunc.make_loader(backend, options)
    if backend == "halide":
        from harness.kernels import halide_pipe

        return halide_pipe.make_loader(options)
    raise ValueError(f"알 수 없는 backend: {backend!r} (허용: {BACKENDS})")
