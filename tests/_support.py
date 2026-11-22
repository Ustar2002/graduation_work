"""테스트 공용 헬퍼 (설계 v3 §0): 비-Linux 개발 머신(Windows)에서 돌 수 없는 테스트를 건너뛰는 표시.

하네스는 Linux 전용 측정 도구다. 개발은 Windows 에서도 하므로 POSIX/Linux 전용 경로(프로세스 그룹, ``/proc``·``/sys``,
``taskset``, gcc 빌드)에 기대는 테스트는 이 모듈의 표시로 건너뛰고, 나머지는 어디서나 돈다. 표시는 가능한 한 좁게
(테스트 메서드·클래스 단위) 붙이고 이유를 적는다. 단언을 약하게 만드는 용도로 쓰지 않는다.

    from tests._support import requires_linux, requires_modules, requires_rust_tool, requires_tool

    @requires_linux                      # 이유 없이: "Linux 전용"
    @requires_linux("/proc/stat 필요")   # 이유를 적는 형태
    @requires_tool("gcc", "nm")          # PATH 에 실행 파일이 전부 있어야 한다
    @requires_rust_tool("cargo")         # protocol.rust_tool 이 찾아야 한다(PATH 밖의 ~/.cargo/bin 포함)
    @requires_modules("anthropic")       # 개발용 venv 에 없을 수 있는 패키지(Linux 에서는 건너뛰지 않고 실패한다)

**Linux(측정 머신)의 동작은 바꾸지 않는다.** 건너뛰기는 비-Linux 개발 편의일 뿐이다: Linux 에서 ``requirements.txt`` 의
패키지가 빠져 있으면 예전처럼 시끄럽게 실패해야 한다(``requires_modules``).

``python -m unittest discover -s tests`` 와 ``python -m unittest tests.test_<module>`` (둘 다 저장소 루트에서)에서 임포트된다.
"""

from __future__ import annotations

import functools
import importlib.util
import shutil
import sys
import unittest
from pathlib import Path
from typing import Any, Callable

IS_LINUX: bool = sys.platform.startswith("linux")


def _linux_reason(detail: str | None) -> str:
    base = f"Linux 전용 (현재 플랫폼: {sys.platform})"
    return f"{base}: {detail}" if detail else base


def requires_linux(arg: Any = None) -> Any:
    """Linux 가 아니면 건너뛴다. ``@requires_linux`` 와 ``@requires_linux("이유")`` 두 형태를 모두 받는다."""
    if arg is None or isinstance(arg, str):
        return unittest.skipUnless(IS_LINUX, _linux_reason(arg))
    return unittest.skipUnless(IS_LINUX, _linux_reason(None))(arg)


def requires_tool(*names: str) -> Callable[[Any], Any]:
    """PATH 에 실행 파일 ``names`` 가 전부 있어야 돈다(없는 것을 이유에 적는다)."""
    if not names:
        raise ValueError("requires_tool: 실행 파일 이름이 필요하다")
    missing = [n for n in names if shutil.which(n) is None]
    return unittest.skipIf(bool(missing), f"실행 파일 필요: {', '.join(missing)} (PATH 에 없음)")


def rust_tool_found(name: str) -> bool:
    """하네스가 쓰는 ``protocol.rust_tool(name)`` 이 실제 실행 파일을 찾았는가(PATH → ``$CARGO_HOME/bin`` → ``~/.cargo/bin``)."""
    from harness.protocol import rust_tool  # 임포트 시점의 순환을 피한다(_support 는 모든 테스트가 먼저 임포트한다)

    resolved = rust_tool(name)
    return Path(resolved).is_file() or shutil.which(resolved) is not None


def requires_rust_tool(*names: str) -> Callable[[Any], Any]:
    """``cargo``·``rustc`` 를 하네스와 같은 방법(``protocol.rust_tool``)으로 찾아야 돈다. rustup 이 PATH 를 바꾸지 않은
    머신(``~/.cargo/bin`` 에만 있는 경우; docs/decisions.md 2026-11-12)에서도 건너뛰지 않는다. 표시한 테스트는 도구를
    ``protocol.rust_tool(name)`` 과 ``protocol.rust_tool_env()`` 로 띄운다(맨 이름으로 띄우면 PATH 에 기대게 된다)."""
    if not names:
        raise ValueError("requires_rust_tool: 도구 이름이 필요하다")
    missing = [n for n in names if not rust_tool_found(n)]
    return unittest.skipIf(bool(missing),
                           f"Rust 도구 필요: {', '.join(missing)} (PATH·$CARGO_HOME/bin·~/.cargo/bin 어디에도 없음)")


def _fail_with_missing_modules(reason: str) -> Callable[[Any], Any]:
    """표시한 클래스(``setUpClass``)·메서드가 ``ModuleNotFoundError(reason)`` 로 실패하게 만드는 데코레이터."""

    def decorate(obj: Any) -> Any:
        if isinstance(obj, type):
            def setUpClass(cls: Any) -> None:  # noqa: N802 - unittest 가 찾는 이름
                raise ModuleNotFoundError(reason)

            obj.setUpClass = classmethod(setUpClass)
            return obj

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raise ModuleNotFoundError(reason)

        return wrapper

    return decorate


def requires_modules(*modules: str) -> Callable[[Any], Any]:
    """``modules`` 가 전부 설치돼 있어야 도는 테스트(클래스·메서드) 표시 — 개발용 venv 에 없을 수 있는 패키지(LLM SDK 등)용.

    - 전부 있으면 아무것도 바꾸지 않는다.
    - 비-Linux 개발 머신에서 빠진 것이 있으면 이유와 함께 **건너뛴다**.
    - Linux(측정 머신)에서 빠진 것이 있으면 건너뛰지 않는다: 표시한 클래스·메서드가 ``ModuleNotFoundError`` 로 **실패**한다
      (``pip install -r requirements.txt`` 가 깨진 것이다 — skip 으로 초록이 되면 그 경로의 테스트가 조용히 사라진다).

    그 패키지를 실제로 쓰는 클래스·메서드에만 붙인다. 모듈 전체를 건너뛰면(``setUpModule``) 같은 모듈에 나중에 더하는,
    패키지가 필요 없는 테스트까지 개발 머신에서 조용히 빠진다. 테스트 모듈의 임포트는 ``try/except ModuleNotFoundError`` 로
    감싸 둔다::

        try:
            import anthropic
        except ModuleNotFoundError:  # 개발용 venv: 아래 표시가 SDK 테스트만 건너뛴다
            anthropic = None

        @requires_modules("anthropic")
        class StreamTest(unittest.TestCase): ...
    """
    if not modules:
        raise ValueError("requires_modules: 패키지 이름이 필요하다")
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing:
        return lambda obj: obj
    reason = f"설치되지 않은 패키지: {', '.join(missing)} (requirements.txt)"
    if IS_LINUX:
        return _fail_with_missing_modules(reason)
    return unittest.skip(reason)
