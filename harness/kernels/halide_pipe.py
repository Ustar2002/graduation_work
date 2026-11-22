"""Halide 어댑터 (설계 v3 §8.3). LLM 모듈은 최상위에 **파이프라인 빌드 함수** ``def {fn}({입력 배열…, dim…}):`` 를 정의한다
(출력 인자 없음; 출력 ``hl.Func`` 를 돌려준다). 크기에 따라 파이프라인 구조가 달라질 수 있으므로(T 스텝 반복 등) 워커는
테스트(형태)마다 새 파이프라인을 만든다 — ``prepare``:

1. 입력 배열마다 ``hl.ImageParam(hl.Float(32) | hl.Int(32), 차원 수, 인자 이름)``, dim 인자는 그 형태의 실제 크기(Python ``int``).
2. 빌드 함수 호출(positional 또는 keyword). 어떤 예외든 → ``compile_error``. ``halide_mode == "auto"`` 면 빌드 함수가 도는
   동안 ``hl.Func``·``hl.Stage`` 의 스케줄 메서드와 ``hl.Pipeline.apply_autoscheduler`` 가 **기록만 하는 무동작**이다
   (``ignore_schedule_calls``; 2026-11-12 결정: auto 에서 모델이 쓴 스케줄 지시는 오류가 아니라 무시·기록). 기록은
   ``"Func.parallel"``·``"Stage.reorder"``·``"Stage.rfactor->compute_root"`` 꼴로 ``ignored_schedule_calls`` 에 남는다.
3. 반환값 검사: 정의된 ``hl.Func`` 하나이고 차원 수·원소 타입이 출력과 같아야 한다. 아니면 ``signature_mismatch``.
4. ``halide_mode == "auto"``: 입력·출력 estimate 를 실제 크기로 주고(빌드 함수가 준 값이 있어도 덮어쓴다) 오토스케줄러를
   적용한다(``parallelism`` = 스레드 수; 기본 오토스케줄러 Li2018 — Li2018 은 다른 매개변수를 받지 않는다).
5. ``hl.Pipeline(func).compile_jit(호스트 타깃)``.
6. 데이터가 있으면 입력 ``ImageParam`` 에 ``hl.Buffer(뷰)`` 를 묶고 출력 ``hl.Buffer(뷰)`` 를 만들어 둔다(복사 없음).

``invoke`` 는 ``pipeline.realize(출력 버퍼)`` 뿐이다(출력 뷰에 직접 쓴다). 1–6 은 측정에 들어가지 않고 ``build_ms`` 로 남는다.
``compile_jit`` 과 ``realize`` 는 같은 ``Pipeline`` 객체로 한다(``func.realize`` 는 다시 컴파일한다 — 실측).

좌표: ``hl.Buffer(numpy 배열)`` 의 기본값(``reverse_axes=True``)을 그대로 쓴다. 형태 (D0, D1) 의 행우선 배열에서 행 i0·열 i1 은
Halide 좌표 ``[i1, i0]`` 이고, estimate 도 같은 순서(마지막 축이 첫 좌표)로 준다.

오토스케줄러(``make_loader``; LLM 코드를 실행하기 전): 플러그인은 이름으로는 로드되지 않는다(실측) → halide 패키지 디렉터리에서
파일을 찾아 절대 경로로 로드한다. 못 찾거나 로드하지 못하면 ``compile_error``/``halide_autoscheduler_unavailable:<name>``
(run.py 의 도구 점검이 먼저 잡는다). 로드한 뒤에는 **하네스가 만든** 한 줄짜리 파이프라인에 같은 매개변수로 한 번 걸어 본다:
여기서 실패하면 도구·설정 문제(등록되지 않은 이름, 틀린 매개변수)이므로 예외를 그대로 올린다. 틀린 매개변수는 Halide 22.1 에서
예외가 아니라 프로세스 중단으로 끝난다(실측: Windows 종료 코드 0xC0000409) — 워커가 이 구간을 ``worker_error`` 로 표시해 둔다.
이 시험을 통과했으므로 그 뒤 LLM 파이프라인에서 나는 오토스케줄러 예외는 그 파이프라인 탓으로 기록할 수 있다.

``inspect``: ``loop_nest`` (``print_loop_nest()`` 는 C++ 쪽에서 stderr 에 쓴다 → fd 2 를 임시 파일로 돌려 읽는다),
``schedule_source`` (오토스케줄러 결과; 수동 스케줄이면 ""), ``autoscheduler``, ``target``, ``autoschedule_ms``, ``jit_ms``, ``dims``,
``ignored_schedule_calls`` (auto 에서 무시한 스케줄 호출; manual 은 늘 빈 리스트).
``dims`` 를 주면(성능 형태) 그 크기로 파이프라인을 새로 만들어 스케줄·루프 중첩을 읽는다 — 오토스케줄러의 결정은 estimate 에
따라 달라진다. 이때는 빌드·오토스케줄·``print_loop_nest`` 만 하고(JIT·realize 없음) ``jit_ms`` 는 마지막 ``prepare`` 의 값이다.
"""

from __future__ import annotations

import contextlib
import functools
import io
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from harness.kernels import (
    HALIDE_MODES,
    KernelError,
    KernelHandle,
    KernelOptions,
    Loader,
    autoscheduler_params,
    exception_record,
    one_line,
)
from harness.kernels.pyfunc import MODULE_NAME, load_function
from harness.taskspec import ArgSpec, TaskSpec

_LOG = "[kernels.halide]"

AUTOSCHEDULER_UNAVAILABLE = "halide_autoscheduler_unavailable"
PLUGIN_SUFFIXES: tuple[str, ...] = (".so", ".dll", ".dylib")
PROBE_EXTENT = 1024
_AUTOSCHEDULER_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
# Halide 22.1 의 Func·Stage 스케줄 지시 가운데 정적 목록(harness.build.halide.SCHEDULE_METHODS)에 아직 없는 것. auto 모드의
# 무시·기록 대상은 두 집합의 합이다(같은 이름이 양쪽에 있어도 된다).
EXTRA_SCHEDULE_METHODS: frozenset[str] = frozenset({
    "aligned_split", "eager_inline", "gpu_max_registers", "hoist_invariants", "sme_streaming", "specialize_fail", "split_storage",
    "stream_loads", "stream_stores", "unscheduled",
})
# 무시한 호출이 돌려주는 값: rfactor 는 새 Func 를 돌려주므로 대리 객체, specialize_fail 은 None, 나머지는 자기 자신(체인 유지)
_RETURNS_PROXY: frozenset[str] = frozenset({"rfactor"})
_RETURNS_NONE: frozenset[str] = frozenset({"specialize_fail"})


@dataclass(frozen=True)
class _Runtime:
    """하네스 쪽 준비가 끝난 Halide: 모듈, JIT 타깃, (auto 면) 오토스케줄러 이름과 매개변수 객체, 무시할 스케줄 메서드."""

    hl: Any
    target: Any
    autoscheduler: str | None  # None 이면 수동 스케줄(halide_mode == "manual")
    params: Any  # hl.AutoschedulerParams | None
    schedule_methods: frozenset[str] = frozenset()  # auto 에서 빌드 함수가 도는 동안 무동작으로 바꿀 메서드 이름


@dataclass(frozen=True)
class _Build:
    """형태 하나로 만든 파이프라인과 그때의 관찰값."""

    dims: dict[str, int]
    inputs: dict[str, Any]  # 입력 인자 이름 → hl.ImageParam
    func: Any  # 출력 hl.Func
    pipeline: Any  # hl.Pipeline
    schedule_source: str
    autoschedule_ms: float | None
    jit_ms: float | None  # JIT 를 하지 않은 빌드(inspect 용)는 None
    build_ms: float
    ignored_schedule_calls: tuple[str, ...] = ()  # auto 에서 무시한 스케줄 호출(호출 순서)


def _not_prepared() -> Any:
    raise RuntimeError("prepare(dims, arrays) 를 먼저 불러야 한다")


class _IgnoredResult:
    """무시한 ``rfactor`` 가 돌려주는 대리 객체: 그 위의 어떤 메서드 호출이든 기록하고 자기 자신을 돌려준다(``intm.compute_root()
    .update(0).vectorize(x)`` 같은 체인이 Halide 오류 없이 끝나게 — 정의 없는 ``hl.Func`` 를 돌려주면 ``update(0)`` 가 실패한다)."""

    def __init__(self, record: list[str], origin: str) -> None:
        self._record = record
        self._origin = origin

    def __getattr__(self, attr: str) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> _IgnoredResult:
            self._record.append(f"{self._origin}->{attr}")
            return self

        return call


@contextlib.contextmanager
def ignore_schedule_calls(hl: Any, methods: frozenset[str], record: list[str]) -> Iterator[None]:
    """``hl.Func``·``hl.Stage`` 의 메서드 ``methods`` (그 클래스에 있는 것)와 ``hl.Pipeline.apply_autoscheduler`` 를 이 구간 동안
    호출 이름만 ``record`` 에 적는 무동작으로 바꾼다. 빠져나올 때(예외 포함) 원래대로 되돌린다.

    halide auto 에서 빌드 함수(LLM 코드)를 이 안에서 부른다: 모델이 쓴 스케줄 지시는 알고리즘을 바꾸지 않으므로 무시해도 결과가
    같고, 하네스의 오토스케줄러가 "이미 스케줄됨"으로 실패하거나 모델의 지시 위에 덧쌓이지 않는다(2026-11-12 결정)."""
    saved: list[tuple[type, str, bool, Any]] = []

    def make(cls_name: str, name: str) -> Callable[..., Any]:
        def ignored(self: Any, *args: Any, **kwargs: Any) -> Any:
            label = f"{cls_name}.{name}"
            record.append(label)
            if name in _RETURNS_PROXY:
                return _IgnoredResult(record, label)
            if name in _RETURNS_NONE:
                return None
            return self

        return ignored

    def ignored_autoscheduler(self: Any, *args: Any, **kwargs: Any) -> None:
        record.append("Pipeline.apply_autoscheduler")

    try:
        for cls, cls_name in ((hl.Func, "Func"), (hl.Stage, "Stage")):
            for name in sorted(methods & set(dir(cls))):
                saved.append((cls, name, name in cls.__dict__, getattr(cls, name)))
                setattr(cls, name, make(cls_name, name))
        saved.append((hl.Pipeline, "apply_autoscheduler", "apply_autoscheduler" in hl.Pipeline.__dict__,
                      getattr(hl.Pipeline, "apply_autoscheduler")))
        hl.Pipeline.apply_autoscheduler = ignored_autoscheduler
        yield
    finally:
        for cls, name, own, orig in reversed(saved):
            if own:
                setattr(cls, name, orig)
            else:
                delattr(cls, name)


@contextlib.contextmanager
def compile_messages_to_stderr() -> Iterator[None]:
    """하네스가 부른 Halide 컴파일 단계의 메시지를 stderr 로 돌린다. Halide 의 컴파일 경고는 파이썬 ``sys.stdout`` 으로 나온다
    (바인딩이 ``print`` 를 부른다 — 실측): 그대로 두면 워커의 stdout 으로 새어 커널이 stdout 에 쓴 것으로 기록된다.
    빌드 함수(LLM 코드)의 출력은 돌리지 않는다."""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            yield
    finally:
        if captured.getvalue():
            sys.stderr.write(captured.getvalue())
            sys.stderr.flush()


def capture_stderr_fd(fn: Callable[[], Any]) -> str:
    """``fn`` 이 파일 디스크립터 2 에 쓰는 것을 가로채 돌려준다. Halide 는 C++ 쪽에서 stderr 에 직접 쓰므로
    ``contextlib.redirect_stderr`` 로는 잡히지 않는다."""
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            os.dup2(tmp.fileno(), 2)
            try:
                fn()
            finally:
                sys.stderr.flush()
                os.dup2(saved, 2)
            tmp.seek(0)
            return tmp.read().decode("utf-8", errors="replace")
    finally:
        os.close(saved)


def find_autoscheduler_plugin(package_dir: Path, name: str) -> Path | None:
    """halide 패키지 디렉터리 아래의 오토스케줄러 플러그인 파일 ``*autoschedule_<name 소문자>*.{so,dll,dylib}``. 없으면 None."""
    if not _AUTOSCHEDULER_NAME_RE.fullmatch(name):
        return None
    found = sorted(p for p in Path(package_dir).rglob(f"*autoschedule_{name.lower()}*") if p.suffix in PLUGIN_SUFFIXES and p.is_file())
    return found[0].resolve() if found else None


def load_autoscheduler(hl: Any, name: str) -> None:
    """오토스케줄러 플러그인을 절대 경로로 로드한다. 못 찾거나 로드하지 못하면 ``KernelError`` (계약: ``compile_error``)."""
    unavailable = KernelError("compile_error", f"{AUTOSCHEDULER_UNAVAILABLE}:{name}")
    package_dir = Path(hl.__file__).resolve().parent
    plugin = find_autoscheduler_plugin(package_dir, name)
    if plugin is None:
        print(f"{_LOG} 오토스케줄러 {name!r} 의 플러그인 파일이 {package_dir} 아래에 없다", file=sys.stderr)
        raise unavailable
    try:
        if hasattr(os, "add_dll_directory"):  # Windows: 플러그인이 같은 디렉터리의 DLL 에 의존한다
            with os.add_dll_directory(str(plugin.parent)):
                hl.load_plugin(str(plugin))
        else:
            hl.load_plugin(str(plugin))
    except Exception as e:  # noqa: BLE001 - 로드 실패의 종류를 가리지 않는다
        print(f"{_LOG} 오토스케줄러 플러그인 로드 실패({plugin}): {one_line(e, join_lines=True)}", file=sys.stderr)
        raise unavailable from None


def probe_autoscheduler(hl: Any, target: Any, params: Any) -> None:
    """하네스가 만든 한 줄짜리 파이프라인에 오토스케줄러를 걸어 본다(모듈 docstring). 실패는 도구·설정 문제다."""
    x = hl.Var("x")
    probe = hl.Func("harness_autoscheduler_probe")
    probe[x] = hl.cast(hl.Float(32), x)
    probe.set_estimates([hl.Range(0, PROBE_EXTENT)])
    hl.Pipeline(probe).apply_autoscheduler(target, params)


def _element_type(hl: Any, dtype: str | None) -> Any:
    return {"float32": hl.Float(32), "int32": hl.Int(32)}[str(dtype)]


def _halide_extents(shape: tuple[int, ...]) -> list[int]:
    """NumPy 형태 → Halide 좌표 순서의 extent(마지막 축이 첫 좌표)."""
    return [int(extent) for extent in reversed(shape)]


class HalideKernel(KernelHandle):
    backend = "halide"
    per_shape_build = True
    join_error_lines = True  # Halide 의 오류 메시지는 줄바꿈으로 끝난다

    def __init__(self, builder: Callable[..., Any], spec: TaskSpec, runtime: _Runtime, args_mode: str) -> None:
        self._builder = builder
        self._spec = spec
        self._rt = runtime
        self._keyword = args_mode == "keyword"
        self._last: _Build | None = None
        self._realize: Callable[[], Any] = _not_prepared

    # ------------------------------------------------------------------ 빌드

    def _check_output(self, func: Any, out: ArgSpec, ndim: int) -> None:
        """빌드 함수의 반환값이 출력 계약(정의된 ``hl.Func`` 하나, 차원 수, 원소 타입)에 맞는지 본다."""
        hl = self._rt.hl
        fn = self._spec.function
        if not isinstance(func, hl.Func):
            raise KernelError("signature_mismatch", f"{fn} returned {type(func).__name__}; expected hl.Func")
        if not func.defined():
            raise KernelError("signature_mismatch", f"{fn} returned an hl.Func that has no definition")
        if func.outputs() != 1:
            raise KernelError("signature_mismatch", f"{fn} returned an hl.Func with {func.outputs()} outputs (a Tuple); expected one")
        if func.dimensions() != ndim:
            raise KernelError("signature_mismatch", f"{fn} returned an hl.Func with {func.dimensions()} dimension(s); "
                                                    f"the output {out.name} has {ndim}")
        if func.type() != _element_type(hl, out.dtype):
            raise KernelError("signature_mismatch", f"{fn} returned an hl.Func of type {func.type()}; "
                                                    f"the output {out.name} is {out.dtype}")

    def _build(self, dims: Mapping[str, int], *, jit: bool) -> _Build:
        """``dims`` 크기의 파이프라인을 새로 만든다(모듈 docstring 1–5). LLM 코드 쪽 실패는 ``KernelError``."""
        rt, hl, spec = self._rt, self._rt.hl, self._spec
        dims = {name: int(dims[name]) for name in spec.dim_names}
        t0 = time.perf_counter()
        inputs: dict[str, Any] = {}
        in_extents: dict[str, list[int]] = {}
        values: dict[str, Any] = {}
        for arg in spec.args:
            if arg.kind == "dim":
                values[arg.name] = dims[arg.name]
            elif arg.kind == "in":
                in_extents[arg.name] = _halide_extents(spec.shape_of(arg, dims))
                inputs[arg.name] = values[arg.name] = hl.ImageParam(_element_type(hl, arg.dtype), len(in_extents[arg.name]), arg.name)
        ignored: list[str] = []
        try:
            if rt.autoscheduler is not None:
                with ignore_schedule_calls(hl, rt.schedule_methods, ignored):
                    func = self._call_builder(values)
            else:
                func = self._call_builder(values)
        except BaseException as e:  # SystemExit/KeyboardInterrupt 도 빌드 실패로 본다
            raise KernelError("compile_error", one_line(e, join_lines=True), exception_record(e)) from None

        out = spec.output
        out_extents = _halide_extents(spec.shape_of(out, dims))
        self._check_output(func, out, len(out_extents))

        schedule_source = ""
        autoschedule_ms: float | None = None
        jit_ms: float | None = None
        step = "pipeline construction"
        try:
            with compile_messages_to_stderr():
                pipeline = hl.Pipeline(func)
                if rt.autoscheduler is not None:
                    step = f"autoscheduler {rt.autoscheduler}"
                    t1 = time.perf_counter()
                    # estimate 는 빌드 함수가 끝난 뒤에 준다: 빌드 함수가 스스로 준 값이 있어도 실제 크기가 남는다
                    for name, param in inputs.items():
                        for d, extent in enumerate(in_extents[name]):
                            param.dim(d).set_estimate(0, extent)
                    func.set_estimates([hl.Range(0, extent) for extent in out_extents])
                    schedule_source = str(pipeline.apply_autoscheduler(rt.target, rt.params).schedule_source)
                    autoschedule_ms = (time.perf_counter() - t1) * 1000.0
                if jit:
                    step = "JIT compilation"
                    t2 = time.perf_counter()
                    pipeline.compile_jit(rt.target)
                    jit_ms = (time.perf_counter() - t2) * 1000.0
        except Exception as e:  # noqa: BLE001 - 오토스케줄러는 하네스의 시험 파이프라인을 통과했다: 여기서의 실패는 이 파이프라인 탓이다
            raise KernelError("compile_error", f"{step} failed: {one_line(e, join_lines=True)}", exception_record(e)) from None
        return _Build(dims=dims, inputs=inputs, func=func, pipeline=pipeline, schedule_source=schedule_source,
                      autoschedule_ms=autoschedule_ms, jit_ms=jit_ms, build_ms=(time.perf_counter() - t0) * 1000.0,
                      ignored_schedule_calls=tuple(ignored))

    def _call_builder(self, values: dict[str, Any]) -> Any:
        return self._builder(**values) if self._keyword else self._builder(*values.values())

    # ------------------------------------------------------------------ KernelHandle

    def prepare(self, dims: Mapping[str, int], arrays: Mapping[str, np.ndarray] | None = None) -> float | None:
        hl = self._rt.hl
        self._realize = _not_prepared  # 이번 빌드가 실패하면 앞 형태의 파이프라인이 불리지 않게
        self.ignored_schedule_calls = ()
        build = self._build(dims, jit=True)
        self._last = build
        self.ignored_schedule_calls = build.ignored_schedule_calls
        if arrays is not None:
            for name, param in build.inputs.items():
                param.set(hl.Buffer(arrays[name]))
            self._realize = functools.partial(build.pipeline.realize, hl.Buffer(arrays[self._spec.output.name]))
        return build.build_ms

    def invoke(self) -> Any:
        return self._realize()

    def _collect(self, info: dict[str, Any], dims: Mapping[str, int] | None) -> None:
        rt = self._rt
        info.update({"loop_nest": "", "schedule_source": "", "autoscheduler": rt.autoscheduler, "target": str(rt.target),
                     "autoschedule_ms": None, "jit_ms": self._last.jit_ms if self._last is not None else None, "dims": None,
                     "ignored_schedule_calls": []})
        if dims is not None:
            missing = [name for name in self._spec.dim_names if name not in dims]
            if missing:
                raise ValueError(f"inspect dims lack {missing}")
            build = self._build(dims, jit=False)
        elif self._last is not None:
            build = self._last
        else:
            raise RuntimeError("no pipeline has been built yet")
        info.update({"dims": dict(build.dims), "schedule_source": build.schedule_source, "autoschedule_ms": build.autoschedule_ms,
                     "ignored_schedule_calls": list(build.ignored_schedule_calls)})
        with compile_messages_to_stderr():
            info["loop_nest"] = capture_stderr_fd(build.func.print_loop_nest)


def load(artifact: Path, spec: TaskSpec, *, runtime: _Runtime, args_mode: str, module_name: str = MODULE_NAME) -> HalideKernel:
    """LLM 모듈을 임포트하고 빌드 함수를 찾는다(python 과 같은 로드 규칙). 파이프라인은 ``prepare`` 에서 만든다."""
    return HalideKernel(load_function(artifact, spec.function, join_error_lines=True, module_name=module_name), spec, runtime,
                        args_mode)


def make_loader(options: KernelOptions) -> Loader:
    """Halide 를 임포트하고(없으면 예외 = 도구 문제) auto 면 오토스케줄러를 로드·시험하고 무시할 스케줄 메서드 목록을 정한다.
    LLM 코드는 실행하지 않는다."""
    if options.halide_mode not in HALIDE_MODES:
        raise ValueError(f"halide 백엔드에는 halide_mode 가 필요하다(허용: {HALIDE_MODES}): {options.halide_mode!r}")
    import halide as hl

    target = hl.get_host_target()
    autoscheduler: str | None = None
    params: Any = None
    methods: frozenset[str] = frozenset()
    if options.halide_mode == "auto":
        from harness.build.halide import SCHEDULE_METHODS  # 정적 검사(빌드 단계)와 같은 목록

        methods = frozenset(SCHEDULE_METHODS) | EXTRA_SCHEDULE_METHODS
        autoscheduler = options.halide_autoscheduler
        params = hl.AutoschedulerParams(autoscheduler, autoscheduler_params(options))
        with compile_messages_to_stderr():
            load_autoscheduler(hl, autoscheduler)
            probe_autoscheduler(hl, target, params)
    runtime = _Runtime(hl=hl, target=target, autoscheduler=autoscheduler, params=params, schedule_methods=methods)
    return functools.partial(load, runtime=runtime, args_mode=options.args_mode, module_name=options.module_name)
