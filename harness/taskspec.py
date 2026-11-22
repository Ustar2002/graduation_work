"""과제 명세(``tasks/<task>/signatures.yaml``)의 기계 판독 부분: 인자 목록·형태·기준선·작업량.

M2(2026-11-05)에서 하네스를 GEMM 전용에서 과제 일반으로 바꾸며 추가했다. 워커·run·generate 가 같이 쓴다.
v3(2026-11-20; docs/harness_design_v3.md §2.1)에서 정수 출력, 인자 설명(``doc``), 지원하지 않는 언어(``unsupported``),
언어별 원문(``languages``) 보관을 더했다.

signatures.yaml 의 관련 키::

    task: gemm
    function: gemm
    params: [A, B, C, M, N, K]            # 호출 순서(= args 의 name 순서와 같아야 한다)
    args:
      - {name: A, kind: in,  dtype: float32, shape: [M, K]}
      - {name: C, kind: out, dtype: float32, shape: [M, N]}
      - {name: M, kind: dim}
      - {name: keys, kind: in, dtype: int32, shape: [n], doc: "each value is in the range 0 <= keys[i] < n_bins"}
    work: "2*M*N*K"                       # 부동소수 연산 수(선택; gflops 계산용)
    out_guard_extra: "max(M,N)**2 - M*N"  # 출력 가드에 더할 원소 수(선택; arg_order 패턴용)
    baseline: {backend: python, file: baseline/baseline.py, openmp: false, gate_blas: true, gate_min_gflops: true}
    unsupported:                          # 선택: 이 과제에서 돌리지 않는 언어 → 이유(영어 한 줄; 분석 표에 그대로 나온다)
      halide_auto: "data-dependent control flow is not expressible as a Halide pipeline"
    languages:                            # 선택: 언어별 시그니처·설명 원문. 있으면 렌더러(harness/sigs.py)보다 우선한다
      numpy: {positional: "def gemm(A, B, C, M, N, K):", ...}

- ``kind``: ``in``(읽기 전용 배열) | ``out``(호출자가 할당한 출력 배열; 과제당 정확히 1개) | ``dim``(C ``int`` 스칼라 인자 —
  차원이 아닌 정수, 예: BFS 의 ``source`` 에도 쓴다).
- 배열 dtype 은 ``float32`` 또는 ``int32``. 출력도 둘 다 된다(정수 출력은 Tier 2 부터). 복수 출력은 없다.
- ``doc``: 프롬프트의 매개변수 설명에 덧붙는 영어 한 구절(선택; 배열·dim 모두).
- ``shape`` 원소는 정수 또는 dim 이름으로 된 식(예: ``"n_rows + 1"``). 테스트 npz 는 인자 이름으로 배열·스칼라를,
  ``<out>_ref`` 로 참조(float 출력은 float64, int32 출력은 int32)를 담는다.
- 식은 이 저장소의 신뢰된 파일에서만 오며, 내장 함수 없이 ``max``/``min`` 만 허용해 평가한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

KINDS = ("in", "out", "dim")
DTYPES = ("float32", "int32")
C_TYPE_NAMES = {"float32": "float", "int32": "int"}
DEFAULT_BASELINE: dict[str, Any] = {"backend": "python", "file": "baseline/baseline.py", "openmp": False,
                                    "gate_blas": True, "gate_min_gflops": False}


class TaskSpecError(ValueError):
    """signatures.yaml 의 기계 판독 부분이 잘못됐다."""


@dataclass(frozen=True)
class ArgSpec:
    name: str
    kind: str
    dtype: str | None = None
    shape: tuple[str, ...] = ()
    doc: str = ""  # 프롬프트 매개변수 설명에 덧붙는 영어 구절(없으면 빈 문자열)


@dataclass(frozen=True)
class TaskSpec:
    task: str
    function: str
    args: tuple[ArgSpec, ...]
    work: str | None = None
    out_guard_extra: str | None = None
    baseline: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_BASELINE))
    path: Path | None = None
    unsupported: dict[str, str] = field(default_factory=dict)  # 언어 → 돌리지 않는 이유(영어 한 줄)
    languages_raw: dict[str, dict[str, Any]] = field(default_factory=dict)  # YAML ``languages`` 원문(언어 → 항목)

    @property
    def params(self) -> list[str]:
        return [a.name for a in self.args]

    @property
    def dim_names(self) -> list[str]:
        return [a.name for a in self.args if a.kind == "dim"]

    @property
    def inputs(self) -> list[ArgSpec]:
        return [a for a in self.args if a.kind == "in"]

    @property
    def output(self) -> ArgSpec:
        return next(a for a in self.args if a.kind == "out")

    def supports(self, language: str) -> bool:
        """이 과제를 그 언어로 돌리는가(= ``unsupported`` 에 없음)."""
        return language not in self.unsupported

    def eval_expr(self, expr: Any, dims: dict[str, int]) -> int:
        """정수 또는 dim 식을 평가한다."""
        if isinstance(expr, (int,)) and not isinstance(expr, bool):
            return int(expr)
        try:
            value = eval(str(expr), {"__builtins__": {}, "max": max, "min": min}, dict(dims))  # noqa: S307 - 신뢰된 저장소 파일
        except Exception as e:
            raise TaskSpecError(f"{self.task}: 식 {expr!r} 평가 실패 ({type(e).__name__}: {e}); dims={dims}") from None
        return int(value)

    def shape_of(self, arg: ArgSpec, dims: dict[str, int]) -> tuple[int, ...]:
        return tuple(self.eval_expr(s, dims) for s in arg.shape)

    def work_of(self, dims: dict[str, int]) -> float | None:
        return float(self.eval_expr(self.work, dims)) if self.work else None

    def guard_extra_of(self, dims: dict[str, int]) -> int:
        return max(0, self.eval_expr(self.out_guard_extra, dims)) if self.out_guard_extra else 0


def _parse_doc(item: dict[str, Any], path: Path | None) -> str:
    doc = item.get("doc")
    if doc is None:
        return ""
    if not isinstance(doc, str):
        raise TaskSpecError(f"{path}: 인자 {item.get('name')!r} 의 doc 는 문자열이어야 한다: {doc!r}")
    return " ".join(doc.split())  # 프롬프트에 한 구절로 들어간다(개행·연속 공백 정리)


def _parse_unsupported(raw: Any, path: Path | None) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TaskSpecError(f"{path}: unsupported 는 언어 → 이유 매핑이어야 한다")
    out: dict[str, str] = {}
    for language, reason in raw.items():
        if not isinstance(language, str) or not language or not isinstance(reason, str) or not reason.strip():
            raise TaskSpecError(f"{path}: unsupported 항목은 언어 이름과 비어 있지 않은 이유 문자열이어야 한다: {language!r}: {reason!r}")
        out[language] = " ".join(reason.split())
    return out


def _parse_languages(raw: Any, path: Path | None) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TaskSpecError(f"{path}: languages 는 언어 → 항목 매핑이어야 한다")
    out: dict[str, dict[str, Any]] = {}
    for language, entry in raw.items():
        if not isinstance(entry, dict):
            raise TaskSpecError(f"{path}: languages.{language} 는 매핑이어야 한다")
        out[str(language)] = dict(entry)
    return out


def parse_taskspec(raw: dict[str, Any], path: Path | None = None) -> TaskSpec:
    if not isinstance(raw, dict):
        raise TaskSpecError(f"{path}: 최상위가 매핑이 아니다")
    task = str(raw.get("task") or "")
    function = str(raw.get("function") or task)
    args_raw = raw.get("args")
    if not task or not isinstance(args_raw, list) or not args_raw:
        raise TaskSpecError(f"{path}: task 와 args(리스트)가 필요하다")
    args: list[ArgSpec] = []
    for item in args_raw:
        if not isinstance(item, dict) or "name" not in item or item.get("kind") not in KINDS:
            raise TaskSpecError(f"{path}: args 항목이 잘못됐다: {item!r}")
        kind = item["kind"]
        doc = _parse_doc(item, path)
        if kind == "dim":
            args.append(ArgSpec(name=str(item["name"]), kind=kind, doc=doc))
            continue
        dtype = str(item.get("dtype") or "")
        shape = item.get("shape")
        if dtype not in DTYPES or not isinstance(shape, list) or not shape:
            raise TaskSpecError(f"{path}: 배열 인자 {item.get('name')!r} 는 dtype∈{DTYPES} 와 shape 가 필요하다")
        args.append(ArgSpec(name=str(item["name"]), kind=kind, dtype=dtype, shape=tuple(str(s) for s in shape), doc=doc))
    names = [a.name for a in args]
    if len(set(names)) != len(names):
        raise TaskSpecError(f"{path}: args 이름이 중복된다: {names}")
    params = raw.get("params")
    if isinstance(params, list) and [str(p) for p in params] != names:
        raise TaskSpecError(f"{path}: params {params} 와 args 순서 {names} 가 다르다")
    outs = [a for a in args if a.kind == "out"]
    if len(outs) != 1:
        raise TaskSpecError(f"{path}: 출력 인자는 정확히 1개여야 한다(현재 {len(outs)}개)")
    # 출력 dtype 은 float32 또는 int32 (v3 §2.1; 배열 dtype 검사는 위에서 끝났다).
    baseline = dict(DEFAULT_BASELINE)
    if isinstance(raw.get("baseline"), dict):
        baseline.update(raw["baseline"])
    if baseline.get("backend") not in ("python", "c"):
        raise TaskSpecError(f"{path}: baseline.backend 는 python|c")
    work = raw.get("work")
    extra = raw.get("out_guard_extra")
    return TaskSpec(task=task, function=function, args=tuple(args), work=str(work) if work else None,
                    out_guard_extra=str(extra) if extra else None, baseline=baseline, path=path,
                    unsupported=_parse_unsupported(raw.get("unsupported"), path),
                    languages_raw=_parse_languages(raw.get("languages"), path))


def load_taskspec(path: Path) -> TaskSpec:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_taskspec(raw, Path(path))
