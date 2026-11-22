"""빌드 백엔드 (설계 §7, v3 §7).

- ``harness.build.c``: ``build_c`` — gcc 컴파일 + nm 심벌 검사. 공통 결과형 ``BuildResult`` 와 ``ToolchainMissing`` 을 정의한다.
- ``harness.build.python``: ``build_python`` — ``kernel.py`` 저장 + ``compile()`` 구문 검사.
- ``harness.build.numba``: ``build_numba`` — 구문 검사 + ``@njit("<서명>")`` 데코레이터의 AST 검사.
- ``harness.build.halide``: ``build_halide`` — 구문 검사 + 스케줄 호출 관찰(``SCHEDULE_METHODS``).
- ``harness.build.rust``: ``build_rust`` — cargo ``cdylib`` 빌드(``rust_template/`` 의 ``Cargo.toml``·``Cargo.lock``) + nm 심벌 검사.

모든 함수가 ``BuildResult`` 를 돌려주고 ``out_dir/build.json`` 에 같은 내용을 저장한다. 빌드 도구를 쓸 수 없으면(컴파일러·cargo
없음 등) 결과 대신 ``ToolchainMissing`` 을 던진다 — 없는 도구는 모델 오류가 아니다.
LLM 코드는 여기서 실행하지 않는다(gcc/cargo/nm/compile() 만). 서브모듈을 여기서 임포트하지 않는 이유:
``python -m harness.build.c`` 실행 시 이중 임포트 경고를 피하기 위해서다.
"""

__all__ = ["c", "python", "numba", "halide", "rust"]
