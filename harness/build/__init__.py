"""빌드 백엔드 (설계 §7).

- ``harness.build.c``: ``build_c`` — gcc 컴파일 + nm 심벌 검사 (``BuildResult`` 정의).
- ``harness.build.python``: ``build_python`` — ``kernel.py`` 저장 + ``compile()`` 구문 검사.

두 함수 모두 ``BuildResult`` 를 돌려주고 ``out_dir/build.json`` 에 같은 내용을 저장한다.
LLM 코드는 여기서 실행하지 않는다(gcc/nm/compile() 만). 서브모듈을 여기서 임포트하지 않는 이유:
``python -m harness.build.c`` 실행 시 이중 임포트 경고를 피하기 위해서다.
"""

__all__ = ["c", "python"]
