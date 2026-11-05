"""LLM 생성 HPC 코드 평가 하네스 (M1 수직 슬라이스).

설계 계약은 ``docs/harness_design.md`` 를 따른다. 모든 서브프로세스는
``harness.procutil.run_isolated`` 로 실행하고, LLM 코드는 ``harness.worker``
서브프로세스 안에서만 실행한다.
"""
