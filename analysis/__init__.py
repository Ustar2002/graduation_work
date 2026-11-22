"""결과 분석 패키지. 설계 계약은 ``docs/harness_design.md`` §12 와 ``docs/harness_design_v3.md`` §13.

- ``analysis.aggregate``: ``results/runs.csv`` / ``results/runs_dummy.csv`` 를 읽어 pass@k·속도비·토큰·오류 유형을 집계한다.
  옵션으로 부트스트랩 구간(``--ci``), 과제 × 언어 표(``--coverage``), 수정 턴별 누적 통과율(``--by-turn``),
  조건 수준 사이의 차이(``--conditions``).
- ``analysis.stats``: 고정 seed 의 백분위 부트스트랩(통계량·통과율·대응 표본의 차·독립 표본의 차).
- ``analysis.plots``: 그림 네 장(언어 축, 오류 분류, 조건 효과, 수정 턴)을 파일로 쓴다.
"""
