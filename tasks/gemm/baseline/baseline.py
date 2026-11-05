"""gemm 성능 기준선: NumPy(.venv의 pip 휠, OpenBLAS 내장) 행렬곱 (docs/harness_design.md §3).

하네스(harness/run.py)가 이 파일을 python 백엔드 커널로서 LLM 생성 커널과 똑같이
harness.worker로 측정한다(§8: run 시작·1스레드·샘플별 대응 기준선). 워커가 임포트해 gemm을
호출하므로 임포트 시 실행되는 코드를 두지 않는다.
"""
import numpy as np
def gemm(A, B, C, M, N, K):
    np.matmul(A, B, out=C)
