"""cholesky 성능 기준선: NumPy float32 Cholesky 분해(라이브러리 위임; 휠에 든 OpenBLAS 의 LAPACK spotrf).

하네스가 이 파일을 python 백엔드 커널로서 LLM 생성 커널과 똑같이 측정한다. ``np.linalg.cholesky`` 는 대각 위가 0 인
하삼각 인자를 새 배열로 돌려주므로(``out=`` 인자가 없다) 그것을 출력 버퍼에 복사한다.
"""
import numpy as np


def cholesky(A, L, n):
    L[...] = np.linalg.cholesky(A)
