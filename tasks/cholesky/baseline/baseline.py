"""cholesky 성능 기준선: NumPy `np.linalg.cholesky`(LAPACK potrf, float32; 라이브러리 위임). 하네스가 python 백엔드 커널로 똑같이 측정한다.
반환되는 L 은 하삼각이고 엄격 상삼각은 0 이라 `L[...] = …` 한 번으로 출력 전체가 기록된다."""
import numpy as np


def cholesky(A, L, n):
    L[...] = np.linalg.cholesky(A)
