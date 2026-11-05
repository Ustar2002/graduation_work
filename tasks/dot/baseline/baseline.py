"""dot 성능 기준선: NumPy float32 내적(라이브러리 위임). 하네스가 python 백엔드 커널로 똑같이 측정한다."""
import numpy as np


def dot(x, y, out, n):
    out[0] = np.dot(x, y)
