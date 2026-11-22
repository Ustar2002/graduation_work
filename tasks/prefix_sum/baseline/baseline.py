"""prefix_sum 성능 기준선: NumPy 누적 합(라이브러리 위임). 하네스가 python 백엔드 커널로 똑같이 측정한다.

``dtype=np.int32`` 를 명시해 누산기를 출력과 같은 형으로 두고 ``out`` 에 직접 쓴다(임시 배열 없음).
"""
import numpy as np


def prefix_sum(x, out, n):
    np.cumsum(x, dtype=np.int32, out=out)
