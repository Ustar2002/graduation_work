"""sort 성능 기준선: NumPy 정렬(라이브러리 위임). 하네스가 python 백엔드 커널로 똑같이 측정한다.

키를 출력 버퍼에 복사한 뒤 그 자리에서 정렬한다(입력 ``keys`` 는 건드리지 않고, n 개짜리 임시 배열도 만들지 않는다).
"""
import numpy as np


def sort(keys, out, n):
    np.copyto(out, keys)
    out.sort()
