"""fft 성능 기준선: NumPy 의 float32 복소수(complex64) FFT(라이브러리 위임). 하네스가 python 백엔드 커널로 똑같이 측정한다.

(n, 2) float32 교차 저장 버퍼를 complex64 뷰로 보고 ``np.fft.fft`` 를 호출해 결과를 out 의 뷰에 쓴다.
NumPy ≥ 2.0 의 pocketfft 는 complex64 입력에 단정밀도 변환을 쓰고 complex64 를 돌려준다(2026-11-09 실측: 복사·승격 없음).
"""
import numpy as np


def fft(x, out, n):
    out.reshape(-1).view(np.complex64)[...] = np.fft.fft(x.reshape(-1).view(np.complex64))
