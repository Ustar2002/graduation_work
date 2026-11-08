"""sort 성능 기준선: NumPy 정렬(라이브러리 위임, 단일 스레드). 하네스가 python 백엔드 커널로 똑같이 측정한다.
병렬 정렬 커널의 속도비가 1 을 넘는 것은 정상이다(dot 과 같은 해석; docs/decisions.md 2026-11-08).
실측 2026-11-08, large(10^7) worker bench 중앙값: 이 기준선 1스레드(core 2) 58.2 ms(CV 1.6 %), 4스레드 설정 57.7 ms(정렬은 단일 스레드);
더미 c_openmp(8비트 4패스 병렬 LSD 기수 정렬, 4스레드) 37.9 ms, 더미 c_seq(qsort) 1269 ms."""
import numpy as np


def sort(x, out, n):
    out[...] = np.sort(x)
