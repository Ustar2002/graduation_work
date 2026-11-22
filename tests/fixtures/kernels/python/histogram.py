"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): NumPy 히스토그램 — int32 입력·int32 출력."""
import numpy as np


def histogram(keys, hist, n, n_bins):
    hist[:] = np.bincount(keys, minlength=n_bins)
