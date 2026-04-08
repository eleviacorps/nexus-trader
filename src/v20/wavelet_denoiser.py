from __future__ import annotations

import numpy as np
import pandas as pd
import pywt


def denoise_series(series: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    original_len = int(values.size)
    padded_len = 2 ** int(np.ceil(np.log2(max(original_len, 2))))
    series_padded = np.pad(values, (0, padded_len - original_len), mode="reflect")
    coeffs = pywt.wavedec(series_padded, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs) > 1 else 0.0
    threshold = float(sigma * np.sqrt(2.0 * np.log(max(original_len, 2))))
    coeffs_denoised = [coeffs[0]]
    coeffs_denoised.extend(pywt.threshold(detail, threshold, mode="soft") for detail in coeffs[1:])
    reconstructed = pywt.waverec(coeffs_denoised, wavelet)
    return np.asarray(reconstructed[:original_len], dtype=np.float32)


class WaveletDenoiser:
    DENOISE_COLS = ("open", "high", "low", "close")

    def __init__(self, wavelet: str = "db4", level: int = 3) -> None:
        self.wavelet = str(wavelet)
        self.level = int(level)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        for column in self.DENOISE_COLS:
            if column not in output.columns:
                continue
            raw_name = f"{column}_raw"
            denoised_name = f"{column}_denoised"
            output[raw_name] = pd.to_numeric(output[column], errors="coerce").astype(np.float64)
            output[denoised_name] = denoise_series(output[raw_name].to_numpy(dtype=np.float64), self.wavelet, self.level)
            output[column] = output[denoised_name]
            output[f"{column}_denoised_delta"] = output[denoised_name] - output[raw_name]
        return output
