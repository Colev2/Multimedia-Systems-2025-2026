"""
Level 2 - Demo AAC me TNS
Dokimastiko programma gia ton kwdikopoihth/apokwdikopoihth me TNS

Ylopoiei:
    SNR = demo_aac_2(filename_in, filename_out)
"""

import numpy as np
import soundfile as sf

from .aac_coder_2 import aac_coder_2, i_aac_coder_2


def demo_aac_2(filename_in, filename_out):
    """
    Dokimh tou AAC level 2 kai epistrofh SNR se dB
    
    To apokwdikopoihmeno shma kovetai sto mhkos tou arxikou
    prin apothikeutei sto arxeio exodou
    """
    x, fs = sf.read(filename_in, always_2d=True)

    # Kodikwpoihsh kai apokwdikopoihsh
    aac_seq = aac_coder_2(filename_in)
    y = i_aac_coder_2(aac_seq, filename_out)

    # Kopsimo sto mhkos tou arxikou
    n = x.shape[0]
    y = y[:n, :]
    sf.write(filename_out, y, fs)

    # Ypologismos SNR
    err = x - y
    signal_power = np.sum(x * x)
    noise_power = np.sum(err * err)

    if noise_power <= 0:
        return float("inf")

    snr = 10.0 * np.log10(signal_power / noise_power)
    return float(snr)
