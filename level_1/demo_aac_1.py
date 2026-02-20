"""
Level 1 - Demo AAC
Dokimastiko programma gia ton kwdikopoihth/apokwdikopoihth

Ylopoiei:
    SNR = demo_aac_1(filename_in, filename_out)
"""

import numpy as np
import soundfile as sf

from .aac_coder_1 import aac_coder_1, i_aac_coder_1


def demo_aac_1(filename_in, filename_out):
    """
    Dokimh tou AAC level 1 kai epistrefei SNR se dB
    
    To apokwdikopoihmeno shma kovetai sto mhkos tou arxikou prin apothikeutei sto arxeio eksodou
    """
    x, fs = sf.read(filename_in, always_2d=True)

    # Kwdikopihsh kai apokwdikopihsh
    aac_seq = aac_coder_1(filename_in)   # Kwdikopoihsh AAC, epistrefei lista apo frames me frame type kai syntelestes MDCT
    y = i_aac_coder_1(aac_seq, filename_out)    # Apokwdikopihsh AAC, epistrefei to apokwdikopoihmeno shma

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
