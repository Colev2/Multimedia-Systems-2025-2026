"""
Level 3 - AAC Demo
Epideixi kodikopoiitis/apokodikopiti AAC me ypologismo SNR, bitrate, compression
"""
import numpy as np
import soundfile as sf
from .aac_coder_3 import aac_coder_3, i_aac_coder_3


def demo_aac_3(filename_in, filename_out, filename_aac_coded):
    """
    Demo AAC Level 3 - Lossy compression
    
    Kodikopoiei kai apokod ikopoiei ixo, ypologizei SNR, bitrate, compression
    
    Anamename xami les times SNR logo apo lon kvantismou
    kai i diafora input-output tha einai aistiti
    """
    # Fortwsh arxikou hxou
    audio_orig, fs = sf.read(filename_in, dtype='float64')
    
    n_samples, duration = len(audio_orig), len(audio_orig) / fs     # duration = n_samples / fs se deuterolepta
    
    # Kodikwpoihsh + Apokwdikopihsh
    aac_seq_3 = aac_coder_3(filename_in, filename_aac_coded)
    audio_decoded = i_aac_coder_3(aac_seq_3, filename_out)
    
    # Rithmisi mikous gia sigkrisi
    if len(audio_decoded) > n_samples:
        audio_decoded = audio_decoded[:n_samples, :]
    elif len(audio_decoded) < n_samples:
        audio_decoded = np.vstack([audio_decoded, np.zeros((n_samples - len(audio_decoded), 2))])
    
    # Ypologismos SNR 
    signal_power = np.sum(audio_orig ** 2)
    noise_power = np.sum((audio_orig - audio_decoded) ** 2)
    if noise_power > 1e-12:
        SNR = 10.0 * np.log10(signal_power / noise_power)
    else:
        SNR = np.inf
    
    # Ypologismos bitrate
    total_bits = 0
    for frame in aac_seq_3:
        total_bits += 3  # frame_type (2) + win_type (1)
        for ch_name in ["chl", "chr"]:
            ch_data = frame[ch_name]
            # TNS coeffs (4 syntelestes x 4 bits kathe enas)
            total_bits += (4*4*8 if frame["frame_type"] == "ESH" else 4*4)
            # Global gain
            total_bits += 8 * (len(ch_data["G"]) if isinstance(ch_data["G"], np.ndarray) else 1)
            # Scale factors + stream + codebook
            total_bits += len(ch_data["sfc"]) + len(ch_data["stream"]) + 4
    
    bitrate = total_bits / duration
    original_bitrate = 2 * 16 * fs  # stereo, 16-bit
    compression = original_bitrate / bitrate
    
    return SNR, bitrate, compression
