"""
Level 2 - Kwdikopoihths / Apokwdikopoihths AAC me TNS

Ylopoiei:
    aac_seq_2 = aac_coder_2(filename_in)
    x = i_aac_coder_2(aac_seq_2, filename_out)
"""

import numpy as np
import soundfile as sf

from .SSC import SSC
from .filter_bank import filter_bank, i_filter_bank
from .TNS import tns, i_tns


# Statheres
FRAME_LEN = 2048
HOP = 1024
FS_EXPECTED = 48000
CHANNELS = 2



def _prepare_signal(x):
    """Prosthetei padding sto shma gia swsth diairesh se frames"""
    x = np.pad(x, ((HOP, HOP), (0, 0)), mode="constant")
    
    n = x.shape[0]
    rem = (n - FRAME_LEN) % HOP
    if rem > 0:
        x = np.pad(x, ((0, HOP - rem), (0, 0)), mode="constant")
    
    return x


def aac_coder_2(filename_in):
    """
    Kodikopitis AAC Level 2: SSC + Filterbank + TNS
    Epistrefei lista apo frames
    """
    x, fs = sf.read(filename_in, always_2d=True)

    x = x.astype(np.float64, copy=False)
    x = _prepare_signal(x)

    num_frames = (x.shape[0] - FRAME_LEN) // HOP + 1
    aac_seq = []
    prev_frame_type = "OLS"

    for i in range(num_frames):
        start = i * HOP
        frame_t = x[start:start + FRAME_LEN, :]

        # Diavazei to epomeno frame gia to SSC
        if i < num_frames - 1:
            next_frame_t = x[start + HOP:start + HOP + FRAME_LEN, :]
        else:
            next_frame_t = np.zeros((FRAME_LEN, CHANNELS))

        frame_type = SSC(frame_t, next_frame_t, prev_frame_type)

        # Filterbank
        frame_f = filter_bank(frame_t, frame_type)

        # Efarmogi TNS ana kanali
        if frame_type == "ESH":
            # Gia ESH: reshape se (128, 8) gia to TNS
            chl_in = frame_f[:, 0].reshape((8, 128)).T  # (128, 8)
            chr_in = frame_f[:, 1].reshape((8, 128)).T
            
            chl_tns, chl_coeffs = tns(chl_in, frame_type)
            chr_tns, chr_coeffs = tns(chr_in, frame_type)
            
            chl_frame_f = chl_tns
            chr_frame_f = chr_tns
        else:
            # Gia long frames
            chl_in = frame_f[:, 0:1]  # (1024, 1)
            chr_in = frame_f[:, 1:2]
            
            chl_tns, chl_coeffs = tns(chl_in, frame_type)
            chr_tns, chr_coeffs = tns(chr_in, frame_type)
            
            chl_frame_f = chl_tns
            chr_frame_f = chr_tns

        aac_seq.append({"frame_type": frame_type, "chl": {"tns_coeffs": chl_coeffs, "frame_F": chl_frame_f},
            "chr": {"tns_coeffs": chr_coeffs, "frame_F": chr_frame_f }})

        prev_frame_type = frame_type

    return aac_seq


def i_aac_coder_2(aac_seq, filename_out):
    """
    Apokodikopitis AAC Level 2: inverse TNS + iFilterbank + overlap-add
    Epistrefei to apokodikopoimeno shma
    """
    num_frames = len(aac_seq)
    out_len = (num_frames - 1) * HOP + FRAME_LEN
    y = np.zeros((out_len, CHANNELS))

    for i, frame in enumerate(aac_seq):
        frame_type = frame["frame_type"]

        # Anagnwrish dedomenwn TNS
        chl_tns_out = np.asarray(frame["chl"]["frame_F"])
        chr_tns_out = np.asarray(frame["chr"]["frame_F"])
        
        chl_coeffs = np.asarray(frame["chl"]["tns_coeffs"])
        chr_coeffs = np.asarray(frame["chr"]["tns_coeffs"])

        # Efarmogh inverse TNS
        if frame_type == "ESH":
            chl_f = i_tns(chl_tns_out, frame_type, chl_coeffs)  # (128, 8)
            chr_f = i_tns(chr_tns_out, frame_type, chr_coeffs)
            
            # Flatten se (1024,) me Fortran order (column-major)
            chl_f_flat = chl_f.flatten(order="F")
            chr_f_flat = chr_f.flatten(order="F")
        else:
            # Long frames (1024, 1)
            chl_f = i_tns(chl_tns_out, frame_type, chl_coeffs)
            chr_f = i_tns(chr_tns_out, frame_type, chr_coeffs)
            
            chl_f_flat = chl_f.flatten()
            chr_f_flat = chr_f.flatten()

        # Anakataskeuh frame (1024, 2) gia to iFilterbank
        frame_f = np.column_stack([chl_f_flat, chr_f_flat])

        # Inverse Filterbank
        frame_t = i_filter_bank(frame_f, frame_type)

        # Overlap-add
        start = i * HOP
        y[start:start + FRAME_LEN, :] += frame_t

    # Afairesh padding
    if y.shape[0] >= 2 * HOP:
        y = y[HOP:-HOP, :]

    sf.write(filename_out, y, FS_EXPECTED, subtype="FLOAT")
    return y
