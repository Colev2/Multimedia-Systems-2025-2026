"""
Level 1 - Kwdikopoihths / Apokwdikopoihths AAC

Ylopoiei:
    aac_seq_1 = aac_coder_1(filename_in)
    x = i_aac_coder_1(aac_seq_1, filename_out)
"""

import numpy as np
import soundfile as sf

from .SSC import SSC
from .filter_bank import filter_bank, i_filter_bank


# Statheres gia to Level 1
FRAME_LEN = 2048
HOP = 1024
FS_EXPECTED = 48000
CHANNELS = 2


def prepare_signal(x):
    """Prosthetei padding sto shma gia swsth diairesh se frames"""
    # Padding 1024 samples sthn arxh kai sto telos gia 
    x = np.pad(x, ((HOP, HOP), (0, 0)), mode="constant")

    # Padding sto telos gia akrivh diairesh
    n = x.shape[0]
    rem = (n - FRAME_LEN) % HOP
    if rem > 0:
        x = np.pad(x, ((0, HOP - rem), (0, 0)), mode="constant")

    return x


def aac_coder_1(filename_in):
    """
    Kwdikopoihths AAC Level 1: SSC + Filterbank 
    Epistrefei lista apo frames
    """
    x, fs = sf.read(filename_in, always_2d=True)    # Diavazei to wav, epistrefei pinaka me diastaseis (samples, channels) 

    x = x.astype(np.float64, copy=False)
    x = prepare_signal(x)

    num_frames = (x.shape[0] - FRAME_LEN) // HOP + 1    # Vriskei to plhthos twn frames
    aac_seq = []
    prev_frame_type = "OLS"     # Arxikopoihsh tou prohgoumenou frame type gia to SSC, arxika OLS

    for i in range(num_frames):   # gia kathe frame, ypologise to frame type me SSC kai tous syntelestes MDCT me filter bank
        start = i * HOP
        frame_t = x[start:start + FRAME_LEN, :]     # trexon frame

        # Diavazei to epomeno frame gia to SSC
        if i < num_frames - 1:      # An den einai to teleutaio frame
            next_frame_t = x[start + HOP:start + HOP + FRAME_LEN, :]     # epomeno frame
        else:
            next_frame_t = np.zeros((FRAME_LEN, CHANNELS))  # alliws, orizw to epomeno frame 0

        frame_type = SSC(frame_t, next_frame_t, prev_frame_type)      # Pare to trexon frame type me to SSC
        frame_f = filter_bank(frame_t, frame_type)          # Pare tous MDCT syntelestes me to filter bank

        # Apothhkeush analoga me ton typo
        if frame_type == "ESH":
            chl = frame_f[:, 0].reshape((8, 128)).T  # Aristero kanali (128, 8)
            chr = frame_f[:, 1].reshape((8, 128)).T  # Deksio kanali (128, 8)
        else:
            chl = frame_f[:, 0:1]  # (1024, 1)
            chr = frame_f[:, 1:2]  # (1024, 1)

        aac_seq.append({"frame_type": frame_type, "chl": {"frame_F": chl}, "chr": {"frame_F": chr}})

        prev_frame_type = frame_type

    return aac_seq    # Epistrefei lista apo frames, opou kathe frame einai lexiko me to frame type kai tous syntelestes MDCT gia kathe kanali


def i_aac_coder_1(aac_seq, filename_out):
    """
    Apokodikopitis AAC Level 1: iFilterbank (KBD) + overlap-add
    Epistrefei to apokwdikopoihmeno shma
    """
    num_frames = len(aac_seq)
    out_len = (num_frames - 1) * HOP + FRAME_LEN
    y = np.zeros((out_len, CHANNELS))

    for i, frame in enumerate(aac_seq):
        frame_type = frame["frame_type"]        # Pare to frame type apo to aac_seq

        # Anakataskeuh tou frame sto pedio syxnothtas
        frame_f = np.zeros((HOP, CHANNELS))

        if frame_type == "ESH":
            chl = np.asarray(frame["chl"]["frame_F"])   # Anakthsh tou frame gia to aristero kanali: (128, 8)
            chr = np.asarray(frame["chr"]["frame_F"])   # Anakthsh tou frame gia to deksio kanali: (128, 8)

            frame_f[:, 0] = chl.flatten(order="F")      # Metatropi se (1024,) me flatten order="F" gia na diatirhthei h seira twn syntelestwn
            frame_f[:, 1] = chr.flatten(order="F")
        else:
            frame_f[:, 0] = np.asarray(frame["chl"]["frame_F"]).reshape(HOP)
            frame_f[:, 1] = np.asarray(frame["chr"]["frame_F"]).reshape(HOP)

        # Epistrofh sto pedio xronou
        frame_t = i_filter_bank(frame_f, frame_type)    # 

        # Overlap-add
        start = i * HOP
        y[start:start + FRAME_LEN, :] += frame_t

    # Afairesh padding
    if y.shape[0] >= 2 * HOP:
        y = y[HOP:-HOP, :]

    sf.write(filename_out, y, FS_EXPECTED, subtype="FLOAT")
    return y
