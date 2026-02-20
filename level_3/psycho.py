"""
Level 3 - Psychoacoustic Model
Psyxoakoustiko Montelo - Ypologismos katoflio n akoustotitas

Ypologizei poso thorivo kvantismou epitrepetai se kathe syxnotita
xoris na akoustetai apo ton anthropino afti.
"""
import numpy as np
from scipy.io import loadmat
import os

# Kagolikoi pinakes (fortono mia fora)
BANDS_LONG = None
BANDS_SHORT = None
SPREADING_LONG = None   # Pinakas diasporas gia long
SPREADING_SHORT = None  # Pinakas diasporas gia short

# Debug: Teleutaia energeia e(b) apo FFT (gia sygkrish me P(b) ston quantizer)
LAST_PSYCHO_E = None
LAST_PSYCHO_NPART = None


def load_bands():
    """Fortosi pinaka bands apo TableB219.mat"""
    global BANDS_LONG, BANDS_SHORT
    if BANDS_LONG is not None:
        return
    
    mat_data = loadmat("TableB219.mat")
    BANDS_LONG = mat_data['B219a']   # 69 bands gia long
    BANDS_SHORT = mat_data['B219b']  # 42 bands gia short


def spreading_fun(i, j, bval):
    """
    Orismata:
    i, j : Deiktes twn band i kai j
    bval : Pinakas me tis kentrikes syxnotites kathe band

    Synarthsh diasporas maskas (spreading function)
    Ypologizei syntelesth pou ekfrazei to poso h energeia enos hxou se mia banda syxnothtwn (band i) 
    maskarei - kalyptei ton hxo se alles bandes (band j)

    Epistrefei:
    Spreading function value gia ta band i kai j
    """
    if i >= j:
        tmpx = 3.0 * (bval[j] - bval[i])
    else:
        tmpx = 1.5 * (bval[j] - bval[i])
    tmpz = 8.0 * min((tmpx - 0.5)**2 - 2.0 * (tmpx - 0.5), 0.0)
    tmpy = 15.811389 + 7.5 * (tmpx + 0.474) - 17.5 * np.sqrt(1.0 + (tmpx + 0.474)**2)
    if tmpy < -100.0:
        return 0.0
    else:
        return 10.0 ** ((tmpz + tmpy) / 10.0)   # timh spreading function gia ta band i kai j 


def precompute_spreading():
    """Proypologismos pinaka spreading function gia taxythta"""
    global SPREADING_LONG, SPREADING_SHORT
    if SPREADING_LONG is not None:
        return
    
    load_bands()
    
    for bands_data, n_type in [(BANDS_LONG, 'long'), (BANDS_SHORT, 'short')]:
        bval = bands_data[:, 3]  # Kentriki syxnotita kathe band
        n_bands = len(bval)
        
        # Dimiourgia pinaka spreading function (n_bands x n_bands)
        spreading = np.zeros((n_bands, n_bands))
        for i in range(n_bands):
            for j in range(n_bands):       # Gia kathe zeugari band i,j
                spreading[i, j] = spreading_fun(i, j, bval)   # Pinakas (n_bands, n_bands) me times 
                                                              # spreading function[i][j] gia kathe zeugos i,j
        
        if n_type == 'long':    # Gia long frames
            SPREADING_LONG = spreading      # 69x69
        else:                   # Gia short frames
            SPREADING_SHORT = spreading    # 42x42


# Psyxoakoustiko montelo gia ena frame/subframe

def psycho_subframe(s, s_prev1, s_prev2, frame_type):
    """
    Orismata:
    s : Trexon frame/subframe sto pedio xronou
    s_prev1 : Prohgoumeno frame
    s_prev2 : Pro-prohgoumeno frame
    frame_type : Typos frame ("OLS", "ESH", "LSS", "LPS")
    
    Psyxoakoustiko montelo gia ena frame/subframe
    
    Vimata:
    1. FFT me parathyro Hann
    2. Provlepsi apo proigoumena frames
    3. Ypologismos provlepsimotitas
    4. Ypologismos energeias ana band
    5. Efarmogi spreading function
    6. Ypologismos tonality index
    7. Ypologismos SNR kai katoflio
    8. Ypologismos telikou SMR

    Epistrefei:

    SMR : Signal-to-Mask Ratio gia kathe band, me sxhma: (69,1) gia long frames, (42,8) gia short frames (8 subframes)
    """
    global LAST_PSYCHO_E
    load_bands()
    
    # Arxikopoihsh bands kai spreading pinakwn analoga me to frame type
    if frame_type in ("OLS", "LSS", "LPS"):
        bands, spreading, N = BANDS_LONG, SPREADING_LONG, 2048
    else:
        bands, spreading, N = BANDS_SHORT, SPREADING_SHORT, 256
    
    w_low, w_high, qsthr = bands[:, 1].astype(int), bands[:, 2].astype(int), bands[:, 4]  # Anathesi w_low, w_high, qsthr apo ton pinaka bands
    n_bands = len(bands)
    
    # 1. Parathyro Hann kai FFT
    hann = 0.5 - 0.5 * np.cos(np.pi * (np.arange(N) + 0.5) / N)
    
    def get_fft(signal):
        S = np.fft.fft(signal * hann)   # FFT tou s_w(n) = s(n)*hann(n)
        return np.abs(S[:N//2]), np.angle(S[:N//2])  # Epistrefei r (eneregia) kai f (fasi) gia syxnothtes 0..N/2-1
    
    r, f = get_fft(s)   # Energeia kai fasi tou trexontos frame
    r_prev1, f_prev1 = get_fft(s_prev1) # Energeia kai fasi tou prohgoumenou frame
    r_prev2, f_prev2 = get_fft(s_prev2) # Energeia kai fasi tou pro-prohgoumenou frame
    
    # 2. Provlepsi
    r_pred = 2.0 * r_prev1 - r_prev2    # Vhma 3 ekfwnhshs: Provlepsh gia energeia
    f_pred = 2.0 * f_prev1 - f_prev2    # Provlepsh gia fash
    
    # 3. Provlepsimotita (predictability)
    numerator = np.sqrt((r * np.cos(f) - r_pred * np.cos(f_pred))**2 + (r * np.sin(f) - r_pred * np.sin(f_pred))**2)
    c = numerator / (r + np.abs(r_pred) + 1e-12)    # Gia apofygh diaireshs me 0, prosthetw 1e-12
    
    # 4. Energeia ana band
    e = np.zeros(n_bands)
    for b in range(n_bands):
        e[b] = np.sum(r[w_low[b]:w_high[b]+1]**2)   # e[b]: dianysma (n_bands,) -> (69,) gia long frames kai (42,) gia short
    
    # Ypologismos vevarymenis provlepsimotitas ana band
    c_band = np.zeros(n_bands)
    for b in range(n_bands):
        c_band[b] = np.sum(c[w_low[b]:w_high[b]+1] * r[w_low[b]:w_high[b]+1]**2)    # c_band[b]: dianysma (n_bands,)
    
    # 5. Efarmogi spreading function
    ecb = spreading.T @ e       # dianysma (n_bands,) me ecb[b] = Σ spreading[i][b] * e[i] gia i=0..n_bands-1
    ct = spreading.T @ c_band   # dianysma (n_bands,) me ct[b] = Σ spreading[i][b] * c_band[i] gia i=0..n_bands-1
    
    cb = np.clip(ct / (ecb + 1e-12), 1e-12, 1.0)    # clip gia na mpei sto logaritmo 
    en = ecb / (np.sum(spreading, axis=0) + 1e-12)
    
    # 6. Tonality index (0=thorivos, 1=tonos)
    tb = np.clip(-0.299 - 0.43 * np.log(cb), 0.0, 1.0)  # tb anhkei (0,1)
    
    # 7. Apaitoumeno SNR ana band 
    # SNR(b) = tb(b)*TMN + (1-tb(b))*NMT
    # tb->1 -> exoume tono: SNR-> TMN=18dB 
    # tb->0 -> exoume thoryvo: SNR-> NMT=6dB 
    SNR_db = tb * 18.0 + (1.0 - tb) * 6.0   # SNR dianysma (n_bands,) gia kathe band se dB

    # Metatroph 
    bc = 10.0 ** (-SNR_db / 10.0)

    # Katwfli energeias gia kathe band
    nb = en * bc        # Dianysma (n_bands,)
    
    # Katofli se isyxia
    qhat_thr = np.finfo(float).eps * N * 10.0 ** (qsthr / 10.0)

    # Katwfli akoustothtas thoryvou kvantismou
    npart = np.maximum(nb, qhat_thr)
    
    # Apothikeush e kai npart gia debug (sygkrish me P(b) ston quantizer)
    global LAST_PSYCHO_E, LAST_PSYCHO_NPART
    LAST_PSYCHO_E = e.copy()
    LAST_PSYCHO_NPART = npart.copy()
    
    # 8. Signal-to-Mask Ratio
    return e / (npart + 1e-12)  # Epistrefei SMR(b): (n_bands,)



def psycho(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2):
    """
    Psyxoakoustiko Montelo 
    
    Ypologizei to Signal-to-Mask Ratio (SMR) gia kathe band.
    To SMR mas leei poso thorivo kvantismou mporoume na exoume xoris na akougetai.
    
    Parametroi:
    frame_T : Trexon frame
    frame_type : Typos frame ("OLS", "ESH", "LSS", "LPS")
    frame_T_prev_1 : Proigoumeno frame
    frame_T_prev_2 : Pro-proigoumeno frame
    
    Epistrefei:
    SMR : Signal-to-Mask Ratio gia kathe band
    """
    frame_type = frame_type.upper()
    precompute_spreading()
    
    # Gia ESH frames, epexergazomaste 8 subframes
    if frame_type == "ESH":
        center_curr = frame_T[448:1600]
        center_prev1 = frame_T_prev_1[448:1600]
        center_prev2 = frame_T_prev_2[448:1600]
        
        SMR = np.zeros((42, 8))
        for s in range(8):
            start = s * 128
            SMR[:, s] = psycho_subframe(center_curr[start:start+256], center_prev1[start:start+256], center_prev2[start:start+256], "ESH")
        return SMR  # Epistrefei SMR gia kathe subframe se morfh (42,8)
    
    # Gia long frames
    return psycho_subframe(frame_T, frame_T_prev_1, frame_T_prev_2, frame_type).reshape(-1, 1)  # Epistrefei SMR gia long frames me sxhma (69,1)




