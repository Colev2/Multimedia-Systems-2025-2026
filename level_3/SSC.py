"""
Level 1 - Sequence Segmentation Control (SSC)
Elegxos diamerishs akolouthias - epilogh typou frame

Ylopoiei:
    frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
"""

import numpy as np
from scipy.signal import lfilter


# Syntelestes high-pass filtrou
_HP_B = np.array([0.7548, -0.7548])
_HP_A = np.array([1.0, -0.5095])


def _is_next_esh(next_frame_ch):
    """
    Elegxei an to epomeno frame einai ESH, ypologizontas attack values me high-pass filtro
    kai elegxontas tis synthhkes
    
    Epistrefei True an to epomeno frame einai ESH, alliws False
    """
    # High-pass filtering
    xf = lfilter(_HP_B, _HP_A, next_frame_ch)

    # Xwrismos se 8 perioxes twn 128 samples kai ypologismos energeias
    energies = np.empty(8)
    for l in range(8):
        seg = xf[l * 128:(l + 1) * 128]   # seg: tmhma tou xf me 128 samples
        energies[l] = np.sum(seg * seg)   # energies[l]: energeia tou tmhmatos seg
                                          # np.sum(seg * seg) element wise pol/smos kai prosthesh = energeia

    # Ypologismos attack values
    for l in range(1, 8):
        prev_mean = np.mean(energies[:l])      # Mesos oros energeias twn prohgoumenwn l tmhmatwn
        
        if prev_mean == 0:  # An o mesos oros einai 0, dhladh h energeia sta prohgoumena tmhmata einai 0, 
                            # dhladh exoume hsyxia : 
            if energies[l] > 0:     # An to tmhma l exei energeia, tote apo hsyxia apotoma hxo, dhladh
                                    # exoume megisto attack, ara attack = apeiro
                ds2 = np.inf
            else:          # Alliws to tmhma l den exei energeia, dhladh exei epishs hsyxia, ara orizw attack = 0
                ds2 = 0.0
        else:
            ds2 = energies[l] / prev_mean   # Attack value gia to tmhma l

        # An yparxei megalo attack, to epomeno frame einai ESH
        if (energies[l] > 1e-3) and (ds2 > 10):
            return True

    return False


# Pinakas 1: Syndyasmos typwn apo ta 2 kanalia kai telikh kathgoriopoihsh
_COMBINE_TABLE = {("OLS", "OLS"): "OLS", ("OLS", "LSS"): "LSS", ("OLS", "ESH"): "ESH", ("OLS", "LPS"): "LPS", ("LSS", "OLS"): "LSS",
    ("LSS", "LSS"): "LSS", ("LSS", "ESH"): "ESH", ("LSS", "LPS"): "ESH", ("ESH", "OLS"): "ESH", ("ESH", "LSS"): "ESH",
    ("ESH", "ESH"): "ESH", ("ESH", "LPS"): "ESH", ("LPS", "OLS"): "LPS", ("LPS", "LSS"): "ESH", ("LPS", "ESH"): "ESH",
    ("LPS", "LPS"): "LPS"}


def SSC(frame_T, next_frame_T, prev_frame_type):
    """
    Sequence Segmentation Control
    Epilegei ton typo tou trexontos frame (OLS, LSS, ESH, LPS)
    
    Parametroi:

    frame_T : To trexon frame sto pedio xronou (2048, 2)
    next_frame_T : To epomeno frame sto pedio xronou (2048, 2)
    prev_frame_type : O typos tou proigoumenou frame
    
    Epistrefei:

    str : O typos tou trexontos frame
    """
    # Elegxos gia ta 2 kanalia an to epomeno frame tha einai ESH
    next_esh_ch0 = _is_next_esh(next_frame_T[:, 0])
    next_esh_ch1 = _is_next_esh(next_frame_T[:, 1])

    # Apofash tou trexon frame gia kathe kanali ksexwrista
    def decide(prev, next_is_esh):
        if prev == "OLS" :      # An to prohgoumeno frame einai OLS
            if next_is_esh:     # Kai to epomeno einai ESH
                return "LSS"    # To trexon frame einai LSS
            else:
                return "OLS"    # Alliws an to epomeno den einai ESH, to trexon frame einai OLS
        elif prev == "ESH":     # Omoiws
            if next_is_esh:
                return "ESH"
            else:
                return "LPS"
        elif prev == "LSS":
            return "ESH"
        else:       # prev == "LPS"
            return "OLS"  

    ch0_type = decide(prev_frame_type, next_esh_ch0)    # Apofash tou trexon frame type gia to aristero kanali
    ch1_type = decide(prev_frame_type, next_esh_ch1)    # Apofash tou trexon frame type gia to deksio kanali

    # Syndyasmos typwn twn 2 kanaliwn gia telikh kathgoriopoihsh tou trexontos frame type me vash ton Pinaka 1
    frame_type = _COMBINE_TABLE[(ch0_type, ch1_type)]
    return frame_type   # Epestrepse ton teliko typo tou trexontos frame 