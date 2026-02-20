"""
Level 3 - Quantizer
Kvantisi kai Apokvantisi synteleston MDCT

O kvantistis xrisimopoiei to psyxoakoustiko montelo (SMR)
gia na elaxistopoiisei to akoisto sfalma.
"""
import numpy as np
from scipy.io import loadmat
import os

# Katholikoi pinakes bands
BANDS_LONG = None
BANDS_SHORT = None

# Debug statistics (global)
LAST_DEBUG_STATS = {
    "alpha_min": 0.0,
    "alpha_max": 0.0,
    "term_iter": 0,
    "termination": "none",
    "smr_min": 0.0,
    "smr_max": 0.0,
    "T_min": 0.0,
    "T_max": 0.0,
    "Pe_initial_min": 0.0,
    "Pe_initial_max": 0.0,
    "P_min": 0.0,
    "P_max": 0.0,
    "P_over_e_min": 0.0,
    "P_over_e_max": 0.0,
    "npart_min": 0.0,
    "npart_max": 0.0,
    # Specific band debug (band 20)
    "T_band20": 0.0,
    "P_band20": 0.0,
    "SMR_band20": 0.0,
    "Pe_band20": 0.0
}


def get_last_quantizer_stats():
    """Epistrefei ta teleutaia debug statistics apo ton quantizer"""
    return LAST_DEBUG_STATS.copy()


def load_bands():
    """Fortwsh pinaka bands apo TableB219.mat"""
    global BANDS_LONG, BANDS_SHORT
    if BANDS_LONG is not None:
        return
    
    mat_data = loadmat("TableB219.mat")
    BANDS_LONG = mat_data['B219a']   # 69 bands gia long
    BANDS_SHORT = mat_data['B219b']  # 42 bands gia short


def quantize_mdct(X, alpha):
    """
    Orismata:
    X : Syntelestes MDCT gia ena frame/subframe
    alpha : Scale factor gia to band

    Kvantish MDCT - Non-uniform quantizer
    Typos: S(k) = sign(X) * floor(|X * 2^(-a/4)|^0.75 + 0.4054)

    Epistrefei:
    S : Kvantismenoi syntelestes 
    """
    scale = 2.0 ** (-alpha / 4.0)
    return (np.sign(X) * np.floor(np.abs(X * scale) ** 0.75 + 0.4054)).astype(np.int32)  # dianysma (1024,) gia long frames, 
                                                                    # (128,) gia short frames me kvantismena S(k) apo X(k) kai alpha



def dequantize_mdct(S, alpha):
    """
    Orismata:
    S : Kvantismenoi syntelestes MDCT
    alpha : Scale factor gia to band

    Apokvantisi MDCT
    Typos: X(k) = sign(S) * |S|^(4/3) * 2^(a/4)

    Epistrefei:
    X : Apokvantismenh timh MDCT
    """
    scale = 2.0 ** (alpha / 4.0)
    return np.sign(S) * (np.abs(S) ** (4.0 / 3.0)) * scale  


def quantize_subframe(X, frame_type, SMR):
    """
    Orismata:
    X : Syntelestes MDCT gia ena frame/subframe
    frame_type : Typos frame
    SMR : Signal-to-Mask Ratio gia to frame/subframe
    Kvantisi enos frame/subframe
    
    Vimata:
    1. Ypologismos katoflio (thresholds) apo SMR
    2. Arxi ki ektimisi alpha (scale factors)
    3. Epanaliptiki veltistopoiisi alpha
    4. DPCM encoding ton scale factors

    Epistrefei:
    S : Kvantismenoi syntelestes pou petyxainoun th megisth sumpiesh
    sfc : Scale factors (DPCM encoded)
    G : Global gain
    """
    global LAST_DEBUG_STATS
    load_bands()
    
    # Epilogi bands analoga me ton typo
    if frame_type in ("OLS", "LSS", "LPS"):
        bands = BANDS_LONG
    else:
        bands = BANDS_SHORT
    w_low, w_high = bands[:, 1].astype(int), bands[:, 2].astype(int)
    n_bands = len(w_low)    # 69 gia long, 42 gia short
    
    # 1. Ypologismos energeias kai threshold gia kathe band (vima 13 psyxoakoustikou montelou)
    # P(b) = Σ X(k)^2 gia k apo w_low(b) mexri w_high(b)
    # T(b) = P(b) / SMR(b)
    P = np.zeros(n_bands)
    T = np.zeros(n_bands)
    for b in range(n_bands):
        P[b] = np.sum(X[w_low[b]:w_high[b]+1]**2)
        T[b] = P[b] / (SMR[b] + 1e-12)
    
    # Apothikeush SMR, P kai T statistics
    LAST_DEBUG_STATS["smr_min"] = float(np.min(SMR))
    LAST_DEBUG_STATS["smr_max"] = float(np.max(SMR))
    LAST_DEBUG_STATS["P_min"] = float(np.min(P))
    LAST_DEBUG_STATS["P_max"] = float(np.max(P))
    LAST_DEBUG_STATS["T_min"] = float(np.min(T))
    LAST_DEBUG_STATS["T_max"] = float(np.max(T))
    
    # Apothikeush gia specific band (band 20) gia elegxo tis sxesis T = P/SMR
    if n_bands > 20:
        LAST_DEBUG_STATS["SMR_band20"] = float(SMR[20])
        LAST_DEBUG_STATS["P_band20"] = float(P[20])
        LAST_DEBUG_STATS["T_band20"] = float(T[20])
    
    # Sygkrish P(b) me e(b) apo psycho gia elegxo klimakas
    try:
        from .psycho import LAST_PSYCHO_E, LAST_PSYCHO_NPART
        if LAST_PSYCHO_E is not None and len(LAST_PSYCHO_E) == n_bands:
            P_over_e = P / (LAST_PSYCHO_E + 1e-12)
            LAST_DEBUG_STATS["P_over_e_min"] = float(np.min(P_over_e))
            LAST_DEBUG_STATS["P_over_e_max"] = float(np.max(P_over_e))
        if LAST_PSYCHO_NPART is not None and len(LAST_PSYCHO_NPART) == n_bands:
            LAST_DEBUG_STATS["npart_min"] = float(np.min(LAST_PSYCHO_NPART))
            LAST_DEBUG_STATS["npart_max"] = float(np.max(LAST_PSYCHO_NPART))
    except:
        pass
    
    # 2. Arxikh ektimhsh alpha  
    # α̂(b) = 16/3 * log2(max_k(X(k))^0.75 / 8191) gia ola ta b
    X_max = np.max(np.abs(X))
    if X_max > 1e-12:
        alpha_hat = (16.0/3.0) * np.log2((X_max**0.75) / 8191)
    else:
        alpha_hat = 0.0
    alpha = np.full(n_bands, alpha_hat)

    term_iter = 0
    termination = "max_iter"
    
    # 3. Epanaliptiki auxhsh alpha (vima 2)
    # Auxanoume alpha mexri Pe(b) >= T(b) i |a(b+1)-a(b)| > 60
    for iteration in range(60):  # Max iterations gia apofygh infinite loop
        # Kvantisi me ta trexonta alpha

        # Arxikopoihsh S kai X_hat
        S = np.zeros(len(X), dtype=np.int32)    
        X_hat = np.zeros(len(X))
        
        for b in range(n_bands):  # Kvantish kai apokvantish gia kathe band me to trexonta alpha[b], gia ypologismo isxyos Pe(b)
            S[w_low[b]:w_high[b]+1] = quantize_mdct(X[w_low[b]:w_high[b]+1], alpha[b])  
            X_hat[w_low[b]:w_high[b]+1] = dequantize_mdct(S[w_low[b]:w_high[b]+1], alpha[b])   
        
        # Elegxos kai auxhsh alpha gia kathe band
        all_done = True
        Pe_values = np.zeros(n_bands)  # Gia apothikeush Pe gia kathe band
        
        for b in range(n_bands):
            # Ypologismos isxyos sfalamtos kvantismou
            Pe = np.sum((X[w_low[b]:w_high[b]+1] - X_hat[w_low[b]:w_high[b]+1])**2)
            Pe_values[b] = Pe   # Apothikeush Pe tou band b gia debug

            # Apothikeush arxikhs Pe gia band 20
            if iteration == 0 and b == 20 and n_bands > 20:
                LAST_DEBUG_STATS["Pe_band20"] = float(Pe)
            
            # Apothikeush arxikwn Pe values 
            if iteration == 0:
                if b == 0:
                    LAST_DEBUG_STATS["Pe_initial_min"] = float(Pe)
                    LAST_DEBUG_STATS["Pe_initial_max"] = float(Pe)
                else:
                    LAST_DEBUG_STATS["Pe_initial_min"] = min(LAST_DEBUG_STATS["Pe_initial_min"], float(Pe))
                    LAST_DEBUG_STATS["Pe_initial_max"] = max(LAST_DEBUG_STATS["Pe_initial_max"], float(Pe))
            
            # An Pe < T(b), theloume na auxhsoume to alpha[b]
            if Pe < T[b]:
                # Elegxos: mporoume na to auxhsoume xwris na paraviasoume ton periorismo 60?
                can_increase = True
                
                
                # Elegxos me prohgoumenh banda (b-1)
                if b > 0:
                    if abs((alpha[b] + 1) - alpha[b-1]) > 60:   # An auxhsh tou alpha[b] kata 1 paraviazei th synthiki
                        can_increase = False    # De mporoume na auxhsoume to alpha[b]  

                
                # Elegxos me epomenh banda (b+1) 
                if b < n_bands - 1:
                    if abs(alpha[b+1] - (alpha[b] + 1)) > 60:
                        can_increase = False    
                
                # Auxhsh alpha an epitrepetai
                if can_increase:
                    alpha[b] += 1
                    all_done = False
    
        # Den eixame allagh sta alpa se kanena band, opote exoume th veltisth sympiesh
        if all_done:
            term_iter = iteration
            termination = "Converged"
            break
        
        term_iter = iteration
       
    # 4. Kvantish me ta telika alpha
    S = np.zeros(len(X), dtype=np.int32)
    for b in range(n_bands):
        S[w_low[b]:w_high[b]+1] = quantize_mdct(X[w_low[b]:w_high[b]+1], alpha[b])
    
    # 5. DPCM encoding twn scale factors
    G = int(np.round(alpha[0]))
    sfc = np.zeros(n_bands, dtype=np.int32)
    sfc[0] = G
    for b in range(1, n_bands):
        sfc[b] = int(np.round(alpha[b] - alpha[b-1]))
    
    # Enimerosi debug statistics
    LAST_DEBUG_STATS["alpha_min"] = alpha.min()
    LAST_DEBUG_STATS["alpha_max"] = alpha.max()
    LAST_DEBUG_STATS["term_iter"] = int(term_iter)
    LAST_DEBUG_STATS["termination"] = termination

    return S, sfc, G


def dequantize_subframe(S, sfc, G, frame_type):
    """
    Orismata:
    S : Kvantismenoi syntelestes
    sfc : Scale factors (DPCM encoded)
    G : Global gain
    frame_type : Typos frame
    
    Apokvantisi enos frame/subframe
    
    Vimata:
    1. Anakataskeyi alpha apo DPCM (inverse DPCM)
    2. Apokvantisi kathe band me to antistoixo alpha
    """
    load_bands()
    
    # Epilogi bands
    if frame_type in ("OLS", "LSS", "LPS"):
        bands = BANDS_LONG
    else:
        bands = BANDS_SHORT
    w_low, w_high = bands[:, 1].astype(int), bands[:, 2].astype(int)
    n_bands = len(w_low)
    
    # 1. Anakataskeyi alpha (inverse DPCM)
    # a[0] = G, a[b] = a[b-1] + sfc[b]
    alpha = np.zeros(n_bands)
    alpha[0] = G
    for b in range(1, n_bands):
        alpha[b] = alpha[b-1] + sfc[b]
    
    # 2. Apokvantisi kathe band
    X = np.zeros(len(S))
    for b in range(n_bands):
        X[w_low[b]:w_high[b]+1] = dequantize_mdct(S[w_low[b]:w_high[b]+1], alpha[b])
    
    return X    # Epistrefei ta apokvantismena X(k) gia to frame/subframe 




def aac_quantizer(frame_F, frame_type, SMR):
    """
    AAC Quantizer - Kvantisi synteleston MDCT
    
    Kvantizei tous syntelestes MDCT xrisimopoiontas to psyxoakoustiko
    montelo (SMR) gia na elaxistopoiisei to akoisto sfalma.
    
    Orismata:

    frame_F : Syntelestes MDCT
    frame_type : Typos frame
    SMR : Signal-to-Mask Ratio apo to psyxoakoustiko montelo
    
    Epistrefei:

    S : Kvantismenoi syntelestes (akeraioi)
    sfc : Scale factors (DPCM encoded)
    G : Global gain
    """
    frame_type = frame_type.upper()
    load_bands()
    
    # Gia ESH frames, epexergazomaste 8 subframes
    if frame_type == "ESH":
        X = frame_F
        n_sub = X.shape[1]
        
        S_all = np.zeros((1024, 1), dtype=np.int32)
        sfc_all, G_all = [], []
        
        for s in range(n_sub):  
            S_sub, sfc_sub, G_sub = quantize_subframe(X[:, s], "ESH", SMR[:, s])  # Epistrefei kvantismenous S_sub (128,) gia kathe subframe, sfc_sub (42,), G_sub (scalar) gia to subframe s
            S_all[s*128:(s+1)*128, 0] = S_sub   # S_all: (1024,) me ta 8 subframes se diadoxika tmhmata
            sfc_all.append(sfc_sub)     # sfc_all: lista me 8 pinakes (42,) me ta scale factors gia kathe subframe
            G_all.append(G_sub)
        
        return S_all, np.column_stack(sfc_all), np.array(G_all)     # S: (1024, 1), sfc: (42, 8), G: (8,)
    
    # Gia long frames
    S, sfc, G = quantize_subframe(frame_F.flatten(), frame_type, SMR.flatten()) # S: (1024,), sfc: (69,), G: scalar
    return S.reshape(-1, 1), sfc.reshape(-1, 1), G  # reshape se (1024, 1), (69, 1), scalar G



def i_aac_quantizer(S, sfc, G, frame_type):
    """
    Inverse AAC Quantizer - Apokvantisi
    
    Anakataskeyi tous syntelestes MDCT apo tous kvantismenous
    syntelestes kai ta scale factors.
    
    Parametroi:
    -----------
    S : Kvantismenoi syntelestes
    sfc : Scale factors (DPCM encoded)
    G : Global gain
    frame_type : Typos frame
    
    Epistrefei:
    -----------
    X_out : Apokvantimenoi syntelestes MDCT
    """
    frame_type = frame_type.upper()
    load_bands()
    
    # Gia ESH frames, epexergazomaste 8 subframes
    if frame_type == "ESH":
        S_flat = S.flatten()
        n_sub = len(G) if isinstance(G, np.ndarray) else 8
        G = G if isinstance(G, np.ndarray) else np.full(8, G)
        
        X_out = np.zeros((128, n_sub))
        for s in range(n_sub):
            sfc_sub = sfc[:, s] if sfc.ndim > 1 else sfc
            X_out[:, s] = dequantize_subframe(S_flat[s*128:(s+1)*128], sfc_sub, G[s], "ESH")
        
        return X_out
    
    # Gia long frames
    G_scalar = G if np.isscalar(G) else (G.item() if isinstance(G, np.ndarray) else float(G))
    X = dequantize_subframe(S.flatten(), sfc.flatten(), G_scalar, frame_type)
    return X.reshape(-1, 1)



