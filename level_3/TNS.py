"""
Level 3 - Temporal Noise Shaping (TNS)
Diafanopoiisi xronikoy thoryvoy - Efarmozei gramiki provlepsi sta MDCT

Ylopoiei:
    frame_F_out, tns_coeffs = tns(frame_F_in, frame_type)
    frame_F_out = i_tns(frame_F_in, frame_type, tns_coeffs)
"""
import numpy as np
from scipy.io import loadmat
from numpy.polynomial.polynomial import Polynomial
import os


# Katholikoi pinakes bands
BANDS_LONG = None
BANDS_SHORT = None

# Fortwsh twn bands apo ta .mat arxeia
def load_bands():
    """Fortwsh bands apo TableB219.mat"""
    global BANDS_LONG, BANDS_SHORT
    if BANDS_LONG is not None:
        return
    
    mat_data = loadmat("TableB219.mat")
    BANDS_LONG = mat_data['B219a']   # 69 bands gia long
    BANDS_SHORT = mat_data['B219b']  # 42 bands gia short

# Extract band edges (w_low, w_high) 
def get_band_edges(frame_type):
    """Epistrefei ta oria twn bands (w_low, w_high)"""
    load_bands()
    if frame_type in ("OLS", "LSS", "LPS"):
        bands = BANDS_LONG          # pinakas B219a gia long frames
    else:
        bands = BANDS_SHORT         # pinakas B219b gia short frames
    return bands[:, 1].astype(int), bands[:, 2].astype(int)     # Epistrefei w_low, w_high



# Efarmogi FIR filtrou
def apply_fir(X, a):
    """
    Orismata:
    X : Syntelestes MDCT arxika
    a : Syntelestes provlepshs (LPC coefficients)

    Efarmogi FIR filtrou: Y(k) = X(k) - sum(a_l * X(k-l)) to opoio me Z transform einai 
    H_TNS(z) = 1 - a₁z⁻¹ - a₂z⁻² - a₃z⁻³ - a₄z⁻⁴        (epeidh einai takshs p=4)
    
    Epistrefei:
    Y : Syntelestes MDCT meta to FIR filtro
    """
    Y = np.copy(X)
    for k in range(len(X)):
        for l in range(1, min(k+1, 5)):   # Apofygh index out of bounds: k-l >= 0, gia k>5 h eksodox Y(k) tou filtrou 
            Y[k] -= a[l-1] * X[k-l]       # eksartatai mono apo ta X(k-l) me l=1,2,3,4
    return Y



# Elegxos eystathias
def is_stable(a):
    """
    Orismata:
    a : Syntelestes provlepshs (LPC coefficients)

    Elegxos eystathias: Oloi oi poloi entos monadiakou diskou |z| < 1

    Epistrefei:
    True an to filtro einai eustathes, False alliws
    """
    p = 4
    poly_coeffs = np.concatenate([[-a[p-1-i] for i in range(p)], [1.0]])   # Enwse tous syntelestes a tou filtou se pinaka
                                                                           # ths morfhs []
    try:
        roots = Polynomial(poly_coeffs).roots()
        return np.all(np.abs(roots) < 1.0)      # Epistrefei True an oloi oi poloi einai entos monadiakou diskou
    except:
        return False        # An apotuxei o ypologismos twn rizon, theoroume to filtro astathes



# TNS Analysis (Encoder)

def tns_subframe(X, frame_type):
    """
    Efarmogh TNS se ena frame/subframe

    Orismata:
    X : Syntelestes MDCT gia ena frame/subframe
    frame_type : Typos frame (OLS/LSS/LPS/ESH)
    
    Vhmata:
    1. Kanonikopoihsh syntelestwn ana band (normalization)
    2. Ypologismos LPC syntelestwn (grammikh provlepsh)
    3. Kvantismos syntelestwn (4 bits, vima 0.1)
    4. Elegxos eystatheias
    5. Efarmogh FIR filtrou

    Epistrefei:
    X_filt : Syntelestes MDCT meta to TNS
    a_quant : Kvantismenoi syntelestes provlepshs
    """
    N = len(X)
    w_low, w_high = get_band_edges(frame_type)
    n_bands = len(w_low)
    
    # Vima 1: Kanonikopoihsh ana band

    # Ypologismos energeias kathe band P(j) (Exiswsh 3)
    P = np.array([np.sum(X[w_low[j]:w_high[j]+1]**2) for j in range(n_bands)])  # P(j) = Σ_(k=b_j)^(b_{j+1}-1) |X(k)|²
    
    # Syntelestes kanonikopoihshs S_w(k) (Exiswsh 4)
    S_w = np.ones(N)
    for j in range(n_bands):
        S_w[w_low[j]:w_high[j]+1] = np.sqrt(P[j])   # S_w(k) = sqrt(P(j)), b_j=w_low[j] <= k <= b_{j+1}-1=w_high[j]
    
    # Exomalynsi (2 perasmata)
    for k in range(N-2, -1, -1):
        S_w[k] = (S_w[k] + S_w[k+1]) / 2.0      # Sw[k] = (S_w[k] + S_w[k+1]) / 2.0, k=N-2,...,0
    for k in range(1, N):
        S_w[k] = (S_w[k] + S_w[k-1]) / 2.0      # Sw[k] = (S_w[k] + S_w[k-1]) / 2.0, k=1,...,N-1                
    
    X_w = X / (S_w + 1e-12)     # Gia apofygh diaireshs me 0
    
    # Vima 2: Ypologismos LPC syntelestwn (taxh p=4)
    p = 4
    r = np.correlate(X_w, X_w, mode='full')     
    r = r[len(X_w)-1:len(X_w)+p]
    
    # Dhmiourgia pinaka Toeplitz (Exiswsh 7)
    # R(m,n) = r(|m-n|), m,n = 0,...,p-1
    # O pinakas Toeplitz exei thn idia diagοnio kai einai symmetrikos
    R = np.zeros((p, p))
    for m in range(p):
        for n in range(p):
            R[m, n] = r[abs(m - n)]  # R = [[r[0], r[1], r[2], r[3]],
                                     #      [r[1], r[0], r[1], r[2]],
                                     #      [r[2], r[1], r[0], r[1]],
                                     #      [r[3], r[2], r[1], r[0]]]
    
    R += np.eye(p) * 1e-8  # Regularization-prosthetw 10^(-8) sta diagwnia stoixeia gia arithmitikh eystatheia
    
    try:
        a = np.linalg.solve(R, r[1:p+1])  # Lyse to R*a = r[1:p+1] gia na vroume tous syntelestes a tou FIR filtrou
    except:
        a = np.zeros(p)   # An apotuxei o ypologismos twn rizwn, tote to filtro einai 
                          # astathes kai oi syntelestes tha einai a=0 (dhladh to filtro den allazei ton xrono)
    
    # Vima 3: Kvantismos me 4 bits, vima 0.1
    a_quant = np.round(a / 0.1) * 0.1
    a_quant = np.clip(a_quant, -0.8, 0.7)
    
    # Vima 4: Elegxos eystatheias filtrou prin thn efarmogh tou
    if not is_stable(a_quant):     # An is_stable epistrefei False, tote to filtro einai astathes
        a_quant = np.zeros(p)       # Syntelestes a=0
    
    # Vima 5: Efarmogi FIR filtrou sto arxiko X
    X_filt = apply_fir(X, a_quant)
    
    return X_filt, a_quant   # Epestrepse tous neous MDCT meta to FIR filtro kai tous kvantismenous syntelestes a



def tns(frame_F_in, frame_type):
    """
    Temporal Noise Shaping - Grammikh provlepsh sto pedio tis syxnothtas
    
    Efarmozei FIR filtro gia na diaperasei to sfalma kvantismou sto xrono, kanontas to na akougetai 
    san omoiomorfos thoryvos anti gia xronika topikopoimenes paramorfseis.
    
    Parametroi:

    frame_F_in : Syntelestes MDCT prin to TNS
                 (1024, 1) gia OLS/LSS/LPS ή (128, 8) gia ESH
    frame_type : Typos frame
    
    Epistrefei:

    frame_F_out : Y: Syntelestes MDCT meta to TNS (idio shape)
    tns_coeffs : Kvantismenoi syntelestes provlepshs (4, 1) ή (4, 8)
    """
    frame_type = frame_type.upper()
    
    if frame_type == "ESH":     # MDCT: (128, 8) 
        # Epexergasia 8 subframes
        n_sub = frame_F_in.shape[1]     # n_sub = 8 
        X_in = frame_F_in       # Plhthos subframes
        
        X_out = np.zeros_like(X_in)         
        tns_coeffs = np.zeros((4, n_sub))
        
        for s in range(n_sub):
            X_out[:, s], tns_coeffs[:, s] = tns_subframe(X_in[:, s], "ESH")  # TNS stous syntelestes MDCT tou subframe s
            # X_out: Neoi syntelestes MDCT meta to TNS gia to subframe s, 
            # tns_coeffs: oi kvantismenoi syntelestes provlepshs gia to subframe s
        
        return X_out, tns_coeffs
    
    else:           # OLS/LSS/LPS: MDCT (1024, 1)
        # Long frame
        X_in = frame_F_in.flatten()
        X_out, coeffs = tns_subframe(X_in, frame_type)
        return X_out.reshape(-1, 1), coeffs.reshape(-1, 1)  # Neoi syntelestes MDCT meta to TNS kai oi kvantismenoi syntelestes provlepshs gia to long frame




# TNS Synthesis (Decoder)

def i_tns(frame_F_in, frame_type, tns_coeffs):
    """
    Inverse TNS - Antistrefi to TNS filtro
    
    Efarmozei IIR filtro (antistrof tou FIR) ston apokwdikopoihth
    """
    frame_type = frame_type.upper()
    
    if frame_type == "ESH":
        n_sub = frame_F_in.shape[1] 
        Y_in = frame_F_in 
        
        X_out = np.zeros_like(Y_in)
        for s in range(n_sub):
            X_out[:, s] = apply_iir(Y_in[:, s], tns_coeffs[:, s])
        
        return X_out
    
    else:
        X_out = apply_iir(frame_F_in.flatten(), tns_coeffs.flatten())
        return X_out.reshape(-1, 1)


def apply_iir(Y, a):
    """
    Efarmogh IIR filtrou: X(k) = Y(k) + sum(a_l * X(k-l))
    """
    X = np.zeros(len(Y))
    for k in range(len(Y)):
        X[k] = Y[k]
        for l in range(1, min(k+1, len(a)+1)):
            X[k] += a[l-1] * X[k-l]
    return X
