"""
Level 1 - Filterbank (MDCT / IMDCT + KBD parathyra)
Metasximatismos pediou xronou se pedio syxnotitas kai antistrofa

Ylopoiei:
    frame_F = filter_bank(frame_T, frame_type)
    frame_T = i_filter_bank(frame_F, frame_type)
    
Xrhsimopoiei mono KBD parathyra (Kaiser-Bessel Derived)
"""

import numpy as np
from scipy.signal.windows import kaiser


# Pinakes MDCT/IMDCT (cache)

def mdct_matrix(N):
    """Kataskeyh pinaka synhmitonwn gia MDCT"""
    n0 = (N / 2 + 1) / 2
    n = np.arange(N)[None, :]   # n : Deikths Deigmatos tou frame 
                                # Arrange(N): dianysma [0, 1, ..., N-1], shape: (N,)
                                # [None, :]: Reshape se (1, N): [0 1 ... N-1] gia broadcasting
    k = np.arange(N // 2)[:, None]  # k : Deikths syxnothtwn 
                                    # Arrange(N//2): dianysma [0, 1, ..., N//2-1] shape: (N//2,)
                                    # [:, None]: Reshape se (N//2, 1): 
                                    # [0 
                                    #  1 
                                    #  ...
                                    #  N//2-1] gia broadcasting
    return np.cos((2.0 * np.pi / N) * (n + n0) * (k + 0.5))     # (n+n0): dianysma (1,N): [[0+n0, 1+n0, ..., N-1+n0]] 
                # (k+0.5): dianysma (N//2,1): [[0.5], [1.5], ..., [N//2-0.5]]
                # (2.0 * np.pi /N): arithmos
                # Estw N = (n+n0) = [[N1, N2, ..., Nn]] kai K = (k+0.5) = [[K1], [K2], ..., [Kk]]
                # A = (n+n0) * (k+0.5) = N * K: 
                # [N1*K1 N2*K1 ... Nn*K1 
                #  N1*K2 N2*K2 ... Nn*K2
                #  ...,
                #  N1*Kk N2*Kk ... Nn*Kk]
                # Epeidh o pol/mos (n+n0)*(k+0.5) einai (1,N)*(N//2,1), to ginomeno tous ypologizetai 
                # me broadcasting pou kanei h python eswterika kai einai diastasewn (N//2, N)
                # C = cos((2.0 * np.pi / N) * A) 
                # Epistrefei ton pinaka C me megethos (N//2, N) pou periexei tis synhmitonikes times gia kathe syndyasmo n,k 
                # Kathe grammh tou C einai mia synhmitonikh vash syxnothtas k, pou einai dianysma mhkous N
                # Dhladh: Ck = [Ck[0], Ck[1], ..., Ck[N-1]] gia k=0, 1, ..., (N//2)-1


C_2048 = mdct_matrix(2048)  # (1024, 2048)
C_256 = mdct_matrix(256)    # (128, 256)


def mdct(x, N):        # Metasxhmatismos apo pedio xronou se pedio syxnothtas
    """Modified Discrete Cosine Transform"""
    C = C_2048 if N == 2048 else C_256
    return 2.0 * (C @ x)        # Pinakas MDCT me megethos (N//2,), opou kathe grammh k einai:
                                # X[k] = <x,Ck> = Σ C[k,n] * x[n] gia ola ta n, opou @ einai to matrix multiplication
                                # Deixnei thn omoiothta tou frame x me tis synhmitonikes vaseis Ck
                                # P.x X[3] = Σ C[3,n] * x[n]  deixnei poso moiazei to frame x me thn cosine vash syxnothtas 3

def imdct(X, N):           # Antistrofos metasxhmatismos: apo pedio syxnothtas se pedio xronou
    """Inverse MDCT"""
    C = C_2048 if N == 2048 else C_256
    return (2.0 / N) * (C.T @ X)


# KBD Parathyro

def kbd_window(N, a):
    """Kaiser-Bessel Derived (KBD) parathyro"""
    M = N // 2 + 1
    w = kaiser(M, beta=np.pi * a)
    denom = np.sum(w)
    left = np.sqrt(np.cumsum(w[:-1]) / denom)
    return np.concatenate([left, left[::-1]])


Wl = kbd_window(2048, a=6.0)
Ws = kbd_window(256, a=4.0)
   

def frame_window(frame_type):
    """Epistrefei to KBD parathyro analoga to frame type"""
    
    if frame_type == "OLS":
        W = Wl
        return W            # W(n) = Wkbd(n) gia 0<= n <2048 = Wl

    elif frame_type == "LSS":
        W = np.concatenate([Wl[:1024], np.ones(448), Ws[128:], np.zeros(448)])  # Enwnei diadoxika ta 4 tmhmata tou parathyrou
        return W        # W(n) = { Wkbd,2048(n), gia 0<= n <1024
                        #          1, gia 1024<= n <1472
                        #          Wkbd,256(n), gia 1472<= n <1600
                        #          0, gia 1600<= n <2048  }
        

    elif frame_type == "LPS":
        W = np.concatenate([np.zeros(448), Ws[:128], np.ones(448), Wl[1024:]])       
        return W        # W(n) = { 0, gia 0<= n <448
                        #          Wkbd,256(n), gia 448<= n <576
                        #          1, gia 576<= n <1024
                        #          Wkbd,2048(n), gia 1024<= n <2048  }

    # ESH
    return Ws           # W0(n) = W1(n) = ... = W8(n) = Wkbd,256(n) gia 0<= n <256 = Ws


# Filterbank: parathyro + MDCT

def filter_bank(frame_T, frame_type):
    """
    Filterbank: KBD parathyro + MDCT
    
    Parametroi:

    frame_T : Frame sto pedio xronou (2048, 2)
    frame_type : Typos frame ('OLS', 'LSS', 'ESH', 'LPS')
    
    Epistrefei:

    frame_F : Syntelestes MDCT (1024, 2)
              Gia ESH: 8 blocks ton 128 grammwn (subframes 0..7)
    """
    frame_type = frame_type.upper()
    W_frame = frame_window(frame_type)

    frame_F = np.zeros((1024, 2))

    if frame_type in ("OLS", "LSS", "LPS"):
        for ch in range(2):     # Gia kathe kanali hxou
            xw = frame_T[:, ch] * W_frame   # Pol/zw ta deigmata me to analogo parathyro (me vash to frame type)
                                            # x_w(n) = x(n)*w(n). Exodos: 2048 nea deigmata
            frame_F[:, ch] = mdct(xw, 2048)    # Dwse ta nea deigmata ston MDCT kai pare tous 1024 syntelestes sto pedio syxnothtwn
        return frame_F
    
    # ESH: 8 short MDCTs sta 1152 kentrika samples
    center = frame_T[448:1600, :]  # center: pinakas (1152, 2)

    for s in range(8):
        start = s * 128
        sub = center[start:start + 256, :]  # subframe: (256, 2) 256 deigmata ths kentrikhs perioxhs apo kathe kanali
        subw = sub * Ws[:, None]        # Pol/smos to usubframe me to antistoixo parathyro Ws (256,) gia kathe kanali, me broadcasting 
        for ch in range(2):
            Xs = mdct(subw[:, ch], 256)  # (128,) syntelestes MDCT gia to subframe tou kanaliou ch
            frame_F[s * 128:(s + 1) * 128, ch] = Xs  # Apothikeuse tous 128 syntelestes MDCT stis theseis 128*s...128*(s+1) tou frame_F gia to kanali ch
                                                 # frame_F: (1024, 2) 

    return frame_F      # Epistrofh twn 1024 syntelestwn MDCT gia kathe kanali, se morfh (1024, 2)

# Inverse Filterbank: IMDCT + parathyro

def i_filter_bank(frame_F, frame_type):
    """
    Filterbank synthesis: IMDCT + KBD parathyro
    
    Parametroi:

    frame_F : Syntelestes MDCT (1024, 2)
    frame_type : Typos frame ('OLS', 'LSS', 'ESH', 'LPS')
    
    Epistrefei:
    
    frame_T : Anakataskevasmeno frame sto pedio xronou (2048, 2)
              Gia telikh anakataskeyh xreiazontai overlap-add diadoxikwn frames
    """
    frame_type = frame_type.upper()
    W_frame = frame_window(frame_type)

    frame_T = np.zeros((2048, 2))

    if frame_type in ("OLS", "LSS", "LPS"):
        for ch in range(2):
            y = imdct(frame_F[:, ch], 2048)  # (2048,)
            frame_T[:, ch] = y * W_frame
        return frame_T

    # ESH 
    buf = np.zeros((1152, 2))

    for s in range(8):
        Xs = frame_F[s * 128:(s + 1) * 128, :]  # (128, 2)
        start = s * 128
        for ch in range(2):
            y = imdct(Xs[:, ch], 256) * Ws  # (256,)
            buf[start:start + 256, ch] += y

    frame_T[448:1600, :] = buf
    return frame_T              # Epistrofh tou anakataskevasmenou frame sto pedio xronou