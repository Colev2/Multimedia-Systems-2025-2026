"""
Level 3 - AAC Encoder/Decoder
Kodikwpoihths kai Apokwdikopoihths AAC Level 3 - Lossy compression me ypologismo SNR, bitrate, compression

Ylopoiei olo to pipeline:
WAV -> SSC -> Filter Bank -> TNS -> Psycho -> Quantizer -> Huffman -> .mat (Encoder)
iHuffman -> iQuantizer -> iTNS -> iFilterBank -> Overlap-Add -> WAV (Decoder)
"""
import numpy as np
import soundfile as sf
from scipy.io import savemat, loadmat
import sys, os
import csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huff_utils import load_LUT, encode_huff, decode_huff
from .SSC import SSC
from .filter_bank import filter_bank, i_filter_bank
from .TNS import tns, i_tns
from .psycho import psycho
from .quantizer import aac_quantizer, i_aac_quantizer, get_last_quantizer_stats


def aac_coder_3(filename_in, filename_aac_coded):
    """
    Kodikopitis AAC - Level 3
    
    Diavazei ixo, ton kodikopoiei frame-by-frame kai apothikeyei to apotelesma.
    
    Pipeline: WAV → Frames → SSC → FilterBank → TNS → Psycho → Quantizer → Huffman → .mat
    """
    # 1. Fortwsh arxeiou ixou
    audio, fs = sf.read(filename_in, dtype='float64')
    
    # 2. Arxikopoihsh
    huff_LUT = load_LUT()
    frame_size, hop_size = 2048, 1024
    
    # Ypologismos posa frames xreiazomaste
    n_frames = int(np.ceil((len(audio) - frame_size) / hop_size)) + 1
    total_samples = frame_size + (n_frames - 1) * hop_size
    
    # Prosthiki padding an xreiazetai
    if total_samples > len(audio):
        audio = np.vstack([audio, np.zeros((total_samples - len(audio), 2))])
    
    # 3. Metavlhtes katastashs
    aac_seq_3 = []  # Lista me ta kwdikopoihmena frames
    prev_frame_type = "OLS"
    prev_frame_T = [np.zeros((2048, 2)), np.zeros((2048, 2))]  # Prohgoumena 2 frames
    
    # Ypologismos interval gia debug printing se long frames
    debug_interval =  n_frames // 10
    
    # CSV logging gia sygkrisi max_iter
    csv_filename = filename_aac_coded.replace('.mat', '_metrics.csv')
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'channel', 'frame_type', 'bits_S', 'bits_sfc', 'nonzero', 'maxAbs', 'codebook', 'iterations'])
    
    # 4. Epexergasia kathe frame
    for i in range(n_frames):   # Gia kathe frame
        # Trexon frame
        start, end = i * hop_size, i * hop_size + frame_size
        if end > len(audio):
            break
        frame_T = audio[start:end, :]   # Trexon frame (2048, 2)
        
        # Epomeno frame gia SSC
        next_start, next_end = (i + 1) * hop_size, (i + 1) * hop_size + frame_size
        if next_end <= len(audio):
            next_frame_T = audio[next_start:next_end, :]
        else:
            next_frame_T = np.zeros((frame_size, 2))
        
        # Typos tou frame (OLS, ESH, LSS, LPS)
        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        
        # Dictionary gia to frame
        frame_dict = {"frame_type": frame_type,"chl": {},  # Aristero kanali
        "chr": {}   # Dexi kanali
        }
        
        # 5. Epexergasia kathe kanaliou xexorista
        for ch_idx, ch_name in enumerate(["chl", "chr"]):
            # 5a. Filter Bank: Metasximatismos se pedio syxnotitas (MDCT)
            frame_F = filter_bank(frame_T, frame_type)[:, ch_idx:ch_idx+1]  # Syntelestes MDCT gia to kanali ch_idx, se morfh (1024, 1) 
            if frame_type == "ESH":
                frame_F = frame_F.reshape(128, 8)   # reshape tou (1024, 1) se (128, 8) gia ta 8 subframes
            
            # 5b. TNS: Temporal Noise Shaping
            frame_F_tns, tns_coeffs = tns(frame_F, frame_type)   # Filtrarismenoi Syntelestes MDCT meta to TNS
            frame_dict[ch_name]["tns_coeffs"] = tns_coeffs  # Apothikeusi tns_coeffs sto dictionary tou kanaliou ch_name
            
            # 5c. Psychoacoustic Model: Ypologismos katoflio
            SMR = psycho(frame_T[:, ch_idx], frame_type, prev_frame_T[1][:, ch_idx], prev_frame_T[0][:, ch_idx] )   # SMR gia to kanali ch_idx, se morfh (69, 1) gia long frames i (42, 8) gia ESH
            frame_dict[ch_name]["T"] = SMR  # Apothhkeush SMR sto dictionary tou kanaliou ch_name
            
            # 5d. Quantizer: Kvantish syntelestwn TNS
            S, sfc, G = aac_quantizer(frame_F_tns, frame_type, SMR)
            frame_dict[ch_name]["G"] = G    # Apothhkeush global gain
            
            # Debug printing gia Long frames
            if frame_type in ["OLS", "LSS", "LPS"] and (i % debug_interval == 0) and ch_name == "chl":
                stats = get_last_quantizer_stats()
                if stats["termination"] == "Converged":
                    reason_text = "Pe>=T gia ola ta bands or |Δa| > 60 gia ola ta diadoxika bands"
                else:
                    reason_text = "reached max iterations"
                print(f"\nFrame {i:4d} | Type: {frame_type:3s} | Channel: {ch_name}")
                print(f"  SMR:        [{stats['smr_min']:8.2f}, {stats['smr_max']:8.2f}]")
                print(f"  P(b):       [{stats['P_min']:8.2e}, {stats['P_max']:8.2e}]  (MDCT energy)")
                
                # npart(b) - threshold apo psycho
                if stats.get('npart_max', 0) > 0:
                    print(f"  npart(b):   [{stats['npart_min']:8.2e}, {stats['npart_max']:8.2e}]  (threshold apo psycho)")
                
                # P/e ratio - eleghxos klimakas
                if stats.get('P_over_e_mean', 0) > 0:
                    print(f"  P(b)/e(b):  [{stats['P_over_e_min']:8.2f}, {stats['P_over_e_max']:8.2f}] ")
                
                print(f"  T(b):       [{stats['T_min']:8.2e}, {stats['T_max']:8.2e}]  (=P/SMR)")
                print(f"  Pe initial: [{stats['Pe_initial_min']:8.2e}, {stats['Pe_initial_max']:8.2e}]")
                print(f"  alpha:      [{stats['alpha_min']:8.2f}, {stats['alpha_max']:8.2f}] | Iter: {stats['term_iter']:3d} | {reason_text}")
                
                # Debug gia specific band (band 20) gia elegxo tis sxesis T = P/SMR
                if stats.get('SMR_band20', 0) > 0:
                    T_calc = stats['P_band20'] / (stats['SMR_band20'] + 1e-12)
                    print(f"\n  [Band 20 Check] SMR={stats['SMR_band20']:8.2e}, P={stats['P_band20']:8.2e}, T={stats['T_band20']:8.2e}")
                    print(f"T = P/SMR = {T_calc:8.2e}  (Match: {abs(T_calc - stats['T_band20']) < 1e-3})")
                    print(f"Pe(initial) = {stats['Pe_band20']:8.2e}  (Pe < T: {stats['Pe_band20'] < stats['T_band20']})")
            
            # 5e. Huffman Encoding
            sfc_stream, _ = encode_huff(sfc.flatten().astype(int), huff_LUT, force_codebook=11) # Gia scalefactors, xrisimopoioume panta to codebook 11 gia to huffman encoding
            frame_dict[ch_name]["sfc"] = sfc_stream
            
            stream, codebook = encode_huff(S.flatten().astype(int), huff_LUT)  # Huffman encoding twn kvantismenwn filtrarismenwn MDCT, me automath epilogh codebook
            frame_dict[ch_name]["stream"] = stream
            frame_dict[ch_name]["codebook"] = codebook
            
            # Metrikes gia logging
            bits_S = len(stream)
            bits_sfc = len(sfc_stream)
            nonzero = np.count_nonzero(S)
            maxAbs = np.max(np.abs(S))
            iterations = get_last_quantizer_stats()["term_iter"]
            
            # Logging se CSV gia ola ta frames
            csv_writer.writerow([i, ch_name, frame_type, bits_S, bits_sfc, nonzero, maxAbs, codebook, iterations])
            
            # Debug gia bits consumption
            if frame_type in ["OLS", "LSS", "LPS"] and (i % debug_interval == 0) and ch_name == "chl":
                print(f"  [Bits] sfc: {bits_sfc:5d} bits | S: {bits_S:5d} bits | Total: {bits_sfc + bits_S:5d} bits")
                print(f"         sfc values: [{sfc.min():4d}, {sfc.max():4d}] | Non-zero S: {nonzero:4d}/{len(S):3d} | maxAbs: {maxAbs:2d} | codebook: {codebook:2d}")
        
        # 6. Apothhkeush frame kai enhmerwsh prohgoumenou frame
        aac_seq_3.append(frame_dict)
        prev_frame_type = frame_type
        prev_frame_T = [prev_frame_T[1].copy(), frame_T.copy()]
    
    # Kleisimo CSV
    csv_file.close()
    print(f"\n[INFO] Metrics saved to: {csv_filename}")
    
    return aac_seq_3


def i_aac_coder_3(aac_seq_3, filename_out):
    """
    Apokodikopitis AAC - Level 3
    
    Pernei kodikopoiimena frames kai ta metratepei piso se ixo.
    
    Pipeline: .mat → Huffman → iQuantizer → iTNS → iFilterBank → Overlap-Add → WAV
    """
    # 1. Arxikopoihsh
    huff_LUT = load_LUT()
    n_frames = len(aac_seq_3)
    frame_size, hop_size = 2048, 1024
    
    # Ypologismos synolikwn deigmatwn
    total_samples = frame_size + (n_frames - 1) * hop_size
    audio_out = np.zeros((total_samples, 2))
    
    # 2. Apokwdikopoihsh kathe frame
    for i, frame_dict in enumerate(aac_seq_3):
        frame_type = frame_dict["frame_type"]   # extract frame type gia to frame i
        frame_F_decoded = np.zeros((1024, 2))
        
        # 3. Epexergasia kathe kanaliou
        for ch_idx, ch_name in enumerate(["chl", "chr"]):
            ch_data = frame_dict[ch_name]
            
            # 3a. Huffman Decoding
            sfc = np.array(decode_huff(ch_data["sfc"], huff_LUT[11]), dtype=np.int32)
            
            # Apokodikopii si kvantismenon synteleston
            if ch_data["codebook"] == 0:
                S_decoded = np.zeros(1024, dtype=np.int32)
            else:
                S_decoded = np.array(decode_huff(ch_data["stream"], huff_LUT[ch_data["codebook"]]), dtype=np.int32)
                
                # Ruthmish tou S_decoded gia na exei 1024 syntelestes (gia long frames) i 128 syntelestes (gia ESH)
                if len(S_decoded) < 1024:
                    S_decoded = np.pad(S_decoded, (0, 1024 - len(S_decoded)))
                elif len(S_decoded) > 1024:
                    S_decoded = S_decoded[:1024]
            
            S = S_decoded.reshape(-1, 1)
            G = ch_data["G"]
            
            # Ruthmish sfc gia ESH i long frames
            if frame_type == "ESH":
                sfc = np.pad(sfc, (0, max(0, 42*8 - len(sfc))))[:42*8].reshape(42, 8)
            else:
                sfc = np.pad(sfc, (0, max(0, 69 - len(sfc))))[:69].reshape(-1, 1)
            
            # 3b. Inverse Quantizer: Apokvantisi
            frame_F_quant = i_aac_quantizer(S, sfc, G, frame_type)
            
            # 3c. Inverse TNS
            frame_F = i_tns(frame_F_quant, frame_type, ch_data["tns_coeffs"])
            
            # Rithmisi shape
            if frame_type == "ESH":
                frame_F = frame_F.reshape(-1, 1)
            
            frame_F_decoded[:, ch_idx:ch_idx+1] = frame_F
        
        # 3d. Inverse Filter Bank: Epistrofi sto pedio xronou
        frame_T = i_filter_bank(frame_F_decoded, frame_type)
        
        # 4. Overlap-Add
        start, end = i * hop_size, i * hop_size + frame_size
        if end <= len(audio_out):
            audio_out[start:end, :] += frame_T
    
    # 5. Apothikeisi
    sf.write(filename_out, audio_out, 48000)
    return audio_out
