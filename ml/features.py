# ml/features.py
import numpy as np
import librosa
from scipy.signal import butter, filtfilt, welch

# Tapping features
def tapping_features(timestamps, x_coords=None, y_coords=None):
    # timestamps: list or np.array of tap timestamps in seconds
    itis = np.diff(timestamps)
    if len(itis) < 2:
        return {
            'iti_mean': 0.0,
            'iti_std': 0.0,
            'iti_cv': 0.0,
            'num_taps': float(len(timestamps)),
            'iti_slope': 0.0
        }
    
    feats = {}
    feats['iti_mean'] = np.mean(itis)
    feats['iti_std'] = np.std(itis)
    feats['iti_cv'] = feats['iti_std'] / (feats['iti_mean'] + 1e-9)
    feats['num_taps'] = float(len(timestamps))
    
    # fatigue slope: slope of ITI over time
    t = np.arange(len(itis))
    slope = np.polyfit(t, itis, 1)[0]
    feats['iti_slope'] = slope
    return feats

# Tremor features - bandpass + spectral features
def bandpass_filter(sig, fs, low=3.0, high=12.0, order=4):
    nyq = 0.5 * fs
    lown = low/nyq
    highn = high/nyq
    b, a = butter(order, [lown, highn], btype='band')
    return filtfilt(b, a, sig)

def tremor_features(accel, fs):
    # accel: 1D numpy array magnitude or single-axis
    if len(accel) < fs:  # <1 second
        return {
            'tremor_peak_freq': 0.0,
            'tremor_peak_power': 0.0,
            'power_3_7': 0.0,
            'power_7_12': 0.0,
            'tremor_rms': 0.0
        }
        
    feats = {}
    sig = accel - np.mean(accel)
    filt = bandpass_filter(sig, fs, low=3.0, high=12.0)
    f, Pxx = welch(filt, fs=fs, nperseg=min(1024, len(filt)))
    
    # dominant freq
    idx = np.argmax(Pxx)
    feats['tremor_peak_freq'] = float(f[idx])
    feats['tremor_peak_power'] = float(Pxx[idx])
    
    # band power 3-7 and 7-12
    band1 = np.logical_and(f>=3, f<=7)
    band2 = np.logical_and(f>7, f<=12)
    feats['power_3_7'] = float(np.trapz(Pxx[band1], f[band1]))
    feats['power_7_12'] = float(np.trapz(Pxx[band2], f[band2]))
    feats['tremor_rms'] = float(np.sqrt(np.mean(filt**2)))
    return feats

# Voice features (MFCC)
def voice_features(wav, sr):
    if len(wav) < int(0.025 * sr): # Not even one frame
        return {
            'mfcc_mean': [0.0] * 13,
            'mfcc_std': [0.0] * 13,
            'spec_centroid_mean': 0.0,
            'hnr': 0.0
        }
        
    feats = {}
    # Pre-emphasis
    pre = np.append(wav[0], wav[1:] - 0.97*wav[:-1])
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=pre, sr=sr, n_mfcc=13, hop_length=int(0.01*sr), n_fft=int(0.025*sr))
    feats['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()  # 13 values
    feats['mfcc_std'] = np.std(mfcc, axis=1).tolist()
    
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=pre, sr=sr)
    feats['spec_centroid_mean'] = float(np.mean(centroid))
    
    # HNR approximate
    harm, percussive = librosa.effects.hpss(pre)
    harmonic_energy = float(np.sum(harm**2))
    total_energy = float(np.sum(pre**2) + 1e-9)
    feats['hnr'] = harmonic_energy / total_energy
    return feats

# Utility to convert feature dict to vector (consistent ordering)
def featurize_dict(fdict):
    vec = []
    # tapping fields (5)
    for k in ['iti_mean','iti_std','iti_cv','num_taps','iti_slope']:
        vec.append(float(fdict.get(k, 0.0)))
    
    # tremor fields (5)
    for k in ['tremor_peak_freq','tremor_peak_power','power_3_7','power_7_12','tremor_rms']:
        vec.append(float(fdict.get(k, 0.0)))
    
    # voice: 13 mean + 13 std (26)
    mfcc_mean = fdict.get('mfcc_mean', [0.0]*13)
    mfcc_std = fdict.get('mfcc_std', [0.0]*13)
    vec.extend([float(x) for x in mfcc_mean])
    vec.extend([float(x) for x in mfcc_std])
    
    # additional voice features (2)
    vec.append(float(fdict.get('spec_centroid_mean',0.0)))
    vec.append(float(fdict.get('hnr',0.0)))
    
    # Total: 5 + 5 + 26 + 2 = 38 features
    return np.array(vec, dtype=np.float32)