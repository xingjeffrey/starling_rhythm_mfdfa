from scipy.io.wavfile import read
from scipy.signal import hilbert, get_window
from librosa import resample
import noisereduce as nr
import pandas as pd
import numpy as np

def int16_to_float32(data):
    """ Converts from uint16 wav to float32 wav
    """
    if np.max(np.abs(data)) > 32768:
        raise ValueError("Data has values above 32768")
    return (data / 32768.0).astype("float32")


def extract_amp_env(
    data, 
    target_sr, 
    win_type = 'boxcar', 
    res_type = 'linear',
    stds = 1,
    buffer = 100,
    spl = True,
    compact = False,
    hilbert_artifacts = False,
    reduce_noise = True
):
    '''
    Extract amplitude envelope of recordings. Attach ID info to the envelope. 
    ## Input: wav_path
    ## Output: df_entry
    '''
    
    sr = 48000
    data = int16_to_float32(data)
    if reduce_noise:
        data = nr.reduce_noise(y=data, sr=sr)
    
    ## Transform waveform into an analytic signal (has no negative frequency), 
    analytic_sig = hilbert(data)
    
    ## The absolute value of an analytic signal is a representative amplitude envelope
    amp_env = np.abs(analytic_sig)
    
    ## all numbers lower than a threshold becomes zero
    if hilbert_artifacts:
        amp_env[amp_env < 0.001 * np.std(amp_env)] = 0
    
    ## make compact support time series
    if compact:
        amp_env = compact_support(amp_env, stds = stds, buffer = buffer)
    
    ## convert to spl if needed
    if spl == True:
        ref_sound_pressure = 2*(10**(-5))
        amp_env = amp_env + ref_sound_pressure
        amp_env = 20 * np.log((amp_env/(ref_sound_pressure)))
    
    ## Rolling window average (boxcar window) 10 ms window
    amp_env = np.convolve(amp_env, get_window(win_type, int((sr*10)/1000)))
    
    ## resample the amp env to have 1 ms step
    amp_env = resample(amp_env, orig_sr = sr, target_sr = target_sr, res_type = res_type)
    
    ## normalize it to 0-1 if spl
    if spl == True:
        target_max = 1
        min_spl = np.min(amp_env)
        max_spl = np.max(amp_env)
        amp_env = np.array([(x/max_spl)*target_max for x in amp_env])
    
    return amp_env

def compact_support(amp_env, stds = 1, buffer = 500):
    '''
    Takes amplitude envelope and create compact support
    '''
    
    ## take standard deviation of the time-series
    amp_env_sd = np.std(amp_env)
    
    ## amplitude envelopes should start some number of standard deviations away from 0
    ae_indices = np.argwhere(amp_env > stds * amp_env_sd)
    strt = ae_indices[0][0] - buffer ## start and ends should have some buffers
    end = ae_indices[-1][0] + buffer
    
    ## check if strt and end are legal
    if strt < 0:
        strt = 0
    if end > (len(amp_env) - 1):
        end = (len(amp_env) - 1)
    
    return amp_env[strt:end]

def time_windows(amp_env, num = 50):
    '''
    Takes amplitude envelope and create log-scaled time windows from 1ms to full length of the song
    '''
    
    lag = np.unique(np.geomspace(1, len(amp_env), num = num).astype(int))
    
    return lag