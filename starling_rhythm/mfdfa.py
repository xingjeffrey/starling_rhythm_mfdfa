import scipy
import numpy as np
from starling_rhythm.utils.audio import compact_support, time_windows
from MFDFA import MFDFA
from starling_rhythm.utils.audio import extract_amp_env
from starling_rhythm.iaaft import surrogates
from scipy.stats import ttest_1samp
import pathlib2

def tmf(
    waveform, 
    amp_env_sr, 
    win_type = 'boxcar', 
    res_type = 'linear',
    stds = 1,
    buffer = 100,
    spl = True,
    compact = False,
    time_window_cnt = 50, 
    q_min = 0.5, 
    q_max = 5, 
    q_num = 10,
    ns = 32, 
    tol_pc = 5.,
    verbose = False,
    maxiter = 1E6,
    sorttype = 'quicksort',
    return_p = False,
    hilbert_artifacts = True, 
    z = False,
    reduce_noise = True
):
    '''
    Analyze the t-statistic of the waveform's empirical MFDFA against 32 IAAFT surrogates. 
    '''
    
    if type(waveform) == pathlib2.PosixPath:
        sr, waveform = scipy.io.wavfile.read(waveform)
    
    ## transform into logscaled amplitude envelope
    amp_env = extract_amp_env(
        waveform, 
        target_sr = amp_env_sr, 
        win_type = win_type,
        res_type = res_type,
        stds = stds, 
        buffer = buffer,
        spl = spl,
        compact = compact,
        hilbert_artifacts = hilbert_artifacts,
        reduce_noise = reduce_noise
    )
    
    ## calculate empirical hurst_expos
    h_expos = hurst_expo(
        amp_env,
        time_window_cnt = time_window_cnt, 
        q_min = q_min, 
        q_max = q_max, 
        q_num = q_num    
    )
    
    ## MF_range
    empirical_MF_range = max(h_expos) - min(h_expos)
    
    ## calculate IAAFT surrogates
    iaaft_surrogates = surrogates(
        amp_env, 
        ns = ns, 
        verbose = verbose, 
        maxiter = maxiter, 
        sorttype = 'quicksort'
    )
    
    ## for each surrogate, find their MF-range
    
    SDoMF = []
    
    for surrogate in iaaft_surrogates:
        iaaft_h_expos = hurst_expo(
            surrogate,
            time_window_cnt = time_window_cnt, 
            q_min = q_min, 
            q_max = q_max, 
            q_num = q_num   
        )
        SDoMF.append(max(iaaft_h_expos) - min(iaaft_h_expos))
        
    ## calculate one sample T of empirical_MF_range to SDoMF
    t_test = ttest_1samp(a = SDoMF, popmean = empirical_MF_range)
    t = -t_test[0] ## flip sign
    p = t_test[1]
    
    if z:
        zmf = (empirical_MF_range - np.mean(SDoMF))/np.std(SDoMF)
        return zmf
    
    if return_p:
        return t, p
    
    else:
        return t

def hurst_expo(
    amp_env, 
    time_window_cnt = 50, 
    q_min = 0.5, 
    q_max = 5, 
    q_num = 10
):
    '''
    Save MF-DFA values
    '''
    
    ## retrieve time windows
    lag = time_windows(amp_env, num = time_window_cnt)
    
    ## q_range
    q_range = np.linspace(q_min, q_max, q_num)
    
    ## Prep containers
    h_expo = []
    
    # MFDFA
    ## For each q, generate its dfa points and estimate the hurst exponent
    for q in q_range:
        lag, dfa = MFDFA(timeseries = amp_env, lag = lag, q = q, order = 1) ## linear detrend
        h_expo.append(np.polyfit(np.log(lag), np.log(dfa), 1)[0][0])
    
    return h_expo