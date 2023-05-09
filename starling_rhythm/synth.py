import numpy as np
from scipy.io.wavfile import read, write
from starling_rhythm.utils.paths import ensure_dir
import librosa
from collections import Counter

def save_synth_shuffle_batch(
    segment_address, syllable_df, bID_DIR, templating = False, gap_type = 'empirical', batchsize = 0, rate = 48000):
    '''
    Produce n amount of shuffled synths and save them
    '''
    for n in np.arange(0, batchsize):
        shuffle_syn = save_synth(segment_address, syllable_df, bID_DIR, templating = False, gap_type = 'shuffle', n = n)
        
    return

def save_synth_full(full_address, syllable_df, bID_DIR, templating = False, gap_type = 'empirical', n = 0, rate = 48000):
    '''
    Take the address of the full song and save its specified synthesis
    '''
    waveform = synth_song(full_address, syllable_df, bID_DIR, templating, gap_type)
    dir_name = 'FULLtemplating' + str(templating) + '-' + gap_type
    filename = str(full_address).split('/')[-1][:-4] + '_' + dir_name + '_' + str(n) + '_.wav'
    save_address = bID_DIR / dir_name / filename
    
    ensure_dir(save_address)
    write(save_address, rate, waveform)
    return

def save_synth(segment_address, syllable_df, bID_DIR, templating = False, gap_type = 'empirical', n = 0, rate = 48000, version = ''):
    '''
    Take the address of the segment and save its specified synthesis
    '''
    waveform = synth_song(segment_address, syllable_df, bID_DIR, templating, gap_type)
    dir_name = 'templating' + str(templating) + '-' + gap_type + str(version)
    filename = str(segment_address).split('/')[-1][:-4] + '_' + dir_name + '_' + str(n) + '_.wav'
    save_address = bID_DIR / dir_name / filename
    
    ensure_dir(save_address)
    write(save_address, rate, waveform)
    return

def synth_song(address, syllable_df, bID_DIR, templating = False, gap_type = 'empirical'):
    '''
    Take an address, and turn it into different types of synth_songs
    '''
    
    filename = str(address).split('/')[-1]
    filename_parts = filename.split('_')
    
    ## if address is a segment, record right query info
    if 'seg' in filename_parts:
        recover_file = ''
        for info in filename_parts:
            if info != 'seg':
                recover_file = recover_file + info + '_'
            else:
                recover_file = recover_file[:-1] + '.wav'
                break
        file = bID_DIR / recover_file
        
        strt = int(filename_parts[-2][:-2])
        end = int(filename_parts[-1][:-6])

    ## if address is not segment, default
    else:
        file = address
        strt = 0
        end = np.inf
        
    ## select the right portion of dataframe
    syllable_df = syllable_df[syllable_df['file'] == file]
    syllable_df = syllable_df[(syllable_df['onsets_ms'] > strt) & (syllable_df['offsets_ms'] < end)]
    
    ## retrieve the audio
    rate, audio = read(address)
    taper = 2*(rate/1000)
    
    ## make an empty container for the song
    min_onset = int(min(syllable_df.onsets_ms) * rate)
    max_offset = int(max(syllable_df.offsets_ms) * rate)
    container = np.zeros(max_offset-min_onset)
    
    ###
    #Manipulation
    ###

    template = {}
    onsets_ms = syllable_df.onsets_ms.values
    offsets_ms = syllable_df.offsets_ms.values
    hdbscan_labels = syllable_df.hdbscan_labels.values
    
    syllable_list = []
    
    ## construct syllable_list
    for onset, offset, label in zip(onsets_ms, offsets_ms, hdbscan_labels):
        onset = int((onset-strt) * rate)
        offset = int((offset-strt) * rate)
        
        ## if templating, use this to make a template
        if templating:
            try:
                ## throws error if there's not an existing list for that item
                template[label].append(audio[onset:offset])
            except:
                ## instantiate list
                template[label] = [audio[onset:offset]]
                    
        ## if not templating, simply just add
        else:
            syllable_list.append(audio[onset:offset])
            
    ## back propagate syllable_list if templating
    if templating:
        for onset, offset, label in zip(onsets_ms, offsets_ms, hdbscan_labels):
            onset = int((onset-strt) * rate)
            offset = int((offset-strt) * rate)
            ## do nothing if label is -1 or there is only one iteration of the syllable
            if (label == -1) or (len(template[label]) == 1):
                syllable_list.append(audio[onset:offset])
            else:
                ## if there is more than one iteration, pick one with the median length
                ## if there are two iterations, pick the first one
                if len(template[label]) == 2:
                    syllable_list.append(template[label][0])
                
                ## if there are more than two iterations, pick the median length
                if len(template[label]) > 2:
                    template_lengths = [len(waveform) for waveform in template[label]]
                    template[label] = [waveform for _, waveform in sorted(zip(template_lengths, template[label]), key = lambda pair: pair[0])]
                    syllable_list.append(template[label][int(len(template[label])/2)])
        
        
    ## construct gap_list
    gap_list = list((onsets_ms[1:] - offsets_ms[:-1]) * rate)
    
    gap_profile = {}
    if gap_type == 'transition_gap_profiles': 
        
        ## sort all transitions with their gaps
        for pre, post, index in zip(hdbscan_labels[:-1], hdbscan_labels[1:], np.arange(0, len(gap_list))):
            gap_id = str(pre) + '-' + str(post)
            try: 
                ## this will throw an error if there isn't an existing list in the dictionary
                gap_profile[gap_id].append(gap_list[index])
            except:
                gap_profile[gap_id] = [gap_list[index]]
                
        ## produce a dictionary with mean gaps
        for key in gap_profile:
            gap_profile[key] = int(np.mean(gap_profile[key]))
            
        gap_list = []
        ## for every transition, use the average (1 gets itself, anything above gets averaged)
        for pre, post in zip(hdbscan_labels[:-1], hdbscan_labels[1:]):
            gap_id = str(pre) + '-' + str(post)
            gap_list.append(gap_profile[gap_id])
            
    ###
    #Concatenate
    ###
    
    if gap_type == 'shuffle':
        np.random.shuffle(gap_list)
        
    ## make both list equal length
    gap_list.append(0)
    
    syn_song = []
    
    for syllable, gap in zip(syllable_list, gap_list):
        syn_song.append(taper_audio(syllable, taper = int(taper)))
        syn_song.append(np.zeros(int(gap)))
                        
    syn_song = np.concatenate(syn_song)
    
    ## normalize
    syn_song = librosa.util.normalize(syn_song)
    
    ## use int16 to save space
    syn_song = float32_to_int16(syn_song)
    
    return syn_song

def float32_to_int16(data, normalize = True):
    ## force into unit range
    if np.max(data) > 1:
        data = data / np.max(np.abs(data))
    
    return np.array(data * 32767).astype("int16")

def taper_audio(audio, taper):
    '''
    Linearly fade audio in/out
    note: taper is in samples
    '''
    
    ## taper audio
    fade_in = audio[0:taper]*np.linspace(0, 1, num = taper)
    fade_out = audio[-taper:]*np.linspace(1, 0, num = taper)
            
    audio[0:taper] = fade_in
    audio[-taper:] = fade_out
    
    return audio

def song_segments(address, syllable_df, SAVE_DIR, window_size = 10, strict = False):
    '''
    Takes a wav address, and write eligible segments to disk
    '''
    
    ## find all syllables of the bout
    bout = syllable_df[syllable_df['file'] == address]
    
    ## terminal syllable end
    terminal = max(bout['offsets_ms'].values)
    
    ## window
    windows_strt = np.arange(0, terminal - window_size + 1, step = 5)
    windows_end = np.arange(window_size, terminal + 1, step = 5)
    
    ## slide window
    for strt, end in zip(windows_strt, windows_end):
        
        ## between this range, pull out all the syllables 
        bout_slice = bout[(bout['onsets_ms'] > strt) & (bout['offsets_ms'] < end)] 
        
        ## pull out all the syllable labels
        sequence = bout_slice['hdbscan_labels'].values
        
        '''
        Discard rules
        '''
        
        ## discard all segments that don't have at least 10 syllables
        if len(sequence) < 10:
            continue
        
        ## discard all segments that don't have repetition of syllables
        
        #### get reptition counts
        uniques, counts = np.unique(sequence, return_counts = True)
        
        #### any unlabeled syllables are not in this search
        if np.any(uniques == -1):
            uniques = uniques[1:]
            counts = counts[1:]
            
        #### if there no repeating syllables skip
        if len(np.unique(counts)) <= 1:
            continue
            
        #### if strict, make sure that there are at least 5 syllable types that repeat
        if strict:
            
            repeated_syllable_counts = np.delete(counts, np.where(counts == 1))
            
            ## eliminate all segments that had less than 3 syllable types that repeat
            if len(repeated_syllable_counts) < 3:
                continue
            
            transition_list = (list(zip(sequence[:-1], sequence[1:])))
            transition_counts = list(dict(Counter(transition_list).items()).values())
            transition_counts = np.int64(transition_counts)
            repeated_transition_counts = np.delete(transition_counts, np.where(transition_counts == int(1)))
            
            ## eliminate all segments that had less than 3 transition types that repeat
            if len(repeated_transition_counts) < 3:
                continue
            
        ## if you made it this far, congrats! You are to be saved. 
        basename = str(address).split('/')[-1].split('.')[0]
        filename = basename + '_seg_' + str(strt) + '_' + str(end) + '.wav'
        save_address = SAVE_DIR / filename
        rate, data = read(address)
        ensure_dir(save_address)
        write(save_address, rate, data[int(strt*rate):int(end*rate)])
        
    return