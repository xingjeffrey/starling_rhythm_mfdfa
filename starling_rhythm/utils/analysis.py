from tqdm.autonotebook import tqdm
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

def test_filter(behav_data, accuracy_threshold = 0.8, past_baseline_trials = 100):
    '''
    filter all test trials that has above accuracy threshold for some amount of baseline trials. 
    
    args: 
        behav_data (dict): behav.loading.load_data_pandas object
        accuracy_threshold (float): cut-off for filter
        past_baseline_trials (int): number of baseline trials to look back
        
    returns:
        test_data (dict): all valid test trials for analysis
        training_data (dict): baseline trials that precede valid test trials
    '''
    
    test_data = {}
    training_data = {}
    
    ## for each subject in the dictionary
    for subj in behav_data.keys():
        numbered_trials = behav_data[subj].reset_index() 
        test_trials = numbered_trials[numbered_trials.type_ == 'test']
        baseline_trials = numbered_trials[numbered_trials.type_ == 'normal']

        valid_test_trial_indices = []
        valid_baseline_trial_indices = []

        ## iterate through each test trial
        for i, trial in tqdm(test_trials.iterrows(), desc = 'Iterating through test trials for subject ' + str(subj)):

            ## find the previous baseline_trials
            baseline_bucket = baseline_trials.loc[:i].iloc[-past_baseline_trials:]
            training_accuracy = np.mean(baseline_bucket.correct)

            ## if training accuracy exceed threshold, append test_trial index, and 
            if training_accuracy > accuracy_threshold:
                valid_test_trial_indices.append(i)
                valid_baseline_trial_indices = valid_baseline_trial_indices + list(baseline_bucket.index)

        test_data[subj] = test_trials.loc[valid_test_trial_indices]
        valid_baseline_trial_indices = list(set(valid_baseline_trial_indices))
        training_data[subj] = baseline_trials.loc[valid_baseline_trial_indices]
        
    return test_data, training_data

def stim_parser(behav_data):
    '''
    parse out the stimuli info
    
    args:
        behav_data(dict): behav.loading.load_data_pandas object
        
    returns: 
        behav_data(dict): same object but with additional columns added
    '''
    
    ## for each subj
    for subj in behav_data.keys():
        
        stim_types = []
        pair_indices = []
        inter_nums = []
        
        for i, row in tqdm(behav_data[subj].iterrows(), desc = 'Output stim labels for ' + subj):
            parsed = row.stimulus.split('_')
            if row.type_ == 'test':
                stim_types.append(parsed[2])
                pair_indices.append(parsed[3])
                inter_nums.append(int(parsed[4].split('.')[0]))
            else:
                stim_types.append('training')
                pair_indices.append(parsed[1])
                inter_nums.append(parsed[2].split('.')[0])
            
        behav_data[subj]['stim_type'] = stim_types
        behav_data[subj]['pair_indices'] = pair_indices
        behav_data[subj]['inter_nums'] = inter_nums
        
    return behav_data   

def acquisition_data_preprocessing(
    behav_data, 
    end_date = None, 
    groupby_list = ['day']
):
    '''
    Prepare data for acquisition data plot
    
    args:
        behav_data = behav.loading.load_data_pandas object
        start_date = datetime.date for acquisition start (default first training)
        end_date = datetime.date for acquisition end (default today)
        
    return:
        acquisition_data = accuracy per stimuli per day
    '''
    
    training_data = {}
    acquisition_data = {}
    
    ## for every subject
    for subj in behav_data.keys():
        
        ## default start date to the day of first trial
        start_date = behav_data[subj].index[0].date()

        ## if no end date is set, default to today
        if end_date is None:
            end_date = datetime.date.today()
        
        ## isolate just training trials
        training_data[subj] = behav_data[subj][behav_data[subj].type_ == 'normal']
        
        ## assign day training
        training_data[subj]['day'] = [
            (timestamp.date() - start_date).days for timestamp in training_data[subj].index
        ]
        
        ## calculate daily accuracy
        acquisition_data[subj] = pd.DataFrame(training_data[subj].groupby(
            groupby_list
        )['correct'].agg(['mean', 'count'])).reset_index()
        
        acquisition_data[subj]['subject'] = subj
    
        ## calculate binomial CI
        binomial_lower_ci = []
        binomial_upper_ci = []

        for i, row in acquisition_data[subj].iterrows():

            try:
                #print(int(row['mean'] * row['count']))
                lower, upper = proportion_confint(int(row['mean'] * row['count']), row['count'], alpha = 0.05)
            except:
                lower = np.nan ## if binomial confidence interval cannot be calculated, use nan
                upper = np.nan

            binomial_lower_ci.append(lower)
            binomial_upper_ci.append(upper)
        
        acquisition_data[subj]['binomial_lower'] = binomial_lower_ci
        acquisition_data[subj]['binomial_upper'] = binomial_upper_ci
    
    acquisition_data = pd.concat(acquisition_data)
    
    return acquisition_data

def plot_subject_acquisition(
    behav_data, 
    end_date = None, 
    groupby_list = ['day'],
    figsize = (8, 4),
    dpi = 300,
    
):
    '''
    Plot acquisition curve by subject
    '''
    
    acquisition_data = acquisition_data_preprocessing(
        behav_data,
        end_date, 
        groupby_list = ['day']
    )
    
    plt.figure(figsize = figsize, dpi = dpi)
    plt.axhline(y = 0.5, linestyle = '--', color = 'black', alpha = 0.5)
    plt.axhline(y = 0.8, linestyle = '--', color = 'red', alpha = 0.5)
    plt.ylim(0, 1)
    
    for subj in behav_data.keys():
        subj_acqui = acquisition_data.loc[subj]
        plt.plot(
            subj_acqui.day,
            subj_acqui['mean'],
            linewidth = 2,
            alpha = 1,
            label = subj
        )
        
        plt.fill_between(
            x = subj_acqui.day,
            y1 = subj_acqui.binomial_lower,
            y2 = subj_acqui.binomial_upper,
            alpha = .2
        )
        
        plt.xlabel('Training Progress (Days)')
        plt.ylabel('Accuracy')
        
        plt.legend()