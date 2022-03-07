import pandas as pd
import numpy as np
from pathlib2 import Path

import seaborn as sns
import matplotlib.pyplot as plt

from butcherbird.utils.paths import DATA_DIR

def get_file_nm (path): 
    return path.__str__().split('/')[-1]

def key_to_bird_nm (key):
    
    rcd_loc = DATA_DIR/"raw/recording_wav/"
    tg_loc = DATA_DIR/"raw/textgrid/"
    
    ## use key to find wav
    wav_path = list(rcd_loc.glob(key+ "*"))
    
    ## use wav to find tg
    ## get recording id
    wav_id = get_file_nm(wav_path).split('.')[0]
    ## use glob to find all paths of textgrids
    tg_path = list (tg_loc.glob(wav_id + "*"))
    
    ## use tg to find name
    return get_file_nm(tg_path).split('_')[-1].split('.')[0]

def simple_dyadic(syllable_df, key):
    ''' This function turns a butcherbird syllable_df into unsorted Roeske et al. 2020 dyadic table
        Note: this function is limited to one wav at a time
    '''
    
    indv = key_to_bird_nm(key)
    
    ## filter out unnecessary variables
    interval_df = syllable_df.filter(items = ['note_strt', 'indv', 'key'])
    
    ## get a list of indv's note_strt
    indv_note_strt = interval_df[interval_df['key'] == key]['note_strt']
    
    ### find all onset intervals from list of note_strt
    ## use a counter design
    i = 0
    indv_intervals = []

    ## for every note
    for onset in indv_note_strt:

        ## if at last note, exit for loop
        if i == (len(indv_note_strt) - 1):
            continue

        ## interval = next onset - current onset, add to interval list
        interval = indv_note_strt[i + 1] - indv_note_strt[i]
        indv_intervals.append(interval)
        
        ## counter up
        i = i + 1
        
    ### collect into dyadic formation
    ## interval 1 removes last interval
    indv_intervals1 = list(indv_intervals)
    del indv_intervals1[-1]
    ##interval 2 removes first interval
    indv_intervals2 = list(indv_intervals)
    del(indv_intervals2[0])
    
    ## put intervals into dataframe
    d = {'interval1': indv_intervals1, 'interval2': indv_intervals2, 'indv': indv}
    dyadic = pd.DataFrame(data = d)
    
    return dyadic
    
def intervals_to_dyadic (interval_list):

    ### collect into dyadic formation
    ## interval 1 removes last interval
    intervals1 = list(interval_list)
    del intervals1[-1]
    ##interval 2 removes first interval
    intervals2 = list(interval_list)
    del(intervals2[0])
    
    ## put intervals into dataframe
    d = {'interval1': intervals1, 'interval2': intervals2}
    dyadic = pd.DataFrame(data = d)
    
    return dyadic
    
def sort_dyadic (dyadic):
    ''' This function turns a butcherbird syllable_df into Roeske et al. 2020 dyadic table
        This version does not sort
        Note: this function is limited to one wav at a time
    '''
    
    ### sort all intervals and calculate necessary components
    s_interval = []
    l_interval = []
    cycle_dur = []
    ratio_custom = []
    ratio_roeske = []
    indv = []
    
    ## for every dyadic
    for index, row in dyadic.iterrows():

        i1 = row['interval1']
        i2 = row['interval2']
        indv.append(row['indv'])

        ## short long decider
        if i1 > i2:
            s = i2
            l = i1
        else:
            s = i1
            l = i2
            
        ## calculate components
        s_interval.append(s)
        l_interval.append(l)
        cycle_dur.append(s + l)
        ratio_roeske.append(i1/(i1+i2))
        ratio_custom.append(s/l)

    ## push into sorted dataframe
    d = {'s_interval': s_interval, 
         'l_interval': l_interval, 
         'cycle_dur': cycle_dur, 
         'ratio_roeske': ratio_roeske, 
         'ratio_custom': ratio_custom, 
         'indv': indv}
    
    dyadic_sorted = pd.DataFrame(data = d)
    
    ## filter out transition between phrases
    dyadic_sorted = dyadic_sorted[dyadic_sorted['l_interval'] < 1]
    dyadic_sorted = dyadic_sorted[dyadic_sorted['s_interval'] > 0]
    
    ## sort ascending by shortest cycle to longest cycle
    dyadic_sorted = dyadic_sorted.sort_values(by = ['cycle_dur'])
    
    ## put cycle rank into data frame
    cycle_rank = dyadic_sorted.rank()['cycle_dur'].astype(int)
    dyadic_sorted['cycle_rank'] = cycle_rank
    
    #### return sorted dyadic table
    return dyadic_sorted

def anon_sort_dyadic (dyadic):
    ''' This function turns a butcherbird syllable_df into Roeske et al. 2020 dyadic table
        This version does not include indv data
        Note: this function is limited to one wav at a time
    '''
    
    ### sort all intervals and calculate necessary components
    s_interval = []
    l_interval = []
    cycle_dur = []
    ratio_custom = []
    ratio_roeske = []

    ## for every dyadic
    for index, row in dyadic.iterrows():

        i1 = row['interval1']
        i2 = row['interval2']

        ## short long decider
        if i1 > i2:
            s = i2
            l = i1
        else:
            s = i1
            l = i2
            
        ## calculate components
        s_interval.append(s)
        l_interval.append(l)
        cycle_dur.append(s + l)
        ratio_roeske.append(i1/(i1+i2))
        ratio_custom.append(s/l)

    ## push into sorted dataframe
    d = {'s_interval': s_interval, 
         'l_interval': l_interval, 
         'cycle_dur': cycle_dur, 
         'ratio_roeske': ratio_roeske, 
         'ratio_custom': ratio_custom
        }
    
    dyadic_sorted = pd.DataFrame(data = d)
    
    ## filter out transition between phrases
    dyadic_sorted = dyadic_sorted[dyadic_sorted['l_interval'] < 1000]
    dyadic_sorted = dyadic_sorted[dyadic_sorted['s_interval'] > 0]
    
    ## sort ascending by shortest cycle to longest cycle
    dyadic_sorted = dyadic_sorted.sort_values(by = ['cycle_dur'])
    
    ## put cycle rank into data frame
    cycle_rank = dyadic_sorted.rank()['cycle_dur'].astype(int)
    dyadic_sorted['cycle_rank'] = cycle_rank
    
    #### return sorted dyadic table
    return dyadic_sorted

def single_dyadic (syllable_df, key):
    ''' Suitable if only one wav needs to be sorted
    '''
    return sort_dyadic(simple_dyadic(syllable_df, key))