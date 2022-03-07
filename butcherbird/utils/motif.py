import pandas as pd
import numpy as np

from butcherbird.visualization import sequential

from tqdm.autonotebook import tqdm

## Main Wrapper
def motif_extractor(note_df, label = 'hdbscan_labels', motif_threshold = 0.8):
    '''
    Take a note_df and automatically find motif boundaries using first-order transition properties. 
    '''
    
    print('// Starting Motif Extraction... //')
    
    print('// Getting transition probability for each first-order transition... //\n')
    first_order = first_order_prob(note_df, label = label)
    
    print('\n// Finding index of high probability transitions... //')
    motif_index = high_transition_index(first_order, threshold = motif_threshold)
    
    print('// Retrieving motif_strt and motif_end times... //')
    motif_strt_ids, motif_end_ids, motif_strt_times, motif_end_times = motif_detector(note_df, motif_index)
    
    print('// Constructing motif_df... //')
    motif_df = motif_df_constructor(note_df, motif_strt_ids, motif_end_ids, motif_strt_times, motif_end_times)
    
    print('// DONE! //')
    
    return motif_df
    
        
## Helpers
def first_order_prob(note_df, label):
    '''
    Return a list of transition probabilities as they occur in the sequence
    '''
    ## Find Transition Matrix of notes
    trans_mat = sequential.construct_seq_matrix(note_df, label = label, sort = False)
    
    ## Build list of first-order transitions
    current_list = note_df[label].values[:-1]
    next_list = note_df[label].values[1:]
    
    # Create container of first-order transition
    first_order = []
    
    ## for each first-order transition, find its transition probability
    for curr_note, next_note in zip (current_list, next_list):
        ## query transition probability of pair and append it to the container
        first_order.append(trans_mat.loc[curr_note, next_note])
        
    return first_order

def high_transition_index(first_order, threshold):
    '''
    Return an index of transitions that are beyond a certain threshold
    '''
    return np.argwhere(np.array(first_order) > threshold)

def motif_detector(note_df, motif_index):
    '''
    Contain decision tree for motif detection. Return a list of motif_strt and motif_end ids, and motif_strt and motif_end times.
    '''
    
    motif_strt_ids = []
    motif_end_ids = []
    motif_strt_times = []
    motif_end_times = []
    
    # for each note of note_df
    for note_id in tqdm(range(len(note_df) - 1)):
        
        ## if construction started
        if len(motif_strt_times) > 0:
            # if note strt of current note is before previous motif end, skip to the next viable motif start point
            if note_df.loc[note_id, 'note_strt'] < motif_end_times[-1]:
                continue
                
        ## designate note_id as motif_strt
        motif_strt = note_id
        
        ## if note is identified to have high transition probability to its next note
        if note_id in motif_index:
            
            ## designate next note_id as motif_end
            motif_end = note_id + 1
            
            ## check if the next transition also qualifies
            while note_id + 1 in motif_index:
                
                ## update motif_end continuously
                motif_end += 1
                ## look for next transition
                note_id += 1
                
        ## if transition is NOT identified as high probability
        else:
            ## designate note as motif
            motif_end = note_id
            
        motif_strt_ids.append(motif_strt)
        motif_end_ids.append(motif_end)
        motif_strt_times.append(note_df.loc[motif_strt, 'note_strt'])
        motif_end_times.append(note_df.loc[motif_end, 'note_end'])
        
    return motif_strt_ids, motif_end_ids, motif_strt_times, motif_end_times

def filler(length, dtype, fill_detail):
    ## construct list of designated length
    l = np.empty(length, dtype = dtype)
    ## fill with value
    l.fill(fill_detail)
    
    return list(l)

def motif_df_constructor(note_df, motif_strt_ids, motif_end_ids, motif_strt_times, motif_end_times):
    '''
    Make motif_df
    '''
    
    ## set up container for motifs within each phrase
    motif_dfs = []
    
    ## For every phrase detail its motif content
    for phrase_nb in tqdm(np.unique(note_df['phrase_nb'])):
        
        ## locate designated phrase
        cur_phrase = note_df[note_df['phrase_nb'] == phrase_nb]
        
        ## find phrase constrain
        phrase_strt = cur_phrase['phrase_strt'].values[0]
        phrase_end = cur_phrase['phrase_end'].values[0]
        
        ## find motif indices that correspond within the phrase contrains
        motif_indices = np.where(
            (np.array(motif_strt_times) >= phrase_strt) &
            (np.array(motif_end_times) <= phrase_end)
        )[0]
        
        ## query onsets of offsets of motifs 
        motif_strt_list = np.take(motif_strt_times, motif_indices)
        motif_end_list = np.take(motif_end_times, motif_indices)
        
        motif_cnt = len(motif_indices)
        
        indv = cur_phrase['indv'].values[0]
        
        ## start dataframe construction
        d = {
            'phrase_nb': filler(motif_cnt, int, phrase_nb),
            'phrase_strt': filler(motif_cnt, float, phrase_strt),
            'phrase_end': filler(motif_cnt, float, phrase_end),
            'phrase_len': filler(motif_cnt, float, phrase_end - phrase_strt),
            'motif_cnt': filler(motif_cnt, int, motif_cnt),
            'motif_nb': list(motif_indices),
            'note_cnt': filler(motif_cnt, float, phrase_end - phrase_strt),
            'note_nb': list(np.take(motif_strt_ids, motif_indices)),
            'motif_strt': motif_strt_list,
            'motif_end': motif_end_list,
            'motif_len': motif_end_list - motif_strt_list,
            'indv': filler(motif_cnt, object, cur_phrase['indv'].values[0]),
            'indvi': filler(motif_cnt, int, cur_phrase['indvi'].values[0]),
            'key': filler(motif_cnt, object, cur_phrase['key'].values[0])
        }
        
        ## add to collection
        motif_dfs.append(pd.DataFrame(d))
        
    motif_df = pd.concat(motif_dfs)
    
    return motif_df
            