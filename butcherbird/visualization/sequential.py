import pandas as pd
import numpy as np

from scipy.stats import entropy
import math

import random

import seaborn as sns

from tqdm.notebook import tqdm

## Wrappers
def construct_seq_matrix(indv_df, label = 'hdbscan_labels', sort = True):
    '''
    Construct Transition Matrix according to specifications
    '''
    if sort == True:
        return rank_transition_matrix(transition_matrix(sequencer(indv_df, label = label)))
    if sort == False:
        return transition_matrix(sequencer(indv_df, label = label))

## Data Structure

def sequencer(indv_df, label = 'hdbscan_labels'):
    '''
    Takes dataframe containing indv information, and then constructed nested sequence structure. 
    '''
        
    return list(indv_df[label].values)

def transition_matrix(seq_container):
    '''
    Parses sequences into transition probability in form of nested ndarray
    '''

    ## Get all the unique states that the bird can transition from
    ## before you can do that, please unfold the seq_container
    unique_states = np.unique(seq_container)
    print(unique_states)

    ## set up transition matrix itself, and fill them with zeros to
    ## indicate that the baseline that no states transitions to each other
    transition_matrix = np.zeros((np.max(unique_states) + 1, np.max(unique_states) + 1))

    ## use range so that last one is controlled
    for note_nb in range(len(seq_container) - 1):
        ## slot into matrix as transition_matrix[incoming][outgoing] + 1 tally
        incoming = seq_container[note_nb]
        outgoing = seq_container[note_nb + 1]

        ## add tally 1
        transition_matrix[incoming][outgoing] += 1
    
    transition_df = pd.DataFrame(transition_matrix)

    return transition_df.divide(transition_df.sum(axis = 1), axis = 'rows')

def rank_transition_matrix(transition_df):
    '''
    For every incoming note, sort its transition probability to outgoing in descending order. 
    '''
    
    ## start sorting!!
    sorted_transition_matrix = []

    ## for each incoming
    for row in transition_df.iterrows():
        ## grab the transition probabilities that it pairs with outgoing
        incoming_markov = list(row[1])

        ## employ wizardry (using negative to sort ascending, then flip back)
        sorted_transition_matrix.append(-np.sort(-np.array(row[1])))

    sorted_transition_df = pd.DataFrame(sorted_transition_matrix)
    
    return sorted_transition_df

def long_cvtr (df, songType):
    '''
    Crunch wide to long data
    '''
    temp_holder = []
    incoming_index = 0

    ## for every row in a dataframe
    for row in df.iterrows():
        incoming_p = list(row[1])
        
        markov_index = 0
        
        ## for every probability in the dataframe
        for p in incoming_p:
            long_data = [songType, incoming_index, markov_index, p]
            temp_holder.append(long_data)
            markov_index +=1

        incoming_index += 1

    return pd.DataFrame(data = temp_holder, columns = ['songType', 'noteType', 'markov', 'p'])

## Constructors
def syn_seq_gen(indv_df, label = 'hdbscan_labels', order = 'zero'):
    '''
    This function takes in the unique states of a Markovian sequence as well as its P distribution, 
    and generates a zero-order synthetic song.
    '''
    # Construct deep copy of indv_df
    indv_df_cc = indv_df.copy(deep = True)
        
    # Get info 
    unique_states, counts_states = np.unique(indv_df_cc[label].values, return_counts = True)
    p_states = counts_states / len(indv_df_cc[label].values)
        
    # Zero order markov selection
    syn = []
        
    if order == 'zero':
        ## for every note, draw a zero-order markov prediction
        for note in indv_df_cc[label].values:
            syn.append(random.choices(unique_states, p_states)[0])
               
    if order == 'first':
        true_transitions = construct_seq_matrix(indv_df, label = label, sort = False)
        
        ## generate first syllable
        current = random.choices(unique_states, p_states)[0]
        
        ## generate the rest with first-order markov
        for stateid in tqdm(range(len(indv_df_cc[label]))):
           
            ## add current syllable to rand
            syn.append(current)
                
            ## locate transition probability for current syllable
            future_markov = true_transitions.transpose()[current].values
                
            ## locate future syllable and set as current
            current = random.choices(unique_states, future_markov)[0]
                
    return syn

def markov_model (indv_df, order = 'original', label = 'hdbscan_labels', sort = True):
    '''
    Construct a Markovian transition matrix according to specifications
    '''
    if order == 'original':
        return construct_seq_matrix(indv_df, label = label, sort = sort)
        
    else:
        # Construct deep copy of indv_df
        indv_df_cc = indv_df.copy(deep = True)
        
        # Get info 
        unique_states, counts_states = np.unique(indv_df_cc[label].values, return_counts = True)
        p_states = counts_states / len(indv_df_cc[label].values)
        
        # Zero order markov selection
        syn = syn_seq_gen(indv_df_cc, label = label, order = order)
            
        ## refresh label column with synthetic labels
        indv_df_cc[label] = syn
        
        ## push final producted to be sorted
        return construct_seq_matrix(indv_df_cc, label = label, sort = sort)
        
        
        
## Normalized Block Entropy
def get_seq_states(seq):
    '''
    This function returns the unique states and their occurence probability
    
    returns unique_states, p_states
    '''
    
    ## get unique_states and tally each state
    unique_states, counts_states = np.unique(seq, return_counts = True)
    
    ## transform tally into probability
    p_states = counts_states / len(seq)
    
    return unique_states, p_states

def seq_n_block(seq, n):
    '''
    This function returns a song in blocks of n syllables
    '''
    
    blocked_seq = []

    ## first create two staggered lists, then unpack list immediately with zip(*)
    ## end result is that each syllable is followed by its neighbor
    for block in zip(*[seq[i:] for i in range(n)]):
        
        ## if block size is smaller or equal than 1, append to blocked_song as str and continue
        if n <= 1:
                blocked_seq.append(str(block[0]))
                continue
        
        ## represent the entire syllable as a string separated by comma
        string_rep = [str(block[0])]
        ## for every other entry of the block, convert it into str and append it
        
        for i in range(1, n):
            string_rep.append(str(block[i]))
        
        blocked_seq.append('|'.join(string_rep))
        
    return np.array(blocked_seq)

def seq_block_entropy(sequence, seq_type, max_block_size):
    '''
    This function outputs a dataframe that calculates normalized block entropy for a song within a specific block size max
    '''
    
    seq_block_entropy = pd.DataFrame()
    
    ## save a copy of block 1 info
    seq_unique_states_b1, seq_p_states_b1 = get_seq_states(seq_n_block(sequence, 1))
    
    ## retrieve song info for that specific block
    for block_size in range(1, max_block_size):
        
        ## make song
        seq_block = seq_n_block(sequence, block_size)
        
        ## get unique states and probability distribution in a song
        seq_unique_states, seq_p_states = get_seq_states(seq_block)
        
        ## calculated normalized block entropy
        seq_entropy = norm_block_entropy(seq_unique_states_b1, seq_p_states, block_size)## remove b1 after??
        
        ## Record it in indv_entropy
        seq_block_entropy = seq_block_entropy.append(
            pd.DataFrame(
                data = {
                    'song_type' : [seq_type], 
                    'song': [seq_block],
                    'block_size' : [block_size], 
                    'unique_states': [seq_unique_states],
                    'p_states' : [seq_p_states], 
                    'normalized_block_entropy' : [seq_entropy]
                }
            )
        )
        
    return seq_block_entropy

def norm_block_entropy(unique_states, p_states, block_size):
    '''
    This function calculates the normalized entropy (efficiency) of a song sequence 
    given its unique states and probability
    
    '''

    #entropy_calc = entropy(p_states, base = len(unique_states)) / math.log(len(p_states), len(unique_states)) 
    entropy_calc = entropy(p_states) / math.log(len(p_states)) ## standard 
    #entropy_calc = entropy(p_states) / math.log(len(unique_states)) ## markowitz
    
    #p_states = uniquestate^blocksize
    
    return entropy_calc

def uniform_p_dist(unique_states):
    '''
    This function returns a numpy array containing a uniform probability distribution
    
    returns:
        np.array of a probability distribution
    '''
    
    ## first, construct dummy container
    dist = []

    ## for each unique state, generate a uniform probability
    for state in range(0, len(unique_states)):
        dist.append(1/len(unique_states))

    ## convert to numpy array
    return np.array(dist)
