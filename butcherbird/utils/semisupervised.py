import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
from joblib import Parallel, delayed
from librosa.util import pad_center
from tqdm.autonotebook import tqdm
import butcherbird.utils.label as lb

from sklearn.neighbors import NearestNeighbors

## UTILITY FUNCTIONS
def attach_tracers(syllable_df):
    '''
    Attach tracers to syllable_df by resetting index
    '''
    
    syllable_df.reset_index(drop = True)
    return syllable_df

def save_prog(syllable_df, save_loc):
    '''
    save to designated location as a pickle
    '''
    
## EXTRACTORS
def label_prototypes(syllable_df, dim = [10, 10], method = 'hdbscan_labels', display = 'spectrogram'):
    '''
    Give an overview of average spectrogram per label
    Set method to 'hdbscan_labels' or 'supervised_hdbscan'
    '''
    plt.figure(figsize=(16, 16))

    unique_clusters = np.unique(syllable_df[method])
    
    for cluster in unique_clusters:
        title = str(['Cluster', cluster, len(syllable_df[syllable_df[method] == cluster])])
        if cluster >=0:
            lab_df = syllable_df[syllable_df[method] == cluster]
            plt.subplot(dim[0], dim[1], cluster+1)
            plt.title(title)
            plt.axis('off')
            
            specs = lab_df[display].values
            
            syll_lens = [np.shape(i)[1] for i in specs]
            pad_length = np.max(syll_lens)
            
            with Parallel(n_jobs=-1, verbose=0) as parallel:
                specs = parallel(
                    delayed(pad_center)(spec, pad_length)
                    for spec in specs
                )
            
            with Parallel(n_jobs=-1, verbose=0) as parallel:
                specs = parallel(
                    delayed(pad_center)(spec, pad_length)
                    for spec in specs
                )
            
            plt.imshow(np.average(np.array(specs), axis = 0), origin='lower') 
            
def spec_gallery(syllable_df, page = 0, display_lim = 100, dim = [10, 10], method = 'hdbscan_labels', display = 'spectrogram'):
    '''
    For syllable_df, display designated spectrograms
    '''

    ## set up canvas
    plt.figure(figsize=(16, 16))

    ## get start and end
    spec_strt = page * 100
    spec_end = display_lim * (page + 1)
    
    ## find slice of df that contain label   
    specs = syllable_df[display].values[spec_strt:spec_end]
    indices = syllable_df.index[spec_strt:spec_end]
    labels = syllable_df[method].values[spec_strt:spec_end]
    
    plt.suptitle(method + str(np.unique(labels)) + display)
    
    ## set up counter
    i = 0
    
    for spec, index, label in zip(specs, indices, labels):
        
        ## counter
        i = i + 1
        
        ## plot spec
        plt.subplot(dim[0], dim[1], i)
        plt.axis('off')
        plt.title(str(index), fontsize = 8, pad = -18)
        plt.imshow(spec, origin='lower')
            
def label_gallery(syllable_df, label, page = 0, display_lim = 100, dim = [10, 10], method = 'hdbscan_labels', display = 'spectrogram'):
    '''
    For specified label, display designated spectrograms
    '''
    
    # Setup
    label_df = syllable_df[syllable_df[method] == label]
    
    ## find slice of df that contain label   
    spec_gallery(label_df, page, display_lim, dim, method, display)
        
def trace_audio(syllable_df, index):
    ''' 
    display audio as ipy audio player
    '''
    loc = syllable_df[syllable_df.index == index]
    loc
    
    return ipd.Audio(
        data=loc['audio'].values[0], 
        rate=loc['rate'].values[0]
    )


### Correcters
def label_rectifier(syllable_df, index, correct_label, label_type = 'hdbscan_labels', verbose = False):
    '''
    correct HDBSCAN label in the supervised_hdbscan column
    '''
    
    if verbose == True:
        old_label = syllable_df.at[index, label_type]
        print('For index #', index, ', changed', old_label, 'to', label_type, correct_label)
        
    syllable_df.at[index, label_type] = correct_label
    
    return syllable_df

def label_merger(syllable_df, label1, label2, label_type = 'hdbscan_labels'):
    '''
    Merge label1 and label2 together
    Let label2 adopt label1
    '''
    
    ## get items that are labeled as label 2 and retrieve their indices
    indices = syllable_df[syllable_df[label_type] == label2].index
    
    
    for index in indices:
        syllable_df = label_rectifier(
            syllable_df, 
            index = index, 
            correct_label = label1, 
            label_type = label_type, 
            verbose = True
        )
    
    return syllable_df


def cluster_unlabeled(syllable_df, label_type = 'hdbscan_labels', umap = 'umap', algorithm = 'brute', verbose = False):
    '''
    Find all unclustered labels and group them with nearest cluster
    '''
    
    ## find all unclustered labels
    unlabeled_df = syllable_df[syllable_df[label_type] == -1]
    
    ## get mean location of identified labels
    
    ### Get Values
    labels = syllable_df[label_type].values
    projections = np.array(list(syllable_df[umap].values))
    
    labels_container = []
    
    ## for every identified label, find mean UMAP location
    for label in np.unique(labels):

        ## if -1 is the only label, do not skip
        if len(np.unique(labels)) == 1:
            label = -1
        ## if -1 is not the only label, skip
        else:
            if label == -1:
                continue
        ## compute mean location
        labels_container.append(np.mean(projections[labels == label], axis = 0))
        
    ## get list of label locations
    X = np.array(labels_container)
    
    ## compute 1st nearest neighbor for each unlabeled point
    nbrs = NearestNeighbors(n_neighbors = 1, algorithm = algorithm).fit(X)
    
    ## output distances and indices of nearest neighbor
    distances, indices = nbrs.kneighbors(np.array(list(unlabeled_df[umap])))
    
    ## correct unlabeled to 1st nearest label
    for index, correct_label in zip(unlabeled_df.index, indices):
        syllable_df = label_rectifier(
            syllable_df, 
            index, 
            correct_label[0],  
            verbose = verbose, 
            label_type = label_type
        )
    
    return syllable_df

### Lasso related
def sel_update(syllable_df, selector):
    '''
    Require syllable_df to be present
    Update selection_df to contain the most current lasso
    '''
    
    ## Find all the selected UMAP coordinates:
    selected_umap = selector.xys[selector.ind].data
    
    ## restructure UMAP coordinates out of the dataframe
    indv_umap = []

    for xy in np.array(syllable_df.umap):
        indv_umap.append(list(xy))

    indv_umap = np.array(indv_umap)
    
    ## produce 1D mask (which will be double the length of syllable_df)
    double_mask = np.in1d(indv_umap, selected_umap)
    
    ## reduce the mask to syllable_df dimensions
    mask = double_mask.reshape(-1, 2)[:, 1:].flatten()
    
    return syllable_df[mask]
def merge_lasso(indv_df, selector, label, reset = True):
    '''
    Take indv_df and merge labels of lasso'ed region
    '''
    selection_df = sel_update(indv_df, selector)
    
    if len(selection_df) <= 1:
        print('Nothing to Merge')
        return indv_df
    
    for i in np.unique(selection_df[label]):
        label_merger(indv_df, np.unique(selection_df[label])[0], i, label_type = label)
       
    if reset == True:
        indv_df = reserialize(indv_df, label = label)
    
    return indv_df

def split_lasso(indv_df, selector, label, stopping_cluster_size):
    '''
    Take indv_df and split labels of lasso'ed region
    '''
    
    if len(selector.ind) <= 1:
        print('Nothing to Split')
        return indv_df
    
    selection_df = sel_update(indv_df, selector)
    
    split_df = lb.recursive_umap(selection_df, cur_lab = label, new_lab = 'temp_lasso', cur_umap = 'umap', 
                                 log_scaling_boolean = False, lin_scaling_boolean = False, n_jobs = -1, 
                                 min_cluster_size_factor = 0, stopping_cluster_size = 10
                                )
    new_lab = 'temp_lasso'
    split_df = lb.clean_recursive_umap(split_df, cur_lab = label, cur_umap = 'umap', new_lab = new_lab)
    split_df[new_lab + '_id'] = split_df[new_lab + '_id'] + np.max(indv_df[label].values) + 1
    
    for i, d in zip(split_df.index, split_df[new_lab + '_id'].values):
        label_rectifier(indv_df, i, d, label_type = label, verbose = True)
    
    return reserialize(indv_df, label = label)

def reserialize(indv_df, label = 'hdbscan_labels'):
    '''
    reorder labels once they are operated on
    '''
    indv_df[label] = indv_df[label].replace(
    np.unique(indv_df[label]),np.arange(0,len(np.unique(indv_df[label]))))
    
    print('Labels reserialized')
    
    return indv_df

def fine_merger(indv_df, label, ids = [], reset = True):
    '''
    Group labels based on their id
    '''
    
    if len(ids) <= 1:
        print('Nothing to Merge')
        return indv_df
    
    for i in ids:
        label_merger(indv_df, ids[0], i, label_type = label)
    
    if reset == True:
        indv_df = reserialize(indv_df, label = label)
    
    return indv_df

def fine_splitter(indv_df, label, ids = [], stopping_cluster_size = 1, stopping_nn = 2):
    '''
    Split labels with provided ids
    '''
    
    if len(ids) <= 0:
        print('Nothing to Split')
        return indv_df
    
    selection_df = pd.DataFrame() 
    
    for i in ids:
        selection_df = selection_df.append(indv_df[indv_df[label] == i])
    
    split_df = lb.recursive_umap(selection_df, cur_lab = label, new_lab = 'temp_lasso', cur_umap = 'umap', 
                                 log_scaling_boolean = False, lin_scaling_boolean = False, n_jobs = -1, 
                                 min_cluster_size_factor = 0, stopping_cluster_size = stopping_cluster_size, n_neighbors_factor = 0,
                                 stopping_nn = stopping_nn,
                                )
    new_lab = 'temp_lasso'
    split_df = lb.clean_recursive_umap(split_df, cur_lab = label, cur_umap = 'umap', new_lab = new_lab)
    split_df[new_lab + '_id'] = split_df[new_lab + '_id'] + np.max(indv_df[label].values) + 1
    
    for i, d in zip(split_df.index, split_df[new_lab + '_id'].values):
        label_rectifier(indv_df, i, d, label_type = label, verbose = True)
    
    return reserialize(indv_df, label = label)
        
def create_lasso(indv_df, selector, label):
    '''
    Create new label for lasso'd region
    '''
    
    if len(selector.ind) <= 0:
        print('Nothing to Create')
        return indv_df
    
    selection_df = sel_update(indv_df, selector)
    
    selection_df[label] = np.max(indv_df[label].values) + 1
    
    for i, d in zip(selection_df.index, selection_df[label].values):
        label_rectifier(indv_df, i, d, label_type = label, verbose = True)
        
    return reserialize(indv_df, label = label)