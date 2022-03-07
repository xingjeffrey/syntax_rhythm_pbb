from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

import numpy as np
import pandas as pd
from PIL import Image

from butcherbird.signalprocessing.create_spectrogram_note_dataset import log_resize_spec, pad_spectrogram, flatten_spectrograms
import butcherbird.utils.semisupervised as semi

import umap.umap_ as umap
from avgn.metrics.dtw_mse import build_dtw_mse
import hdbscan

def log_resize_spec(spec, scaling_factor = 10):
    resize_shape = [int(np.log(np.shape(spec)[1]) * scaling_factor), np.shape(spec)[0]]
    ## override and create lower bound (make sure no null timebins are returned)
    if (resize_shape[0] == 0):
        resize_shape[0] = 1
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape, Image.ANTIALIAS))
    return resize_spec

def linear_resize_spec(spec, scaling_factor = 10):
    resize_shape = [int(np.shape(spec)[1] * scaling_factor), np.shape(spec)[0]]
    ## override and create lower bound (make sure no null timebins are returned)
    if (resize_shape[0] == 0):
        resize_shape[0] = 1
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape, Image.ANTIALIAS))
    return resize_spec

def labeler(indv_df, 
            label = 'hdbscan_labels',
            umap_nm = 'umap',
            log_scaling_boolean = True, 
            log_scaling_factor = 10, 
            lin_scaling_boolean = True, 
            lin_scaling_factor = 1, 
            n_jobs = -1, 
            verbosity = 0, 
            min_dist = 0,
            n_neighbors_factor = 0.01, 
            stopping_nn = 50,
            min_cluster_size_factor = 0.05,
            stopping_cluster_size = 30,
            pad_mode = 'center',
            dtw = False
            ):
    '''
    Conducts resize, padding, UMAP, HDBSCAN, Neareset Neighbor on indv_df and returns indv_df
    '''
    
    ## Start Labeling
    print('\n/Initiating Labeler Unit')
    
    ## Log size spectrograms
    
    if log_scaling_boolean == True: 
        print('//Log-resize with factor', log_scaling_factor)
        with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
            syllables_spec = parallel(
            delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
            for spec in tqdm(indv_df['spectrogram'].values, desc="log-scaling spectrograms", leave=False)
        ) ## override [0, 64] to [1,64], creating lower bound
    else:
        syllables_spec = indv_df['spectrogram'].values
        
    if lin_scaling_boolean == True: 
        if lin_scaling_factor > 1:
            lin_scaling_factor = 1
        
        print('//lin-resize with factor', lin_scaling_factor)
        with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
            syllables_spec = parallel(
            delayed(linear_resize_spec)(spec, scaling_factor=lin_scaling_factor)
            for spec in tqdm(syllables_spec, desc="linear scaling spectrograms", leave=False)
        )
        
    ## Pad spectrograms
    pad_length = np.max([np.shape(i)[1] for i in syllables_spec])
    print('//Pad Spectrograms to', pad_length, 'mode:', pad_mode)
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllables_spec = parallel(
        delayed(pad_spectrogram)(spec, pad_length, mode = pad_mode)
        for spec in tqdm(
            syllables_spec, desc="padding spectrograms", leave=False
        )
    )
    
    ## Run UMAP 
    ### Reduce range to 0 ~ 1
    syllables_spec = np.array([i/np.max(i) for i in syllables_spec])

    ### flatten spectrogram
    specs_flattened = flatten_spectrograms(syllables_spec)

    
    nn = int(len(syllables_spec) * n_neighbors_factor)
    
    if nn < stopping_nn:
        nn = stopping_nn
    
    print('Nearest Neighbor Count:', nn)
    
    ##
    print('/Projecting to UMAP...')
    
    if dtw == True:
        print('Use dynamic time warping')
        syllables_spec_T = np.array([i.T for i in syllables_spec])
        ## add dynamic time warping
        dtw_metric = build_dtw_mse(syllables_spec_T[0].shape)
    
        ### Create UMAP space
        fit = umap.UMAP(
            metric = dtw_metric,
            min_dist = min_dist, 
            n_neighbors = nn, 
            verbose = verbosity,
            n_jobs = 36
        )
        
    else:
        ### Create UMAP space
        print('using regular')
        fit = umap.UMAP(
            #metric = dtw_metric,
            min_dist = min_dist, 
            n_neighbors = nn, 
            verbose = verbosity,
            n_jobs = 36
        )

    ### Project Data into UMAP and retrieve coordinates
    z = list(fit.fit_transform(specs_flattened))
    
    ### Write into indv_df
    indv_df[umap_nm] = z
    
    ### get all the UMAP coordinates
    z = np.vstack(indv_df[umap_nm].values)

    mcs = int(len(z) * min_cluster_size_factor)
    
    if mcs <= stopping_cluster_size:
        mcs = stopping_cluster_size
    
    print('/Clustering with min_cluster_size =', mcs)
    
    ### make HDBSCAN clusters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = mcs
    ).fit(z)

    ## append to indv_df
    indv_df.loc[:, label] = list(clusterer.labels_)
    
    ## check if there are unlabeled data
    if sum(indv_df[label] == -1) > 0:
        # scurry unlabeled data to nearest cluster
        indv_df = semi.cluster_unlabeled(indv_df, label_type = label, verbose = False)
        
    cluster_nb = len(np.unique(indv_df[label]))
    
    if cluster_nb <= 1:
        discoverable = False
        print('\nDiscovered', cluster_nb, 'Cluster, No more subclusters are hidden\n\n')
    else:
        discoverable = True
        print('\nDiscovered', cluster_nb, 'Clusters, potential subclusters discoverable in deeper layers\n\n')
    
    return indv_df, discoverable

def recursive_umap(
    indv_df, 
    cur_lab = 'hdbscan_labels', 
    new_lab = 'recur_labels', 
    cur_umap = 'umap', 
    layer = 1, 
    log_scaling_boolean = True,
    lin_scaling_boolean = False, 
    log_scaling_factor = 5,      
    lin_scaling_factor = 1,
    n_jobs = -1,                                
    verbosity = 0,                                
    min_dist = 0,                           
    n_neighbors_factor = 0.01,                            
    stopping_nn = 50,                           
    min_cluster_size_factor = 0.15,                           
    stopping_cluster_size = 50,
    pad_mode = 'center'
):
    '''
    Construct Recursive UMAP for Labeling
    '''

    ## Print Initiation Messages
    print('Start Recursive Cycle...\n')
    print('Layer', layer)
    
    ## add one to layer
    layer = layer + 1
    
    next_lab = cur_lab + '1'
    next_umap = cur_umap + '1'
    
    ## set up container for this layer
    container = []
    
    for label in tqdm(np.unique(indv_df[cur_lab])):
        print('[ Label', label, '], layer = ', layer-1)
        ## discover first layer
        label_df, discoverable = labeler(
            indv_df[indv_df[cur_lab] == label],    
            label = next_lab, ##adds new column for recursively spawned labels                               
            umap_nm = next_umap, ## adds new column for recursively spawn UMAP                               
            log_scaling_boolean = log_scaling_boolean,                                
            log_scaling_factor = log_scaling_factor,     
            lin_scaling_boolean = lin_scaling_boolean,                                
            lin_scaling_factor = lin_scaling_factor,  
            n_jobs = n_jobs,                                
            verbosity = verbosity,                                
            min_dist = min_dist,                           
            n_neighbors_factor = n_neighbors_factor,                            
            stopping_nn = stopping_nn,                           
            min_cluster_size_factor = min_cluster_size_factor,                           
            stopping_cluster_size = stopping_cluster_size,
            pad_mode = pad_mode
        )
        
        if discoverable == True:
            sub_clusters = recursive_umap(
                label_df, 
                cur_lab = next_lab, 
                cur_umap = next_umap, 
                layer = layer,                                                   
                log_scaling_boolean = log_scaling_boolean,                                            
                log_scaling_factor = log_scaling_factor, 
                lin_scaling_boolean = lin_scaling_boolean,                                
                lin_scaling_factor = lin_scaling_factor*2,
                n_jobs = n_jobs,                                            
                verbosity = verbosity,                                            
                min_dist = min_dist,                                      
                n_neighbors_factor = n_neighbors_factor,                                        
                stopping_nn = stopping_nn,                                       
                min_cluster_size_factor = min_cluster_size_factor,                           
                stopping_cluster_size = stopping_cluster_size,
                pad_mode = pad_mode
            )
        else:
            sub_clusters = label_df
            
        container.append(sub_clusters)
    
    indv_df = pd.concat(container)
    
    return indv_df

def clean_recursive_umap(
    indv_df,
    cur_lab = 'hdbscan_labels',
    cur_umap = 'umap',
    new_lab = 'recur_labels'
):
    ## clean up interim columns and add cleaned up results
    
    ## find max numbers of layers present
    layer_nb = sum(np.char.find(list(indv_df.columns), cur_lab, start = 0) + 1)
    

    ## Duplicate hdbscan_labels first
    indv_df[new_lab] = indv_df[cur_lab].astype(str)

    ## now that it's duplicated, append stuff into iter_labels
    for layer in np.arange(1, layer_nb):
        suffix = ''
        for i in np.arange(0,layer):
            suffix = suffix + '1'
        layer_label = cur_lab + suffix
        indv_df[new_lab] = indv_df[new_lab] + '|' + indv_df[layer_label].astype(str)
        
    new_lab_id = new_lab + '_id'
    # Parse into a simpler version too
    indv_df[new_lab_id] = indv_df[new_lab]
    indv_df[new_lab_id] = indv_df[new_lab_id].replace(
        np.unique(indv_df[new_lab]),np.arange(0,len(np.unique(indv_df[new_lab]))))
    
    ## sort_dataframe to get phrases in order
    indv_df = indv_df.sort_index()
    
    ## Drop excess columns
    hdbscan_mask = list((np.char.find(list(indv_df.columns), cur_lab + '1', start = 0)))
    umap_mask = list((np.char.find(list(indv_df.columns), cur_umap + '1', start = 0)))
    
    container = []
    for col, mask, mask2 in zip(indv_df.columns, hdbscan_mask, umap_mask):
        if mask == 0 or mask2 == 0:
            container.append(col)
   
    indv_df = indv_df.drop(container, axis = 1)
    
    return indv_df