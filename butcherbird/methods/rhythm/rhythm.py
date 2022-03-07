import pandas as pd
import numpy as np
import scipy.stats

## This takes the raw onsets and syntactic identities and parses its rhythm

def construct_rhythm_df(onsets, syntactic_units, expected_interval_range):
    '''
    This function transforms a list of onsets and list of syntactic units for rhythm analyses
    '''
    
    ### DISCOVER ALL INTER-ONSET INTERVALS
    ## Use counter design
    i = 0
    intervals = []
    
    ## for every note onset
    for onset in onsets:
        
        ## if at last note, exit for loop
        if i == (len(onsets) - 1):
            continue
            
        ## interval = next onset - current onset, add to interval list
        interval = onsets[i + 1] - onsets[i]
        intervals.append(interval)
        
        ## counter up
        i = i + 1
        
    ## Collect intervals into dyadic formation
    
    ## interval 1 does not include last interval
    intervals1 = list(intervals)
    del(intervals1[-1])
    
    ## interval 2 does not include first interval
    intervals2 = list(intervals)
    del(intervals2[0])
    
    ## Put intervals into datafrmae
    d = {'intervals1': intervals1, 'intervals2': intervals2}
    dyadic = pd.DataFrame(data = d)
    
    ## back propagate label and spec information
    
    ## note identity 1 // Delete last two values
    labels1 = list(syntactic_units)
    del(labels1[-1])
    del(labels1[-1])
    
    ## note identity 2 // Delete first and last value
    labels2 = list(syntactic_units)
    del(labels2[0])
    del(labels2[-1])
    
    ## note identity 3 // Delete first two values
    labels3 = list(syntactic_units)
    del(labels3[0])
    del(labels3[0])
    
    ## fill
    dyadic['label1'] = labels1
    dyadic['label2'] = labels2
    dyadic['label3'] = labels3
    
    ## Sort dyadic into short/long interval
    ### sort all intervals and calculate necessary components
    s_interval = []
    l_interval = []
    cycle_dur = []
    ratio_custom = []
    ratio_roeske = []
    
    ## for every dyadic
    for index, row in dyadic.iterrows():

        i1 = row['intervals1']
        i2 = row['intervals2']

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

    ## push into dyadic
    dyadic['s_interval'] = s_interval
    dyadic['l_interval'] = l_interval
    dyadic['cycle_dur'] = cycle_dur
    dyadic['ratio_roeske'] = ratio_roeske
    dyadic['ratio_custom'] = ratio_custom
    
    ## sort ascending by shortest cycle to longest cycle
    dyadic = dyadic.sort_values(by = ['cycle_dur'])
    
    ## filter out transition between phrases
    dyadic = dyadic[dyadic['l_interval'] < expected_interval_range[1]]
    dyadic = dyadic[dyadic['s_interval'] > expected_interval_range[0]]
    
    ## put cycle rank into data frame
    dyadic['cycle_rank'] = dyadic['cycle_dur'].rank().astype(int)
    
    return dyadic

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
## from https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/    
    
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
    #print(rand_X)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print (ujd, wjd)
        H = 0
 
    return H

## Plot Rhythm Decomposition Charts
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm

def decompose_rhythm(
    ## Macro controls
    dyadic, 
    label = None, 
    position = 1, 
    hopkins_iteration = 1,
    dataset_description = "Songbird/Language/Music",
    ## Graph controls
    figsize = (16, 32),
    xlim = (0, 1),
    ylim = (0, 250),
    s_interval = 's_interval',
    l_interval = 'l_interval', 
    cycle_rank = 'cycle_rank',
    marker = '.',
    color = 'black', 
    hue_norm = (0, 1), 
    palette = 'deep',
    legend = False, 
    linewidth = 0,
    s = 20,
    s_ratio = 0.1,
    alpha = 1,
    alpha_ratio = 0.1,
    rhythm_ratio = 'ratio_roeske',
    binwidth = 0.01,
    kde = True
):
    '''
    Produces a series of analyses that attempts to decompose the rhythms parsed in pied butcherbird songs by labels
    '''
    
    '''PARSE LABEL LOCATION'''
    
    ## Parse label position
    label_position = 'label' + str(position)
    
    '''FILTER LABELS OF INTEREST'''
    if label == None:
        ## skip filter
        dyadic_filtered = dyadic
    else:
        ## Get filtered dataframe
        dyadic_filtered = dyadic[dyadic[label_position] == label]

    '''GENERATE CANVAS'''    
        
    ## Generate subplot that 5 rows * 2 cols, with large enough canvas
    FIG_ROWS = 5
    FIG_COLS = 2
    fig, axes= plt.subplots(FIG_ROWS, FIG_COLS, figsize = figsize)
    title = 'Rhythm Decomposition of '+ dataset_description + ' by Label: ' + str(label) + ' in position ' + str(position)
    fig.suptitle(title)

    '''FIRST ROW: RHYTHMS BY LABEL'''

    ## LEFT COLUMN: Short Interval
    
    ### Restrict Graph Display Range
    axes[0][0].set_xlim(xlim[1], xlim[0])
    axes[0][0].set_xlabel('Short Interval Length [second]')
    axes[0][0].set_ylabel('Rhythm Speed (Fast to Slow)')
    
    ## Plot rhythm rastor for short interval (MASK)
    sns.scatterplot(
        ax = axes[0][0], ## Place on ROW 0, COL 0
        data = dyadic, 
        x = s_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = label_position,
        palette = palette,
        legend = legend,
        linewidth = linewidth,
        alpha = alpha * alpha_ratio,
        s = s * s_ratio
    ).set_title(
        "Short Intervals Rastor Colored by Label"
    )
    
    ## Plot rhythm rastor for short interval
    sns.scatterplot(
        ax = axes[0][0], ## Place on ROW 0, COL 0
        data = dyadic_filtered, 
        x = s_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = label_position,
        palette = palette,
        linewidth = linewidth,
        alpha = alpha,
        s = s,
        legend = False
    ).set_title(
        "Short Intervals Rastor Colored by Label"
    )

    ## RIGHT COLUMN: Long Interval
    
    ### Restrict Graph Display Range
    axes[0][1].set_xlim(xlim[0], xlim[1])
    axes[0][1].set_xlabel('Long Interval Length [second]')
    axes[0][1].set_ylabel('Rhythm Speed (Fast to Slow)')
    
    ## Plot rhythm rastor for long interval (MASK)
    sns.scatterplot(
        ax = axes[0][1], ## Place on ROW 0, COL 1
        data = dyadic, 
        x = l_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = label_position,
        palette = palette,
        legend = legend,
        linewidth = linewidth,
        alpha = alpha * alpha_ratio,
        s = s * s_ratio
    ).set_title(
        "Long Intervals Rastor Colored by Label"
    )
    
    ## Plot rhythm rastor for long interval
    sns.scatterplot(
        ax = axes[0][1], ## Place on ROW 0, COL 1
        data = dyadic_filtered, 
        x = l_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = label_position,
        palette = palette,
        linewidth = linewidth,
        alpha = alpha, 
        s = s,
        legend = False
    ).set_title(
        "Long Intervals Rastor Colored by Label"
    )

    '''SECOND ROW: RHYTHMS BY ROESKE RATIO'''
    
    ## LEFT COLUMN: Short Interval
    
    ### Restrict Graph Display Range
    axes[1][0].set_xlim(xlim[1], xlim[0])
    axes[1][0].set_xlabel('Short Interval Length [second]')
    axes[1][0].set_ylabel('Rhythm Speed (Fast to Slow)')
    
    ## Use diverging palette for illustration
    cmap = sns.diverging_palette(250, 30, l=70, center="dark", as_cmap=True)
    
    ## Plot rhythm rastor for short interval (MASK)
    sns.scatterplot(
        ax = axes[1][0], ## Place on ROW 1, COL 0
        data = dyadic, 
        x = s_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        palette = cmap,
        linewidth = linewidth, 
        alpha = alpha * alpha_ratio,
        s = s * s_ratio
    ).set_title(
        "Short Intervals Rastor Colored by Rhythm Ratio"
    )
    
    ## Plot rhythm rastor for short interval
    sns.scatterplot(
        ax = axes[1][0], ## Place on ROW 1, COL 0
        data = dyadic_filtered, 
        x = s_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        palette = cmap,
        linewidth = linewidth, 
        alpha = alpha,
        s = s, 
        legend = False
    ).set_title(
        "Short Intervals Rastor Colored by Rhythm Ratio"
    )

    ## RIGHT COLUMN: Long Interval
    
    ### Restrict Graph Display Range 
    axes[1][1].set_xlim(xlim[0], xlim[1])
    axes[1][1].set_xlabel('Long Interval Length [second]')
    axes[1][1].set_ylabel('Rhythm Speed (Fast to Slow)')

    ## Plot rhythm rastor for long interval (MASK)
    sns.scatterplot(
        ax = axes[1][1], ## Place on ROW 1, COL 1
        data = dyadic, 
        x = l_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        palette = cmap,
        linewidth = linewidth, 
        alpha = alpha * alpha_ratio,
        s = s * s_ratio
    ).set_title(
        "Long Intervals Rastor Colored by Rhythm Ratio"
    )
    
    ## Plot rhythm rastor for long interval
    sns.scatterplot(
        ax = axes[1][1], ## Place on ROW 1, COL 1
        data = dyadic_filtered, 
        x = l_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        palette = cmap,
        linewidth = linewidth, 
        alpha = alpha,
        s = s,
        legend = False
    ).set_title(
        "Long Intervals Rastor Colored by Rhythm Ratio"
    )

    '''THIRD ROW: DECOMPOSE LONG-SHORT RHYTHMS'''
    
    ## Limit graph display range
    axes[2][0].set_xlim(xlim[1], xlim[0])
    axes[2][0].set_xlabel('Short Interval Length [second]')
    axes[2][0].set_ylabel('Rhythm Speed (Fast to Slow)')
    
    '''    ## Use upper half of diverging palette
    cmap = sns.dark_palette("#ff6347", as_cmap=True)'''
    
    ## Plot rhythm rastor for short interval (MASK)
    sns.scatterplot(
        ax = axes[2][0],
        data = dyadic[dyadic['ratio_roeske'] >= 0.5], 
        x = s_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        palette = cmap,
        linewidth = linewidth, 
        alpha = alpha * alpha_ratio,
        s = s * s_ratio
    ).set_title(
        "Short Intervals Rastor for LONG-SHORT Rhythms Colored by Rhythm Ratio"
    )
    
    ## Plot rhythm rastor for short interval
    sns.scatterplot(
        ax = axes[2][0],
        data = dyadic_filtered[dyadic_filtered['ratio_roeske'] >= 0.5], 
        x = s_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        palette = cmap,
        linewidth = linewidth, 
        alpha = alpha,
        s = s,
        legend = False
    ).set_title(
        "Short Intervals Rastor for LONG-SHORT Rhythms Colored by Rhythm Ratio"
    )

    ## Limit graph display range
    axes[2][1].set_xlim(xlim[0], xlim[1])
    axes[2][1].set_xlabel('Long Interval Length [second]')
    axes[2][1].set_ylabel('Rhythm Speed (Fast to Slow)')
    
    ## Plot rhythm rastor for long intervals (mask)
    sns.scatterplot(
        ax = axes[2][1],
        data = dyadic[dyadic['ratio_roeske'] >= 0.5], 
        x = l_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        palette = cmap,
        linewidth = linewidth, 
        alpha = alpha * alpha_ratio,
        s = s * s_ratio
    ).set_title(
        "Long Intervals Rastor for LONG-SHORT Rhythms Colored by Rhythm Ratio"
    )
    
    ## Plot rhythm rastor for long intervals
    sns.scatterplot(
        ax = axes[2][1],
        data = dyadic_filtered[dyadic_filtered['ratio_roeske'] >= 0.5], 
        x = l_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        palette = cmap,
        linewidth = linewidth, 
        alpha = alpha,
        s = s,
        legend = False
    ).set_title(
        "Long Intervals Rastor for LONG-SHORT Rhythms Colored by Rhythm Ratio"
    )

    '''FOURTH ROW: DECOMPOSE SHORT-LONG RHYTHMS'''
    
    ## LEFT COLUMN
    
    ## Limit graph display range
    axes[3][0].set_xlim(xlim[1], xlim[0])
    axes[3][0].set_xlabel('Short Interval Length [second]')
    axes[3][0].set_ylabel('Rhythm Speed (Fast to Slow)')
    
    '''## Use lower half of diverging palette
    cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)'''
    
    ## Plot rhythm rastor for short intervals (MASK)
    sns.scatterplot(
        ax = axes[3][0],
        data = dyadic[dyadic[rhythm_ratio] <= 0.5], 
        x = s_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        linewidth = linewidth, 
        palette = cmap,
        alpha = alpha * alpha_ratio,
        s = s * s_ratio
    ).set_title(
        "Short Intervals Rastor for SHORT-LONG Rhythms Colored by Rhythm Ratio"
    )
    
    ## Plot rhythm rastor for short intervals
    sns.scatterplot(
        ax = axes[3][0],
        data = dyadic_filtered[dyadic_filtered[rhythm_ratio] <= 0.5], 
        x = s_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        linewidth = linewidth, 
        palette = cmap,
        alpha = alpha,
        s = s,
        legend = False
    ).set_title(
        "Short Intervals Rastor for SHORT-LONG Rhythms Colored by Rhythm Ratio"
    )

    ## RIGHT COLUMN
    
    ## Limit graph display range
    axes[3][1].set_xlim(xlim[0], xlim[1])
    axes[3][1].set_xlabel('Long Interval Length [second]')
    axes[3][1].set_ylabel('Rhythm Speed (Fast to Slow)')
    
    ## Plot rhythm rastor for long intervals
    sns.scatterplot(
        ax = axes[3][1],
        data = dyadic[dyadic[rhythm_ratio] <= 0.5], 
        x = l_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        linewidth = linewidth, 
        alpha = alpha * alpha_ratio,
        palette = cmap,
        s = s * s_ratio
    ).set_title(
        "Long Intervals Rastor for SHORT-LONG Rhythms Colored by Rhythm Ratio"
    )
    
    ## Plot rhythm rastor for long intervals
    sns.scatterplot(
        ax = axes[3][1],
        data = dyadic_filtered[dyadic_filtered['ratio_roeske'] <= 0.5], 
        x = l_interval, 
        y = cycle_rank, 
        marker = marker, 
        color = color, 
        hue = rhythm_ratio,
        hue_norm = hue_norm,
        linewidth = linewidth, 
        alpha = alpha,
        palette = cmap,
        s = s,
        legend = False
    ).set_title(
        "Long Intervals Rastor for SHORT-LONG Rhythms Colored by Rhythm Ratio"
    )

    '''FIFTH ROW: RHYTHM RATIO AND HOPKINS STATISTIC HISTOGRAM'''
    
    ## Limit Display Range of Ratio
    axes[4][0].set_xlim(xlim[0], xlim[1])
    axes[4][0].set_xlabel('Rhythm Ratio')
    
    ## Generate Histogram
    sns.histplot(
        ax = axes[4][0],
        data = dyadic_filtered, 
        x = rhythm_ratio, 
        binwidth = binwidth,
        ##hue= rhythm_ratio,
        palette = cmap,
        kde = kde, 
        color = 'deepskyblue'
    ).set_title(
        'Rhythm Ratio Histogram: i_1/(i_1 + i_2)'
    )

    ## Give Simple Integer Ratio Lines
    axes[4][0].axvline(1/4, color = 'red')
    axes[4][0].axvline(1/3, color = 'red')
    axes[4][0].axvline(1/2, color = 'red')
    axes[4][0].axvline(2/3, color = 'red')
    axes[4][0].axvline(3/4, color = 'red')

    axes[4][0].set_xlim(0, 1)
    
    ## Get Sampling Distribution of Hopkins Statistics
    SDoH = []
    for i in tqdm(range(0, hopkins_iteration)):
        SDoH.append(hopkins(dyadic_filtered[[rhythm_ratio]]))

    ## Limit Display Range of Hopkins Statistics
    axes[4][1].set_xlim(xlim[0], xlim[1])
    axes[4][1].set_xlabel('Hopkins Statistic; 0 = Uniform, 0.5 = Random, 1 = Cluster')
    
    ## Generate Histogram
    sns.histplot(
        ax = axes[4][1],
        data = SDoH, 
        binwidth = binwidth, 
        kde = kde, 
        color = 'red'
    ).set_title(
        "Sampling Distribution of Hopkins Statistics"
    )

    ## Handle duplicate legends
    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    legend_without_duplicate_labels(axes[1][0])
    legend_without_duplicate_labels(axes[1][1])
    legend_without_duplicate_labels(axes[2][0])
    legend_without_duplicate_labels(axes[2][1])
    legend_without_duplicate_labels(axes[3][0])
    legend_without_duplicate_labels(axes[3][1])
    legend_without_duplicate_labels(axes[4][0])
    legend_without_duplicate_labels(axes[4][1])
    
    plt.show()
    return SDoH

def rhythm_std(dyadic):    
    position_list = []
    label_list = []
    cycle_std = []
    ratio_std = []

    posits = ['label1', 'label2', 'label3']

    ## For each syntactic unit in one position
    for position in tqdm(posits):

        ## find rhythm clusters for every unique symbol in position 1
        for label in tqdm(np.unique(dyadic[position].astype(str).values)):

            ## get dataframe
            sub_dyadic = dyadic[dyadic[position] == label]
            
            ## if dataframe doesn't have entries continue
            if len(sub_dyadic) == 0:
                continue

            ## Get data
            cycle_dur = dyadic[dyadic[position] == label]['cycle_dur'].values
            ratio = dyadic[dyadic[position] == label]['ratio_roeske'].values

            ## constrict cycle_rank from 0 to 1
            cycle_rank_standardized = cycle_dur / np.max(cycle_dur)

            ## append lists
            position_list.append(position)
            label_list.append(label)
            cycle_std.append(np.std(cycle_rank_standardized))
            ratio_std.append(np.std(ratio))

    range_analysis = pd.DataFrame(
        {'position':position_list, 'label':label_list, 'cycle_std': cycle_std, 'ratio_std': ratio_std}
    )
    
    return range_analysis

def decompose_flexibility (
    rhythm_flex,
    figsize = (16, 6),
    xlim = [0, 1],
    ylim = [0, 0.5],
    dataset_description = 'Birdsong/Language/Music',
    bypass_zero = False
):
    '''
    Produces a series of analyses that attempts to decompose rhythm diffusion in corpus
    '''
    
    if (bypass_zero == True):
        rhythm_flex = rhythm_flex[rhythm_flex['ratio_std'] != 0]
    
    '''GENERATE CANVAS'''    
        
    ## Generate subplot that 5 rows * 2 cols, with large enough canvas
    
    FIG_ROWS = 1
    FIG_COLS = 2
    fig, axes = plt.subplots(FIG_ROWS, FIG_COLS, figsize = figsize)
    title = 'Flexibility Decomposition of ' + dataset_description
    fig.suptitle(title)
    
    ## First Col: Temporal Flexibility
    axes[0].set_xlim(xlim[0], xlim[1])
    axes[0].set_ylim(ylim[0], ylim[1])
    axes[0].set_xlabel('Standard Deviation')
    axes[0].set_ylabel('Proportion')
    
    cycle_mean = np.mean(rhythm_flex['cycle_std'])
    
    sns.histplot(
        data = rhythm_flex['cycle_std'], 
        ax = axes[0], 
        stat = 'probability', 
        binwidth = 0.025,
        common_norm=False
    ).set_title('STD of Rhythm LENGTH Per Syntactic Unit, Normalized by Max Length')
    axes[0].axvline(cycle_mean, color='red', linestyle='--')
    axes[0].text(cycle_mean + 0.05, 0.50, "M = " + str(np.round((cycle_mean),3)), color = 'red', ha="left", va="top", transform=axes[0].transAxes)
    
    ## Second Col: Temporal Flexibility
    axes[1].set_xlim(xlim[0], xlim[1])
    axes[1].set_ylim(ylim[0], ylim[1])
    axes[1].set_xlabel('Standard Deviation')
    axes[1].set_ylabel('Proportion')
    
    ratio_mean = np.mean(rhythm_flex['ratio_std'])
    
    sns.histplot(
        data = rhythm_flex['ratio_std'], 
        ax = axes[1], 
        stat = 'probability', 
        binwidth = 0.025,
        common_norm=False,
        color = 'green',
    ).set_title('STD of Rhythm RATIO Per Syntactic Unit, Normalized by Max Length')
    axes[1].axvline(ratio_mean, color='red', linestyle='--')
    axes[1].text(ratio_mean + 0.05, 0.50, "M = " + str(np.round((ratio_mean),3)), color = 'red', ha="left", va="top", transform=axes[1].transAxes)

    plt.show()
    return np.mean(rhythm_flex['cycle_std']), np.mean(rhythm_flex['ratio_std'])
    