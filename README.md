Syntactic modulation of rhythm in Australian pied butcherbird song
==============================

This repository contains all the analysis and figures for

*Syntactic modulation of rhythm in Australian pied butcherbird song.* Jeffrey Xing, Tim Sainburg, Hollis Taylor, Timothy Q. Gentner (In prep)

Abstract:

> The complex acoustic structure of birdsong supports many conspecific communication functions.
This complexity is often investigated in a syntactic framework, focusing on the statistical features
of symbolic song sequences. Alternatively, song complexities can be investigated in a rhythmic
framework focusing on the relative timing patterns across song units, which may complement
insights from syntactic analyses. Here, we investigate the merits of combining both frameworks by
integrating syntactic and rhythmic analyses of Australian pied butcherbird (Cracticus nigrogularis)
songs, which exhibit an organized syntax and diverse rhythms at the note level. We present methods
for investigating syntactic-rhythmic relations in birdsong, and show that pied butcherbird song
rhythms are categorically organized and predictable by the first-order sequential syntactic structure of
song. These song rhythms remain categorically distributed and strongly associated with first-order
sequential syntax after controlling for variance in note length, suggesting that the intervals between
notes are structured in a way that gives rise to a syntactically independent rhythm. We discuss the
implication of syntactic-rhythmic relations as a relevant feature of song complexity in respect to
signals such as human speech and music, and advocate for a broader conception of song complexity
that takes into account syntax, rhythm, and their interaction along with potentially other acoustic and
perceptual features.

Data is available at {DRYAD LINK}


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project
    │
    ├── notebooks          <- Jupyter notebooks for all analyses
    ├── figures            <- Generated graphics and figures used in reporting
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── butcherbird        <- Source code for use in this project.
    │   ├── data           <- Scripts for data manipulation
    │   ├── label          <- Scripts for song labeling
    │   ├── signalprocessing <- Scripts to perform signal processing
    │   ├── utils          <- Misc. scripts
    │   ├── visualization  <- Scripts to create visualizations
    │   ├── rhythm.py      <- Scripts for rhythm analysis 
    │   └── sequential.py  <- scripts for sequential modeling
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
