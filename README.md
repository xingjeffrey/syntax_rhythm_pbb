Syntactic modulation of rhythm in Australian pied butcherbird song
==============================

This repository contains all the analysis and figures for

*Syntactic modulation of rhythm in Australian pied butcherbird song.* Jeffrey Xing, Tim Sainburg, Hollis Taylor, Timothy Q. Gentner

```@article{xing2022syntactic,
  title={Syntactic modulation of rhythm in Australian pied butcherbird song},
  author={Xing, Jeffrey and Sainburg, Tim and Taylor, Hollis and Gentner, Timothy Q},
  journal={Royal Society Open Science},
  volume={9},
  number={9},
  pages={220704},
  year={2022},
  publisher={The Royal Society}
}
```

Abstract:

> The acoustic structure of birdsong is spectrally and temporally complex. Temporal complexity is often
investigated in a syntactic framework focusing on the statistical features of symbolic song sequences.
Alternatively, temporal patterns can be investigated in a rhythmic framework that focuses on the
relative timing between song elements. Here, we investigate the merits of combining both frameworks
by integrating syntactic and rhythmic analyses of Australian pied butcherbird (Cracticus nigrogularis)
songs, which exhibit organized syntax and diverse rhythms. We show that pied butcherbird song
rhythms are categorically organized and predictable by the song’s first-order sequential syntax. These
song rhythms remain categorically distributed and strongly associated with first-order sequential
syntax after controlling for variance in note length, suggesting that the intervals between notes are
structured in a way that gives rise to a syntactically independent rhythm. We discuss the implication
of syntactic-rhythmic relations as a relevant feature of song complexity with respect to signals such
as human speech and music, and advocate for a broader conception of song complexity that takes
into account syntax, rhythm, and their interaction with other acoustic and perceptual features.

Data is available at https://zenodo.org/record/6585884


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
