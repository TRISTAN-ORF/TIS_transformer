# TIS transformer [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://jdcla.ugent.be)
Supporting repository to equally named study
## About <a name="about"></a>
The TIS transformer is created to annotate translation initiation sites on transcripts based on its nucleotide sequences. The repository holds the scripts, data, models and model outputs used to perform the benchmarks and remap the human proteom discussed in [this paper](https://www.biorxiv.org/content/10.1101/2021.11.18.468957v1).

To apply the models on custom data or train new models following our approach, we refer to the documentation provided alongside the [`transcript_transformer`](www.github.com/jdcla/transcript_transformer) python package, written as part of this research.

Data files to large to host on GitHub, specifically those in the `data/` and `outputs/` folders, can be downloaded from [here](www.biobix.be/tis_transformer).
## Methodology <a name="methodology"></a>
Annotations are performed by a machine learning model following a methodology similar to those defined for natural language modelling tasks. Ensemble annotations have been used to obtain transcripts and TIS annotations. The model processes the full transcript sequence to impute the presence of TIS at each position on the transcript. 

The model architecture is based on that of the [Performer](https://arxiv.org/abs/2009.14794), which allows the use of longer input sequences due to the memory efficiency of the attention-based calculations.

## Benchmark <a name="benchmark"></a>

The tool has been compared to similar approaches applying TIS imputation based on the transcript nucleotide sequence. More details about the benchmarking approach are listed in the [article](https://www.biorxiv.org/content/10.1101/2021.11.18.468957v1). The scripts to obtain the scores for TISRover, TITER, and DeepGSR are deposited in `scripts/benchmarks`. The models are listed under `models/benchmarks`

## Remapping of the human proteome <a name="human"></a>

Using this method, the proteome of the complete human genome has been remapped by training multiple models. Annotations are performed on chromosomes excluded from the training/validation process. The scripts used to train the relevant models, and the models themselves, are found in `/scripts/proteome` and `/models/proteome`

For each chromosome, an annotated set of the top `3*(k)` predictions are given, where `(k)` denotes the number of translation initiation sites featured by the Ensembl. More information about the column data is given in `outputs/README.md`.

### Browse remapped proteome
The annotations performed by the model can be browsed through our custom app: 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://jdcla.ugent.be)

Alternatively, all results can be acquired under `/outputs/`


## User guide <a name="userguide"></a>

To apply the methodology for custom data or perform predictions, see the [transcript transformer](https://github.com/jdcla/transcript_transformer) package.


## Citation <a name="citation"></a>
       
```
@article {Clauwaert2021.11.18.468957,
	author = {Clauwaert, Jim and McVey, Zahra and Gupta, Ramneek and Menschaert, Gerben},
	title = {TIS Transformer: Re-annotation of the human proteome using deep learning},
	elocation-id = {2021.11.18.468957},
	year = {2021},
	doi = {10.1101/2021.11.18.468957},
	URL = {https://www.biorxiv.org/content/early/2021/11/19/2021.11.18.468957},
	eprint = {https://www.biorxiv.org/content/early/2021/11/19/2021.11.18.468957.full.pdf},
	journal = {bioRxiv}
}
```
