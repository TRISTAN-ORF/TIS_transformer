<div align="center">
<h1>🧬 TIS transformer</h1>

Driving coding sequence discovery since 2023

[![DOI:10.1007/978-3-319-76207-4_15](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1093/nargab/lqad021)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8338189.svg)](https://doi.org/10.5281/zenodo.8338189)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://jdcla.ugent.be)
</div>

## 📋 About <a name="about"></a>
TIS transformer is created to annotate translation initiation sites (TISs) on transcripts using nucleotide sequence information. The repository holds the scripts, data, models and model outputs used to perform the benchmarks and remap the human proteome as discussed in [NAR Genomics & Bioinformatics](https://academic.oup.com/nargab/article/5/1/lqad021/7069281).

To apply the TIS Transformer for new transcript sequences of the human transcriptome, check out [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8550/TIS_Transformer_Tool) to skip local installation. For training models on new organisms or applying larger data, check out the installation guide.

Data files too large to host on GitHub that were created as part of the study, specifically those in the `data/`, `models/`, and `outputs/` folders, can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.8338189).

## 🔗 Methodology <a name="methodology"></a>
Annotations are performed by a machine learning model following a methodology similar to those defined for natural language modelling tasks. Ensemble annotations have been used to obtain transcripts and TIS annotations. The model processes the full transcript sequence to predict the presence of TISs at each position on the transcript. The model architecture is based on that of the [Performer](https://arxiv.org/abs/2009.14794), which allows the use of longer input sequences due to the memory efficiency of the attention-based calculations.

## 📏 Benchmark <a name="benchmark"></a>

The tool has been compared to similar approaches applying TIS prediction based on the transcript nucleotide sequence. More details about the benchmarking approach are listed in the [article](https://doi.org/10.1093/nargab/lqad021). The scripts to obtain the scores for TISRover, TITER, and DeepGSR are deposited in `scripts/benchmarks`. The models are found under `models/benchmarks`

## 📊 Remapping of the human proteome <a name="human"></a>

Using this method, the proteome of the complete human genome has been remapped by training multiple models. Annotations are performed on chromosomes excluded from the training/validation process. The scripts used to train the relevant models, and the models themselves, are stored under `/scripts/proteome` and `/models/proteome`

Model predictions are stored under `outputs/`. For each chromosome, an annotated set of the top `3*(k)` predictions have been curated, where `(k)` denotes the number of translation initiation sites featured by the Ensembl. More information about the column data is given in `outputs/README.md`.

### 🔍 Browse remapped proteome
The annotations performed by the model can be browsed through our custom app. It is furthermore possible to filter the results based on a variety of TIS and transcript properties. 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jdcla.ugent.be/TIS_Transformer_Browser)

Alternatively, the model outputs for each position on the transcriptome can be acquired under `/outputs/`


## 📖 User guide <a name="userguide"></a>

For smaller data sets, check out [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8550/TIS_Transformer_Tool) to apply TIS Transformer without installation.


> [!CAUTION]
> RiboTIE and TIS transformer are two tools that evolved from a central package `transcript-transformer`, but were handled in different repositories due to various reasons. With the release of v1.0.0, both tools are now again included and documented by a single toolset called TRISTAN. For an up-to-date documentation and the most recent updates, or for submitting issues, make sure to refer to [the TRISTAN repository](https://github.com/TRISTAN-ORF/TRISTAN)!

## 🖊️ Citation <a name="citation"></a>
       
```bibtex
@article {10.1093/nargab/lqad021,
    author = {Clauwaert, Jim and McVey, Zahra and Gupta, Ramneek and Menschaert, Gerben},
    title = "{TIS Transformer: remapping the human proteome using deep learning}",
    journal = {NAR Genomics and Bioinformatics},
    volume = {5},
    number = {1},
    year = {2023},
    month = {03},
    issn = {2631-9268},
    doi = {10.1093/nargab/lqad021},
    url = {https://doi.org/10.1093/nargab/lqad021},
    note = {lqad021},
    eprint = {https://academic.oup.com/nargab/article-pdf/5/1/lqad021/49418780/lqad021\_supplemental\_file.pdf},
}
```
