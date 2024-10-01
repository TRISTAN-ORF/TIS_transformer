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

### Installation 

`pytorch` needs to be separately [installed by the user](https://pytorch.org/get-started/locally/). GPU/CUDA support is not required.

Next, the package can be installed running 
```bash
pip install transcript-transformer
```
### Training a model from scratch

Dictionary files (YAML/JSON) are the recommended approach to pass arguments to the tool. A genome-level reference and assembly file (`*.gtf`, `*.fa`) and output path for the created database file (`*.h5`) are three required arguments. 

For example, using `config.yaml`:

```yaml
gtf_path : Homo_sapiens.GRCh38.110.gtf
fa_path : Homo_sapiens.GRCh38.dna.primary_assembly.fa
h5_path : Homo_sapiens.GRCh38.110.h5
```
we can run TIS Transformer:

```bash
tis_transformer config.yaml
```
The tool offers various options to alter the configuration of the model, training parameters and output files. Evaluate these by running `tis_transformer --help`.

In essence, running `tis_transformer` will go through the following steps:

1. Parse all data to a HDF5 database (h5_path)
2. Fine-tune 5 models on non-overlapping folds of the data.
3. Get model predictions for all positions of the transcriptome
4. Generate table with metadata for the top ranking predictions (determined by a set cut-off)

Various options are available, such as increasing the set of positive predictions provided by TIS Transformer. This can be done by lowering the cut-off with which the positive set is derived:

```bash
tis_transformer config.yaml --prob_cutoff 0.01 --results
```

**Note that `--results` will prevent the the tool from retraining new models from scratch by skipping steps 1--3 by using the already existing set of predictions of the transcriptome to recreate the final table!**

See also [https://github.com/TRISTAN-ORF/RiboTIE](https://github.com/TRISTAN-ORF/RiboTIE).

### Predict single sequences on the published model

In addition to predicting the full transcriptome as featured in the previous step, it is also possible to use single RNA sequences as input or a path to a `.fa` file on a trained model. The predict function returns probabilities for all nucleotide positions on the transcript and is stored as a numpy vector format (`.npy`). When high scoring sites are present, a `.csv` file containing the relevant positions and additional information is generated. 

Six models were trained using different sets of chromosomes. When applying transcript sequences derived from existing genes, ensure the use of the correct model. The model to be used is determined by the chromosome the gene/transcript is located on:

| Model                                    | Chromosomes   |
|------------------------------------------|---------------|
| models/proteome/TIS_transformer_L_1.ckpt | 1, 7, 13, 19  |
| models/proteome/TIS_transformer_L_2.ckpt | 2, 8, 14, 20  |
| models/proteome/TIS_transformer_L_3.ckpt | 3, 9, 15, 21  |
| models/proteome/TIS_transformer_L_4.ckpt | 4, 10, 16, 22 |
| models/proteome/TIS_transformer_L_5.ckpt | 5, 11, 17, X  |
| models/proteome/TIS_transformer_L_6.ckpt | 6, 12, 18, Y  |

This step ensures the model used is not one trained on related data. For other types of transcript sequences, any model is valid.

Example usage:

```bash
transcript_transformer predict AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACGGT RNA models/proteome/TIS_transformer_L_1.ckpt
transcript_transformer predict data/example_data.fa fa models/proteome/TIS_transformer_L_2.ckpt
```

Main function arguments, run `transcript_transformer predict -h` for a complete list:

```
positional arguments:
  input_data            path to JSON dict (h5) or fasta file, or RNA sequence
  input_type            type of input
  checkpoint            path to checkpoint of trained model

options:
  -h, --help            show this help message and exit
  --prob_th             minimum prediction threshold at which site is deemed worthy of attention (default: 0.01)
  --save_path           save file path (default: results)
  --output_type         file type of raw model outputs (default: npy)
```

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
