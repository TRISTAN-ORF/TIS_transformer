# Model predictions

## `GRCh38v107_outputs.npy`
 
```
GRCh38v107_outputs.h5                       (h5py.file)
    seq                                     (h5py.group)
    ├── id                                  (h5py.dataset, dtype=str)
    ├── output                              (h5py.dataset, dtype=vlen(int))  
```

## `GRCh38v107_top_preds.csv`

A `.csv` table with the following columns:

<details>

| **Column name**    | **Definition**                                                                                                                          |
| :----------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| contig             | Contig ID                                                                                                                               |
| gene\_name         | Name of the Gene                                                                                                                        |
| tr\_ID             | Ensembl Identifier for the transcript                                                                                                   |
| TIS\_pos           | TIS location w.r.t. the transcript coordinates                                                                                          |
| output             | Model probability output, ranges from 0 to 1.                                                                                           |
| pos\_on\_tr        | Nucleotide position on the transcript (1-based coordinate)                                                                              |
| strand             | Strand on which the transcript is present                                                                                               |
| gen\_loc           | Genome coordinates of the TIS.                                                                                                          |
| tr\_len            | Length of the transcript                                                                                                                |
| target             | (boolean) Ensembl TIS Annotation                                                                                                        |
| exon\_ID           | Ensembl Identifier for the exon                                                                                                         |
| gene\_ID           | Ensembl identifier of the Gene                                                                                                          |
| tr\_support\_level | Transcript support level                                                                                                                |
| biotype            | Transcript biotype tag                                                                                                                  |
| output\_rank       | Rank of output w.r.t. all outputs on the chromosome, lower rank denotes higher model probability                                        |
| k\_rank            | Rank (k) of the output w.r.t. all outputs on the chromosome, scaled by the number of positive annotations as given by Ensembl per chrom |
| tr\_has\_target    | (boolean) Whether the transcript has a TIS as annotated by Ensembl (see target)                                                         |
| dist\_from\_target | Distance to the annotated TIS, when present                                                                                             |
| frame\_wrt\_target | Reading frame w.r.t. the annotated TIS, when present                                                                                    |
| start\_codon       | Start codon of the TIS                                                                                                                  |
| prot               | Protein sequence of the resulting coding sequence                                                                                       |
| prot\_len          | Length of resulting protein sequence                                                                                                    |
| TTS\_pos           | TTS location w.r.t. the transcript coordinates and given TIS position                                                                   |
| TTS\_on\_transcript| (boolean) Whether the resulting CDS region has a stop codon on transcript                                                               |
| stop\_codon        | stop codon of the resulting coding sequence                                                                                             |
| TISs\_on\_tr       | Number of TISs on the transcript when assuming the top k\*3 positions to be positive predictions                                        |
| ORF\_type          | Type of TIS annotation, element of [dORF, doORF, intORF, annotated ORF, uORF, uoORF, non-translated transcript]                         |
| bl\_uniprot        | Uniprot ID from top BLAST search hit                                                                                                    |
| bl\_perc\_id       | Percentage overlap score returned by top BLAST search hit                                                                               |
| bl\_e\_value       | E value of top BLAST search hit                                                                                                         |

</details>
