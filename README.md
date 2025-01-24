## CANYA a neural net to predict aggregation propensity
![Alt text](canyafig.png)
CANYA is a hybrid neural network that was trained on 100,000 random peptides to predict their aggregation status in a massively parallel experiment of aggregation rates. We include here the package and technical details for setting up and running CANYA on your own sequences. Please see [biorxiv link] for further information.

### Installation
To start, you'll need a python installation with tensorflow, numpy, and pandas installed. If you don't have these, CANYA will attempt to install the respective packages (and versions) with which it was developed. In this case, we recommend using a blank virtual environment or conda environment and installing CANYA from there.

e.g. by conda:
```
conda create canyaenv python=3.9
conda activate canyaenv
```

CANYA can then be installed via pip:
```
python -m pip install --no-cache-dir https://github.com/lehner-lab/canya/tarball/master
```

### Running CANYA

Once installed, CANYA can be run very simply with the following options:

```--input``` Input sequences, either a FASTA or a text file with two *tab-delimited* columns with *no* header or column-names. Columns contain a sequence idenity (arbirtrary) as well as the amino acid sequence. See example data folder for examples.

```--output``` Name/directory of the output txt file. CANYA will output a single, tab-delimited file named after this prefix with two columns: (1) with the sequence identity (FASTA header or corresponding column of the input text file) (2) The CANYA nucleation score.

To run CANYA on the example file, run the following lines:
```
wget https://raw.githubusercontent.com/lehner-lab/canya/main/example_data/example.txt
canya --input example.txt --output example_out.txt
```

In addition, CANYA offers two other options:

```--summarize``` Either "no", which will report the CANYA score at every length-20 window along the sequence (rather than summarizing one score per sequence), or one of \{min, max, mean, median\}, which will summarize all scores along the sequence by using the specified function.

```--mode``` CANYA scores can be calculated using the model whose interpretations are presented in the paper (option "default"), or by taking the average of the top-10 most interpretable trained instances of CANYA ("ensemble"), which will also report the standard deviation of scores across the 10 models (i.e. epistemic uncertainty). 



CANYA has been tested on HPC and laptops---on a 2020 MacBook pro, using a single-core CPU, CANYA can generate predictions for roughly 100,000 sequences in less than a minute.
