## CANYA a neural net to predict nucleation propensity
![Alt text](canyafig.png)
CANYA is a hybrid neural network that was trained on 100,000 random peptides to predict their nucleation status in a massively parallel experiment of nucleation rates. We include here the package and technical details for setting up and running CANYA on your own sequences. Please see [biorxiv link] for further information.

### Installation
To start, you'll need a python installation with tensorflow, numpy, and pandas installed. If you don't have these, CANYA will attempt to install the respective packages (and versions) with which it was developed. In this case, we recommend using a blank virtual environment or conda environment and installing CANYA from there.

e.g. by conda:
```
conda create canyaenv python=3.9
conda activate canyaenv
```

CANYA can then be installed via pip:
```
python -m pip install --no-cache-dir https://github.com/mj-thompson/canya/tarball/master
```

### Running CANYA

Once installed, CANYA can be run very simply with the following options:
```--input``` Input sequences, either a FASTA or a text file with two *tab-delimited* columns with *no* header or column-names. Columns contain a sequence idenity (arbirtrary) as well as the amino acid sequence. See example data folder for examples.

```--output``` Name/directory of the output txt file. CANYA will output a single, tab-delimited file named after this prefix with two columns: (1) with the sequence identity (FASTA header or corresponding column of the input text file) (2) The CANYA nucleation score.

To run CANYA on the example file, run the following lines:
```
wget https://raw.githubusercontent.com/mj-thompson/canya/main/example_data/example.txt
canya --input example.txt --output example_out.txt
```

CANYA has been tested on HPC and laptops---on a 2020 MacBook pro, using a single-core CPU, CANYA can generate predictions for roughly 100,000 sequences in less than a minute.