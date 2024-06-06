import pandas as pd
import numpy as np
from itertools import groupby
windowsize=145


# check if the first line contains ">"
# if so, read as fasta, else tab-delim file
def isFasta(filename):
    filecheck=open(filename,"r")
    curline=filecheck.readline()
    fastafile=False
    if curline.startswith(">"):
        fastafile=True
        print("Detected FASTA input (starts with '>')")
    filecheck.close()
    return fastafile


"""
modified from Brent Pedersen
https://www.biostars.org/p/710/
given a fasta file. yield tuples of header, sequence
"""

def fasta_iter(fasta_name):

    fh = open(fasta_name)

    # ditch the boolean (x[0]) and just keep the header or sequence since
    # we know they alternate.
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()

        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())

        yield (headerStr, seq)

def get_input_sequences(filename):
    readAsFasta=isFasta(filename)
    seqs=[]
    seqnames=[]
    if(readAsFasta):
        fiter=fasta_iter(filename)
        for ff in fiter:
            curname, seq = ff
            if "X" in seq or "Z" in seq:
                continue
            seqadd=seq.split('*')[0]
            if len(seqadd) > 145:
                curseqs=[seqadd[i:i+windowsize] for i in range(0, len(seqadd), windowsize)]
            else:
                curseqs=[seqadd]
            seqs.extend(curseqs)
            seqnames.extend([curname]*len(curseqs))
    else:
        seqdf=pd.read_csv(filename,sep="\t",header=None)
        seqnames_prop=seqdf.iloc[:,0].tolist()
        seqs_prop=seqdf.iloc[:,1].tolist()
        for seq, curname in zip(seqs_prop,seqnames_prop):
            if "X" in seq or "Z" in seq:
                continue
            seqadd=seq.split('*')[0]
            if len(seqadd) > 145:
                curseqs=[seqadd[i:i+windowsize] for i in range(0, len(seqadd), windowsize)]
            else:
                curseqs=[seqadd]
            seqs.extend(curseqs)
            seqnames.extend([curname]*len(curseqs))
    return [seqnames, seqs]
        
# https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]







