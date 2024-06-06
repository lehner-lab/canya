import sys
import argparse
import pkg_resources
import numpy as np
import pandas as pd
from . import utils
from . import modeling
batchsize=10000

def parse_inputs():
    parser = argparse.ArgumentParser(prog='CANYA', description='Command line interface for the CANYA package')
    parser.add_argument(
        '--input', default='example.fa', help='Input sequences, either a FASTA or a text file with\
        two *tab-delimited* columns with *no* header or column-names. Columns contain a \
        sequence idenity (arbirtrary) as well as the amino acid sequence.'
    )
    parser.add_argument(
        '--output', default='example_out', help='Name/directory of the output txt file. If run in\
        the default mode, CANYA will output a single, tab-delimited file named after this prefix\
        with two columns: (1) with the sequence identity (FASTA header or corresponding column of\
        the input text file) (2) The CANYA nucleation score.\
        If run in motif mode, a folder will be created that contains a file per sequence, where rows\
        are named with the specific cluster, columns after the sequence position, and each entry contains\
        the activation energy of the cluster at that position. Activation energies can be summarized with\
        the --act argument.'
    )
    parser.add_argument(
        '--mode', default='default', help='Mode to run CANYA. Default mode outputs a single\
        tab-delimited text file containing sequence identities and a CANYA score. \'motif\' \
        mode will dump a file per sequence containing the activation energies per clusterXposition.\
        Activation energies within a cluster can be summarized according to the --act argument.',
        choices=["default", "motif"]
    )
    parser.add_argument(
        '--act', default='max', help='Function by which to summarize filter activations for a given\
        cluster. Default is \'max\', other options are \'mean\' and \'min\'. Explicitly, this function\
        collapses the activation energies across all filters comprising a cluster for a given position.',
        choices=["max","mean","min"]
    )

    # Parse all command line arguments
    args = parser.parse_args()
    return args

def main():
    args=parse_inputs()
    if None in [args.input, args.output, args.mode, args.act]:
        logging.error('Usage: canya [-input [input]] [-output [output]] --mode default --act max')
        exit()
    canyamod=modeling.get_canya()
    seqlist=utils.get_input_sequences(args.input)
    print("Read input")
    resdfs=[]
    numseqs=len(seqlist[0])
    idxprocess=[np.array(x) for x in list(utils.chunks(list(range(numseqs)),batchsize))]
    sumseqsdone=0
    for curidx in idxprocess:
        seqstopred=np.array(seqlist[1])[curidx]
        seqstopred=seqstopred.tolist()
        namespred=np.array(seqlist[0])[curidx]
        namespred=namespred.tolist()
    
        preds=modeling.get_predictions(model=canyamod,sequences=seqstopred)
        curpreddf=pd.DataFrame({"seqid" : namespred,
                            "pred" : preds})
        mindf=curpreddf.groupby("seqid").min()
        mindf=pd.DataFrame({"seqid" : mindf.index.tolist(),
                           "pred" : mindf["pred"].tolist()},index=None)
        resdfs.append(mindf)
        sumseqsdone+=curpreddf.shape[0]
        propdone=sumseqsdone / numseqs * 100
        printstr="Finished "+ str(round(propdone)) + "% ("
        printstr+=str(sumseqsdone)+"/"+str(numseqs)+") "
        print(printstr + "of sequences")
        
    resdf=pd.concat(resdfs,axis=0)
    resdf=resdf.groupby("seqid").min()
    resdf=pd.DataFrame({"seqid" : resdf.index.tolist(),
                           "CANYA" : resdf["pred"].tolist()},index=None)
    resdf.to_csv(args.output,sep="\t",index=False)
    
if __name__ == '__main__':
    main()