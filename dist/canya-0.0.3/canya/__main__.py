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
        the default mode, CANYA will output a tab-delimited file named after this prefix\
        with two columns: \
        (1) with the sequence identity (FASTA header or corresponding column of the input text file)\
        (2) The CANYA nucleation score.\
        If run with the `--summarize no` option, CANYA will output a longer file that includes the output\
        of CANYA across the sequence (if the sequence is longer than 20 amino acids: \
        (1) The sequence identity\
        (2) The positions within the sequence\
        (3) The subsequence at the specified positions\
        (4) The corresponding CANYA score at the corresponding positions.',
    )
    parser.add_argument(
        '--mode', default='default', help='Mode to run CANYA. Default mode outputs a tab-delimited\
        text file containing sequence identities and a CANYA score.\
         \'ensemble\' mode will do the same but report the averaged scores across 10 models, \
        as well as an additional column that contains the standard deviation of scores across models.',
        choices=["default", "ensemble"]
    )
    parser.add_argument(
        '--summarize', default='median', help='Function by which to summarize CANYA scores across a sequence that\
        is longer than CANYA\'s input window of 20 amino acids. Default is \'median\', other options are\
         \'mean\', \'max\', and \'min\'. Explicitly, this function summarizes CANYA scores taken across\
        the sequence (those with length > 20) and reports one single score as calculated by this summarizing\
        function across the sequence. \'no\' will not summarize the score and will instead report the\
        CANYA score calculated at each subsequence in the sequence.',
        choices=["max","mean","min", "median", "no"]
    )

    # Parse all command line arguments
    args = parser.parse_args()
    return args

def main():
    args=parse_inputs()
    if None in [args.input, args.output, args.mode, args.summarize]:
        logging.error('Usage: canya [-input [input]] [-output [output]] --mode default --summarize median')
        exit()
    modellist=[]
    if args.mode=="ensemble":
        modellist=[modeling.get_canya("models/model_weights"+str(x)+".h5") for x in range(1,11)]
    else:
        modellist=[modeling.get_canya("models/model_weights1.h5")]
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
        posesseq=np.array(seqlist[2])[curidx]
        posesseq=posesseq.tolist()

        embeddedseqs=modeling.get_embedded_seqs(seqstopred)
        preds=[modeling.get_predictions(model=x,embeddedseqs=embeddedseqs) for x in modellist]
        if len(preds)==1:
            curpreddf=pd.DataFrame({"seqid" : namespred,
                                "seq" : seqstopred,
                                "pos" : posesseq,
                                "pred" : np.array(preds).flatten()})
        else:
            curpreddf=pd.DataFrame({"seqid" : namespred,
                                "seq" : seqstopred,
                                "pos" : posesseq,
                                "pred" : np.mean(np.array(preds),axis=0).flatten(),
                                "sd" : np.std(np.array(preds),axis=0).flatten()})
        resdfs.append(curpreddf)
        sumseqsdone+=curpreddf.shape[0]
        propdone=sumseqsdone / numseqs * 100
        printstr="Finished "+ str(round(propdone)) + "% ("
        printstr+=str(sumseqsdone)+"/"+str(numseqs)+") "
        print(printstr + "of sequences")
        
    resdf=pd.concat(resdfs,axis=0)
    if args.mode=="ensemble":
        print("Mode is ensemble, CANYA will not summarize")
    else:
        if args.summarize=="max":
            resdf=resdf.groupby("seqid").max()
            resdf=pd.DataFrame({"seqid" : resdf.index.tolist(),
                               "CANYA" : resdf["pred"].tolist()},index=None)
        elif args.summarize=="min":
            resdf=resdf.groupby("seqid").min()
            resdf=pd.DataFrame({"seqid" : resdf.index.tolist(),
                               "CANYA" : resdf["pred"].tolist()},index=None)
        elif args.summarize=="mean":
            resdf=resdf.groupby("seqid").mean()
            resdf=pd.DataFrame({"seqid" : resdf.index.tolist(),
                               "CANYA" : resdf["pred"].tolist()},index=None)
        elif args.summarize=="no":
            print("summarize set to \'no\', writing long format output")
        else:
            resdf=resdf.groupby("seqid").median()
            resdf=pd.DataFrame({"seqid" : resdf.index.tolist(),
                               "CANYA" : resdf["pred"].tolist()},index=None)
    resdf.to_csv(args.output,sep="\t",index=False)
    
if __name__ == '__main__':
    main()