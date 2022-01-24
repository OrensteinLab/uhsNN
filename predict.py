import argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" #added by lior to make tensorflow run on cpu
import tensorflow as tf
import itertools
import sys
from timeit import default_timer as timer
from multiprocessing import Pool
from pathlib import Path #added by lior

def buildDict(k):
    bases = ['A', 'C', 'G', 'T']
    basedict={"A": 0, "C": 1, "G": 2, "T": 3}
    intdict = {}
    kmerArray = [''.join(p) for p in itertools.product(bases, repeat=k)]
    for kmer in kmerArray:
        totval = 0
        for pos,char in enumerate(kmer):
            val = basedict[char] * pow(4, (k - pos - 1))
            totval += val
        intdict[totval] = kmer
    return intdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Makes predictions for provided k-mer length (k) and sequence length (L) with trained model (.h5) as input.')
    parser.add_argument('-k', metavar='k', type=int, help='K-mer size (k) for the UHS')
    parser.add_argument('-L', metavar= 'L', type=int, help='Sequence size (L) for the UHS')
    parser.add_argument('-m', metavar='m', help='Complete model (.h5) file')
    parser.add_argument('-t', metavar='t', help='Model threshold')
    parser.add_argument('-n', metavar= 'n', help='Number of threads')
    print("checkpoint")

    
    args = parser.parse_args()
    k = args.k
    L = args.L
#    m = args.m changed by lior
    m = Path(args.m)
    decycling=pd.read_csv("int/decyc"+str(k)+".int")
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=int(args.n), 
                        inter_op_parallelism_threads=2, 
                        allow_soft_placement=True,
                        device_count = {'CPU': int(args.n)})

    session = tf.compat.v1.Session(config=config)
    maxmaxk = 15
    intdict = buildDict(k)
    model = load_model(args.m, compile=False)
    print("Loaded model from disk")
    print(L)
    lst = np.array(list(itertools.product([0, 1], repeat=2*k)))
    padlst = np.pad(lst, ((0, 0),(0,maxmaxk*2-lst.shape[1])), 'constant', constant_values=((2, 2),(2,2)))
    lstL = np.append(np.full(lst.shape[0], L).reshape(lst.shape[0],1), padlst, axis=1)
    lstkL = np.append(np.full(lst.shape[0], k).reshape(lstL.shape[0],1), lstL, axis=1)
    start = timer()
    labels=model.predict(lstkL, batch_size=32768, workers=args.n, use_multiprocessing=True)
    end = timer()
    labels[decycling]=2
    print(labels.shape)
#    outF = open("preds.txt", "w")
    predfilename = "/data/liorkot/gen/xpreds/preds_" + str(k) + "_" + str(L) + ".txt"
    outF = open(predfilename, "w+")
    for i in range(len(labels)):
        outF.write(intdict[i] + '\t' + str(labels[i][0]))
        outF.write("\n")
    outF.close()
    predTime = end - start
    print("Predictions done, took " + str(predTime) + " seconds.")
    print("Starting PASHA...")
    start = timer()
    command = "./pasha " +  str(k) + " " + str(L) + " " + str(args.n) + " sets/decyc" + str(k) + ".txt " + "sets/hit" + str(k) + "_" + str(L) + "_" + str(args.t) + ".txt " + str(args.t)
    print(command)
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname + "/pashasrc/")
    stream = os.popen(command)
    output = stream.read()
    print(output)
    end = timer()
    PASHAtime = end - start
    print("PASHA done, took " + str(PASHAtime) + " seconds.")
    print("Total is " + str(predTime+PASHAtime) + " seconds.")
