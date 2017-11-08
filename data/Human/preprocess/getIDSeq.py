import collections
import numpy as np
import scipy.sparse as sp
import cPickle as pkl
import json

det getIDSeq(positiveAFile, positiveBFile, proteinListFile, sequenceListFile):
    sequence = []
    proteinList = []
    sequenceList = []
    proteinMap = dict()

    with open(positiveAFile, 'r') as fa:
        linesa = fa.readlines()
    with open(positiveBFile, 'r') as fb:
        linesb = fb.readlines()

    for k in range(len(linesa)):
        if k%2==0:   #ID lines
            s = linesa[k][1:].strip()
            if s not in proteinMap:
                seq = linesa[k+1].strip()
                flag = True
                for x in seq:
                    if x=='U' or x=='X':
                        flag = False
                        break
                if flag == True and len(seq) > 50:
                    proteinMap[s] = seq

    for k in range(len(linesb)):
        if k%2==0:
            s = linesb[k][1:].strip()
            if s not in proteinMap:
                seq = linesb[k+1].strip()
                flag = True
                for x in seq:
                    if x=='U' or x=='X':
                        flag = False
                        break
                if flag==True and len(seq) > 50:
                    proteinMap[s] = seq

    for k,v in proteinMap.items():
        proteinL.append(k)
        sequenceL.append(v)

    print len(proteinL)


    with open(proteinListFile, 'w') as f:
        for idx, x in ex
    

