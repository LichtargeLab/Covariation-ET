'''
Created on Mar 10, 2017

@author: Benu Atri
'''
from multiprocessing import Pool, cpu_count
from sklearn.metrics import roc_curve, auc, mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from seaborn import heatmap
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from Bio import pairwise2
import cPickle as pickle
import numpy as np
import pylab as pl
import datetime
import argparse
import time
import csv
import sys
import re
import os
from IPython import embed


def parseArguments():
    '''
    parse arguments

    This method provides a nice interface for parsing command line arguments
    and includes help functionality.

    Returns:
    --------
    dict:
        A dictionary containing the arguments parsed from the command line and
        their arguments.
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--alignment', metavar='A', type=str, nargs=1,
                        help='The file path to the alignment to analyze in this run.')
    parser.add_argument('--pdb', metavar='P', type=str, nargs='?',
                        help='The file path to the PDB structure associated with the provided alignment.')
    parser.add_argument('--query', metavar='Q', type=str, nargs=1,
                        help='The name of the protein being queried in this analysis.')
    parser.add_argument('--threshold', metavar='T', type=float, nargs='?',
                        help='The distance within the molecular structure at which two residues are considered interacting.')
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--output', metavar='O', type=str, nargs='?',
                        default='./', help='File path to a directory where the results can be generated.')
    args = parser.parse_args()
    return vars(args)


def importAlignment(faFile, saveFile=None):
    '''
    Import alignments:

    This method imports alignments into an existing dictionary, and replaces .

    Parameters:
    -----------
    faFile: File
        File object holding a handle to an alignment file.
    saveFile: str
        Path to file in which the desired alignment was stored previously.
    Returns:
    --------
    alignment_dict: dict    
        Dictionary which will be used to store alignments from the file.
    seq_order: list
        List of sequence ids in the order in which they were parsed from the
        alignment file.
    '''
    start = time.time()
    if((saveFile is not None) and (os.path.exists(saveFile))):
        alignment, seqOrder = pickle.load(open(saveFile, 'rb'))
    else:
        alignment = {}
        seqOrder = []
        for line in faFile:
            if line.startswith(">"):
                key = line.rstrip()
                alignment[key] = ''
                seqOrder.append(key)
            else:
                alignment[key] += line.rstrip().replace('.',
                                                        '-').replace('_', '-')
        faFile.close()
        if(saveFile is not None):
            pickle.dump((alignment, seqOrder), open(saveFile, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Importing alignment took {} min'.format((end - start) / 60.0))
    return alignment, seqOrder


def writeOutAlignment(seqDict, seqOrder, fileName):
    outFile = open(fileName, 'wb')
    for seqId in seqOrder:
        if(seqId in seqDict):
            outFile.write(seqId + '\n')
            seqLen = len(seqDict[seqId])
            breaks = seqLen / 60
            if((seqLen % 60) != 0):
                breaks += 1
            for i in range(breaks):
                outFile.write(seqDict[seqId][0 + i * 60:60 + i * 60] + '\n')
        else:
            pass
    outFile.close()


def removeGaps(alignment, query, saveFile=None):
    '''
    Remove Gaps

    Removes all gaps from the query sequence and removes characters at the
    corresponding positions in all other sequences.

    Parameters:
    -----------
    alignment: dict
        Dictionary mapping sequence id to sequence.
    query: str
        The name id of the query sequence.
    saveFile: str
        Path to a file where the alignment with gaps in the query sequence
        removed was stored previously.

    Returns:
    --------
    str
        A new query name.
    dict
        A transform of the input dictionary without gaps.
    '''
    # Getting gapped columns for query
    start = time.time()
    if((saveFile is not None) and os.path.exists(saveFile)):
        queryName, newAlignment = pickle.load(
            open(saveFile, 'rb'))
    else:
        gap = ['-', '.', '_']
        queryGapInd = []
        queryName = '>query_{}'.format(query)
        for idc, char in enumerate(alignment[queryName]):
            if char in gap:
                queryGapInd.append(idc)
        if(len(queryGapInd) > 0):
            newAlignment = {}
            for key, value in alignment.iteritems():
                newAlignment[key] = value[0:queryGapInd[0]]
                for i in range(1, len(queryGapInd)):
                    newAlignment[key] += value[queryGapInd[i - 1] + 1:
                                               queryGapInd[i]]
                newAlignment[key] += value[queryGapInd[-1] + 1:]
        else:
            newAlignment = alignment
        if(saveFile is not None):
            pickle.dump((queryName, newAlignment),
                        open(saveFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Removing gaps took {} min'.format((end - start) / 60.0))
    return queryName, newAlignment


def alignment2num(alignment, keyList, seqLength, aaDict):
    '''
    Alignment2num

    Converts an alignment dictionary to a numerical representation.

    Parameters:
    -----------
    alignment: dict
        Dictionary where the keys are sequence ids and the values are the
        actual sequences used in the alignment.
    key_list: list
        Ordered set of sequence ids which specifies the ordering of the
        sequences along the row dimension of the resulting matrix.
    seq_length: int
        Length of the sequences in the alignment.
    aa_dict: dict
        Dictionary mapping characters which can appear in the alignment to
        digits for representation.
    Returns:
    --------
    numpy ndarray
        A matrix of values which represents the alignment numerically. The
        first dimension (rows) will iterate over sequences (as ordered by
        key_list) while the second dimension (columns) will iterate over
        positions in the sequence (0 to seq_length).
    '''
    alignment2Num = np.zeros((len(alignment), seqLength))
    for i in range(len(keyList)):
        for j in range(seqLength):
            alignment2Num[i, j] = aaDict[alignment[keyList[i]][j]]
    return alignment2Num


def distanceMatrix(alignment, aaDict, saveFiles=None):
    '''
    Distance matrix

    Computes the sequence identity distance between a set of sequences and
    returns a matrix of the pairwise distances.

    Parameters:
    -----------
    alignment_dict: dict
        Dictionary of aligned sequences. This is meant to be a corrected
        dictionary where all gaps have been removed from the query sequence,
        and the same positions have been removed from other sequences.
    aa_dict: dict
        Dictionary mapping amino acids and other possible characters from an
        alignment to integer representations.
    saveFiles: tuple
        A tuple or list containing two file paths the first should be the path
        for a .npz file containing distances between sequences in the alignment
        (leave out the .npz as it will be added automatically) and the file
        path for a .pkl file containing the the sequence order for the distance
        matrix.
    Returns:
    --------
    matrix
        A symmetric matrix of pairwise distance computed between two sequences
        using the sequence identity metric.
    list
        List of the sequence identifiers in the order in which they appear in
        the matrix.
    '''
    # Generate distance_matrix: Calculating Sequence Identity
    start = time.time()
    if((saveFiles is not None) and os.path.exists(saveFiles[0]) and
       os.path.exists(saveFiles[1])):
        valueMatrix = np.load(saveFiles[0] + '.npz')['X']
        keyList = pickle.load(open(saveFiles[1], 'rb'))
    else:
        keyList = sorted(alignment.keys())
        valueMatrix = np.zeros([len(alignment), len(alignment)])
        seqLength = len(alignment[keyList[0]])
        alignment2Num = alignment2num(alignment, keyList, seqLength, aaDict)
        for i in range(len(keyList)):
            check = alignment2Num - alignment2Num[i]
            valueMatrix[i] = np.sum(check == 0, axis=1)
        valueMatrix[np.arange(len(keyList)), np.arange(len(keyList))] = 0
        valueMatrix /= seqLength
        if(saveFiles is not None):
            np.savez(saveFiles[0], X=valueMatrix)
            pickle.dump(keyList, open(saveFiles[1], 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Computing the distance matrix took {} min'.format(
        (end - start) / 60.0))
    return valueMatrix, keyList


def aggClustering(nCluster, X, keyList, precomputed=False):
    '''
    Agglomerative clustering

    Performs agglomerative clustering on a matrix of pairwise distances between
    sequences in the alignment being analyzed.

    Parameters:
    -----------
    nCluster: int
        The number of clusters to separate sequences into.
    X: numpy nd_array
        The distance matrix computed between the sequences.
    keyList: list
        Set of sequence ids ordered to correspond with the ordering of
        sequences along the dimensions of X.
    precomputed: boolean
        Whether or not to use the distances from X as the distances to cluster
        on, the alternative is to compute a new distance matrix based on X
        using Euclidean distance.
    Returns:
    --------
    dict
        A dictionary with cluster number as the key and a list of sequences in
        the specified cluster as a value.
    set
        A unique sorted set of the cluster values.
    '''
    start = time.time()
    if(precomputed):
        affinity = 'precomputed'
        linkage = 'complete'
    else:
        affinity = 'euclidean'
        linkage = 'ward'
    model = AgglomerativeClustering(affinity=affinity, linkage=linkage,
                                    n_clusters=nCluster)
    model.fit(X)
    # ordered list of cluster ids
    # unique and sorted cluster ids for e.g. for n_cluster = 2, g = [0,1]
    clusterList = model.labels_.tolist()
    clusterDict = {}
    ####---------------------------------------#####
    #       Mapping Clusters to Sequences
    ####---------------------------------------#####
    for i in range(len(clusterList)):
        key = clusterList[i]
        if(key not in clusterDict):
            clusterDict[key] = []
        clusterDict[key].append(keyList[i])
    end = time.time()
    print('Performing agglomerative clustering took {} min'.format(
        (end - start) / 60.0))
    return clusterDict, set(clusterList)


def wholeAnalysis(alignment, aaDict, saveFile=None):
    '''
    Whole Analysis

    Generates the MIP matrix.

    Parameters:
    -----------
    alignment_dict: dict
        Dictionary of aligned sequences. This is meant to be a corrected
        dictionary where all gaps have been removed from the query sequence,
        and the same positions have been removed from other sequences.
    aa_dict: list
        Dictionary of amino acids and other characters possible in an alignment
        mapping to integer representations.
    saveFile: str
        File path to a previously stored MIP matrix (.npz should be excluded as
        it will be added automatically).
    Returns:
    --------
    matrix
        Matrix of MIP scores which has dimensions seq_length by seq_length.
    '''
    start = time.time()
    if((saveFile is not None) and os.path.exists(saveFile + '.npz')):
        mipMatrix = np.load(saveFile + '.npz')['wholeMIP']
    else:
        overallMMI = 0.0
        keyList = alignment.keys()
        seqLength = len(alignment[keyList[0]])
        # generate an MI matrix for each cluster
        miMatrix = np.zeros((seqLength, seqLength))
        # Vector of 1 column
        MMI = np.zeros(seqLength)
        apcMatrix = np.zeros((seqLength, seqLength))
        mipMatrix = np.zeros((seqLength, seqLength))
        # Create matrix converting sequences of amino acids to sequences of integers
        # representing sequences of amino acids.
        alignment2Num = alignment2num(alignment, keyList, seqLength, aaDict)
        # Generate MI matrix from alignment2Num matrix, the MMI matrix,
        # and overallMMI
        for i in range(seqLength):
            for j in range(i + 1, seqLength):
                columnI = alignment2Num[:, i]
                columnJ = alignment2Num[:, j]
                currMIS = mutual_info_score(
                    columnI, columnJ, contingency=None)
                # AW: divides by individual entropies to normalize.
                miMatrix[i, j] = miMatrix[j, i] = currMIS
                overallMMI += currMIS
        MMI += np.sum(miMatrix, axis=1)
        MMI -= miMatrix[np.arange(seqLength), np.arange(seqLength)]
        MMI /= (seqLength - 1)
        overallMMI = 2.0 * (overallMMI / (seqLength - 1)) / seqLength
        # Calculating APC
        apcMatrix += np.outer(MMI, MMI)
        apcMatrix[np.arange(seqLength), np.arange(seqLength)] = 0
        apcMatrix /= overallMMI
        # Defining MIP matrix
        mipMatrix += miMatrix - apcMatrix
        mipMatrix[np.arange(seqLength), np.arange(seqLength)] = 0
        if(saveFile is not None):
            np.savez(saveFile, wholeMIP=mipMatrix)
    end = time.time()
    print('Whole analysis took {} min'.format((end - start) / 60.0))
    return mipMatrix


def importPDB(pdbFile, saveFile=None):
    '''
    import_pdb

    This method imports a PDB files information generating a list of lists. Each
    list contains the Amino Acid 3-letter abbreviation, residue number, x, y,
    and z coordinate.

    Parameters:
    -----------
    pdbFile: File
        The file handle for the PDB file.
    saveFile: str
        The file path to a previously stored PDB file data structure.
    Returns:
    --------
    dict:
        A dictionary mapping a residue number to its spatial position in 3D.
    list:
        A sorted list of residue numbers from the PDB file.
    dict:
        A dictionary mapping residue number to the name of the residue at that
        position.
    '''
    start = time.time()
    convertAA = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'ASX': 'B',
                 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLX': 'Z', 'GLY': 'G',
                 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M',
                 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
                 'TYR': 'Y', 'VAL': 'V'}
    if((saveFile is not None) and os.path.exists(saveFile)):
        residue3D, pdbResidueList, residuePos, seq = pickle.load(
            open(saveFile, 'rb'))
    else:
        residue3D = {}
        pdbResidueList = []
        residuePos = {}
        seq = []
        prevRes = None
        pdbPattern = r'ATOM\s*(\d+)\s*(\w*)\s*([A-Z]{3})\s*([A-Z])\s*(\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*([A-Z])'
        for line in pdbFile:
            res = re.match(pdbPattern, line)
            if not res:
                continue
            resName = convertAA[res.group(3)]
            resNum = int(res.group(5))
            resAtomList = np.asarray([float(res.group(6)),
                                      float(res.group(7)),
                                      float(res.group(8))])
            try:
                residue3D[resNum].append(resAtomList)
            except KeyError:
                if(prevRes):
                    residue3D[prevRes] = np.vstack(residue3D[prevRes])
                prevRes = resNum
                residue3D[resNum] = [resAtomList]
                pdbResidueList.append(resNum)
                residuePos[resNum] = resName
                seq.append(resName)
        residue3D[prevRes] = np.vstack(residue3D[prevRes])
        # list of sorted residues - necessary for those where res1 is not 1
        pdbResidueList = sorted(pdbResidueList)
        seq = ''.join(seq)
        pdbFile.close()
        if(saveFile is not None):
            pickle.dump((residue3D, pdbResidueList, residuePos, seq),
                        open(saveFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Importing the PDB file took {} min'.format((end - start) / 60.0))
    return residue3D, pdbResidueList, residuePos, seq


def mapAlignmentToPDBSeq(fastaSeq, pdbSeq):
    '''
    Map sequence positions between query from alignment and residues in PDB

    Parameters:
    -----------
    fastaSeq: str
        A string providing the amino acid (single letter abbreviations)
        sequence for the protein.
    pdbSeq: str
        A string providing the amino acid (single letter abbreviations)
        sequence for the protein.
    Returns:
    --------
    dict
        A structure mapping the index of the positions in the fasta sequence
        which align to positions in the PDB sequence based on a local alignment
        with no mismatches allowed.
    '''
    alignments = pairwise2.align.globalxs(query_sequence, pdbSeq, -1, 0)
    from Bio.pairwise2 import format_alignment
    print(format_alignment(*alignments[0]))
    fCounter = 0
    pCounter = 0
    fToPMap = {}
    for i in range(len(alignments[0][0])):
        #         print('i: {}'.format(i))
        if((alignments[0][0][i] != '-') and (alignments[0][1][i] != '-')):
            fToPMap[fCounter] = pCounter
        if(alignments[0][0][i] != '-'):
            #             print('Alignment: {}'.format(fCounter))
            fCounter += 1
        if(alignments[0][1][i] != '-'):
            #             print('PDB: {}'.format(pCounter))
            pCounter += 1
#     print fToPMap
    return fToPMap


def findDistance(querySequence, qToPMap, residue3D, pdbResidueList,
                 saveFile=None):
    '''
    Find distance

    This code takes in an input of a pdb file and outputs a dictionary with the
    nearest atom distance between two residues.

    Parameters:
    -----------
    query_sequence: str
        A string providing the amino acid (single letter abbreviations)
        sequence for the protein.
    qToPMap: dict
        A structure mapping the index of the positions in the fasta sequence
        which align to positions in the PDB sequence based on a local alignment
        with no mismatches allowed.
    residue_3d: dictionary
        A dictionary mapping a residue position from the PDB file to its three
        dimensional position.
    pdb_residue_list: list
        A sorted list of residue numbers from the PDB file.
    saveFile: str
        File name and/or location of file containing a previously computed set
        of distance data for a PDB structure.
    Returns:
    list
        List of minimum distances between residues, sorted by the ordering of
        residues in pdb_residue_list.
    --------
    '''
    start = time.time()
    if((saveFile is not None) and os.path.exists(saveFile)):
        sortedPDBDist = np.load(saveFile + '.npz')['pdbDists']
    else:
        qLen = len(querySequence)
        sortedPDBDist = []
        # Loop over all residues in the pdb
#         for i in range(len(pdb_residue_list)):
        for i in range(qLen):
            # Loop over residues to calculate distance between all residues i
            # and j
            #             for j in range(i + 1, len(pdb_residue_list)):
            for j in range(i + 1, qLen):
                # Getting the 3d coordinates for every atom in each residue.
                # iterating over all pairs to find all distances
                #                 key1 = pdb_residue_list[i]
                #                 key2 = pdb_residue_list[j]
                try:
                    key1 = pdbResidueList[qToPMap[i]]
                    key2 = pdbResidueList[qToPMap[j]]
                    # finding the minimum value from the distance array
                    # Making dictionary of all min values indexed by the two residue
                    # names
                    res1 = residue3D[key2] - residue3D[key1][:, np.newaxis]
                    norms = np.linalg.norm(res1, axis=2)
                    sortedPDBDist.append(np.min(norms))
                except(KeyError):
                    sortedPDBDist.append(float('NaN'))
        sortedPDBDist = np.asarray(sortedPDBDist)
        if(saveFile is not None):
            np.savez(saveFile, pdbDists=sortedPDBDist)
    end = time.time()
    print('Computing the distance matrix based on the PDB file took {} min'.format(
        (end - start) / 60.0))
    return sortedPDBDist


def poolInit(a, seqDists, sK, fAD, sO, sL):
    '''
    poolInit

    A function which initializes processes spawned in a worker pool performing
    the etMIPWorker function.  This provides a set of variables to all working
    processes which are shared.

    Parameters:
    -----------
    a: dict
        A dictionary mapping amino acids and other characters found in
        alignments to dictionary representations.
    seqDists: numpy ndarray
        Matrix of distances between sequences in the initial alignments.
    sK: list
        A set of sorted sequence ids with the same ordering as that used in the
        construction of seqDists.
    fAD: dictionary
        A corrected dictionary mapping sequence ids to sequences.
    sO: list
        A list of sequence ids in the order which they should appear in an
        alignment.
    sL: int
        The length of the query sequence
    '''
    global aaDict
    aaDict = a
    global X
    X = seqDists
    global sortedKeys
    sortedKeys = sK
    global fixedAlignment
    fixedAlignment = fAD
    global seqOrder
    seqOrder = sO
    global seqLen
    seqLen = sL


def etMIPWorker(inTup):
    '''
    ETMIP Worker

    Performs clustering and calculation of cluster dependent sequence distances.
    This function requires initialization of threads with poolInit, or setting
    of global variables as described in that function.

    Parameters:
    -----------
    inTup: tuple
        Tuple containing the number of clusters to form during agglomerative
        clustering.
    Returns:
    --------
    int
        The number of clusters formed
    numpy ndarray
        A matrix of pairwise distances between sequences based on subalignments
        formed during clustering.
    float
        The time in seconds which it took to perform clustering.
    '''
    clus = inTup[0]
#     resMatrix = None
    start = time.time()
    print "Starting clustering: K={}".format(clus)
    clusterDict, clusterDet = aggClustering(clus, X, sortedKeys)
    rawScores = np.zeros((clus, seqLen, seqLen))
    for c in clusterDet:
        newAlignment = {}
        for key in clusterDict[c]:
            newAlignment[key] = fixedAlignment[key]
        writeOutAlignment(newAlignment, seqOrder,
                          'AligmentForK{}_{}.fa'.format(clus, c))
        clusteredMIPMatrix = wholeAnalysis(newAlignment, aaDict)
#         if(resMatrix is None):
#             resMatrix = clusteredMIPMatrix
#         else:
#             resMatrix += clusteredMIPMatrix
        rawScores[c] = clusteredMIPMatrix
    resMatrix = np.sum(rawScores, axis=0)
    end = time.time()
    timeElapsed = end - start
    print('ETMIP worker took {} min'.format(timeElapsed / 60.0))
    return clus, resMatrix, timeElapsed, rawScores


def poolInit2(c, seqL, PDBRL, sPDBD):
    '''
    pool_init2

    A function which initializes processes spawned in a worker pool performing
    the et_mip_worker2 function.  This provides a set of variables to all working
    processes which are shared.

    Parameters:
    -----------
    c: float
        The threshold distance at which to consider two residues as interacting
        with one another.
    PDBRL: list
        List of PDB residue numbers sorted by position.
    sPDBD: list
        List of distances between PDB residues sorted by pairwise residue
        numbers.
    '''
    global cutoff
    cutoff = c
    global seqLen
    seqLen = seqL
    global pdb_residue_list
    pdbResidueList = PDBRL
    global sortedPDBDist
    sortedPDBDist = sPDBD


def etMIPWorker2(inTup):
    '''
    ETMIP Worker 2

    Performs clustering and calculation of cluster dependent sequence distances.
    This function requires initialization of threads with poolInit, or setting
    of global variables as described in that function.

    Parameters:
    -----------
    inTup: tuple
        Tuple containing the number of clusters to form during agglomerative
        clustering and a matrix which is the result of summing the original
        MIP matrix and the matrix resulting from clustering at this clustering
        and lower clusterings.
    Returns:
    --------
    int
        Number of clusters.
    list
        Coverage values for this clustering.
    list
        Positions corresponding to the coverage values.
    list
        List of false positive rates.
    list
        List of true positive rates.
    float
        The ROCAUC value for this clustering.
    '''
    clus, summedMatrix = inTup
    start = time.time()
    scorePositions = []
    etmipResScoreList = []
    for i in range(0, seqLen):
        for j in range(i + 1, seqLen):
            newkey1 = "{}_{}".format(i + 1, j + 1)
            scorePositions.append(newkey1)
            etmipResScoreList.append(summedMatrix[i][j])
    etmipResScoreList = np.asarray(etmipResScoreList)
    # Converting to coverage
    etmiplistCoverage = []
    numPos = float(len(etmipResScoreList))
    for i in range(len(etmipResScoreList)):
        computeCoverage = (((np.sum((etmipResScoreList[i] >= etmipResScoreList)
                                    * 1.0) - 1) * 100) / numPos)
        etmiplistCoverage.append(computeCoverage)
    # AUC computation
    if((sortedPDBDist is not None) and
       (len(etmiplistCoverage) != len(sortedPDBDist))):
        print "lengths do not match"
        sys.exit()
    if(sortedPDBDist is not None):
        y_true1 = ((sortedPDBDist <= cutoff) * 1)
        fpr, tpr, _thresholds = roc_curve(y_true1, etmiplistCoverage,
                                          pos_label=1)
        roc_auc = auc(fpr, tpr)
    else:
        fpr = None
        tpr = None
        roc_auc = None
    end = time.time()
    timeElapsed = end - start
    print('ETMIP worker 2 took {} min'.format(timeElapsed / 60.0))
    return (clus, etmipResScoreList, etmiplistCoverage, scorePositions,
            timeElapsed, fpr, tpr, roc_auc)


def determineUsablePositions(alignment, ratio):
    '''
    '''
    gaps = (alignment == 21) * 1.0
    perColumn = np.sum(gaps, axis=0)
    percentGaps = perColumn / alignment.shape[0]
    usablePositions = np.where(percentGaps >= ratio)[0]
    evidence = (np.ones(alignment.shape[1]) * alignment.shape[0]) - perColumn
    return usablePositions, evidence


def plotAUC(fpr, tpr, rocAUC, qName, clus, today, cutoff):
    '''
    Plot AUC

    This function plots and saves the AUCROC.  The image will be stored in the
    eps format with dpi=1000 using a name specified by the query name, cutoff,
    clustering constant, and date.

    Parameters:
    fpr: list
        List of false positive rate values.
    tpr: list
        List of true positive rate values.
    rocAUC: float
        Float specifying the calculated AUC for the curve.
    qName: str
        Name of the query protein
    clus: int
        Number of clusters created
    today: date
        The days date
    cutoff: int
        The distance used for proximity cutoff in the PDB structure.
    '''
    start = time.time()
    pl.plot(fpr, tpr, label='(AUC = {0:.2f})'.format(rocAUC))
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    title = 'Ability to predict positive contacts in {}, Cluster = {}'.format(
        qName, clus)
    pl.title(title)
    pl.legend(loc="lower right")
    imagename = '{}{}A_C{}_{}roc.eps'.format(
        qName, cutoff, clus, today)
    pl.savefig(imagename, format='eps', dpi=1000, fontsize=8)
    pl.close()
    end = time.time()
    print('Plotting the AUC plot took {} min'.format((end - start) / 60.0))


def writeOutClusterScoring(today, qName, seq, clus, scorePositions,
                           etmipResScoreList, etmiplistCoverage,
                           clusterScores, originalMIP, clusterMat, summedMat):
    '''
    Write out clustering scoring results

    This method writes the results of the clustering to file.

    Parameters:
    today: date
        Todays date.
    qName: str
        The name of the query protein
    clus: int
        The number of clusters created
    scorePositions: list
        A list of the order in which the sequence distances are presented.
        The element format is {}_{} where each {} is the number of a sequence
        in the alignment.
    etmiplistCoverage: list
        The coverage of a specific sequence comparison
    sortedPDBDist: numpy nd array
        Array of the distances between sequences, sorted by sequence indices.
    seq: Str
        The query alignment sequence.
    '''
    start = time.time()
    convertAA = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'B': 'ASX',
                 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'Z': 'GLX', 'G': 'GLY',
                 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET',
                 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
                 'Y': 'TYR', 'V': 'VAL'}
    e = "{}_{}_{}.all_scores.txt".format(today, qName, clus)
    etMIPOutFile = open(e, "wb")
    etMIPWriter = csv.writer(etMIPOutFile, delimiter='\t')
    etMIPWriter.writerow(['Pos1', 'AA1', 'Pos2', 'AA2', 'OriginalScore'] +
                         ['C.' + i for i in map(str, range(1, clus + 1))] +
                         ['Cluster_Score', 'Summed_Score', 'ETMIp_Score',
                          'ETMIp_Coverage'])
    for i in range(0, len(seq)):
        for j in range(i + 1, len(seq)):
            res1 = i + 1
            res2 = j + 1
            key = '{}_{}'.format(i + 1, j + 1)
            ind = scorePositions.index(key)
            rowP1 = [res1, convertAA[seq[i]], res2, convertAA[seq[j]],
                     round(originalMIP[i, j], 2)]
            rowP2 = [round(clusterScores[c, i, j], 2) for c in range(clus)]
            rowP3 = [round(clusterMat[i, j], 2), round(summedMat[i, j], 2),
                     round(etmipResScoreList[ind], 2),
                     round(etmiplistCoverage[ind], 2)]
            etMIPWriter.writerow(rowP1 + rowP2 + rowP3)
    etMIPOutFile.close()
    end = time.time()
    print('Writing the ETMIP worker data to file took {} min'.format(
        (end - start) / 60.0))


def writeOutClusteringResults(today, qName, cutoff, clus, scorePositions,
                              etmipResScoreList, etmiplistCoverage, seq,
                              sortedPDBDist, pdbPositions):
    '''
    Write out clustering results

    This method writes the results of the clustering to file.

    Parameters:
    today: date
        Todays date.
    qName: str
        The name of the query protein
    clus: int
        The number of clusters created
    scorePositions: list
        A list of the order in which the sequence distances are presented.
        The element format is {}_{} where each {} is the number of a sequence
        in the alignment.
    etmiplistCoverage: list
        The coverage of a specific sequence comparison
    sortedPDBDist: numpy nd array
        Array of the distances between sequences, sorted by sequence indices.
    seq: Str
        The query alignment sequence.
    '''
    start = time.time()
    convertAA = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'B': 'ASX',
                 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'Z': 'GLX', 'G': 'GLY',
                 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET',
                 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
                 'Y': 'TYR', 'V': 'VAL'}
    e = "{}_{}_{}.etmipCVG.clustered.txt".format(today, qName, clus)
    etMIPOutFile = open(e, "w+")
    header = '{}\t({})\t{}\t({})\t{}\t{}\t{}\t{}\n'.format(
        'Pos1', 'AA1', 'Pos2', 'AA2', 'ETMIp_Score', 'ETMIp_Coverage',
        'Residue_Dist', 'Within_Threshold', 'Cluster')
    etMIPOutFile.write(header)
    counter = 0
    for i in range(0, len(seq)):
        for j in range(i + 1, len(seq)):
            if(sortedPDBDist is None):
                res1 = i + 1
                res2 = j + 1
                r = '-'
                dist = '-'
            else:
                res1 = pdbPositions[i]
                res2 = pdbPositions[j]
                dist = round(sortedPDBDist[counter], 2)
                if(sortedPDBDist[counter] <= cutoff):
                    r = 1
                elif(np.isnan(sortedPDBDist[counter])):
                    r = '-'
                else:
                    r = 0
            key = '{}_{}'.format(i + 1, j + 1)
            ind = scorePositions.index(key)
            etMIPOutputLine = '{}\t({})\t{}\t({})\t{}\t{}\t{}\t{}\t{}\n'.format(
                res1, convertAA[seq[i]], res2, convertAA[seq[j]],
                round(etmipResScoreList[ind], 2),
                round(etmiplistCoverage[ind], 2),
                dist, r, clus)
            etMIPOutFile.write(etMIPOutputLine)
            counter += 1
    etMIPOutFile.close()
    end = time.time()
    print('Writing the ETMIP worker data to file took {} min'.format(
        (end - start) / 60.0))


def writeFinalResults(qName, today, sequenceOrder, seqLength, cutoff, outDict):
    '''
    Write final results

    This method writes the final results to file for an analysis.  In this case
    that consists of the cluster numbers, the resulting AUCs, and the time
    spent in processing.

    Parameters:
    -----------
    qName: str
        The id for the query sequence.
    today: str
        The current date in string format.
    sequenceOrder: list
        A list of the sequence ids used in the alignment in sorted order.
    seq_length: int
        The length of the gap removed sequences from the alignment.
    cutoff: float
        The distance threshold for interaction between two residues in a
        protein structure.
    outDict: dict
        A dictionary with the lines of output mapped to the clustering
        constants for which they were produced.
    '''
    o = '{}_{}etmipAUC_results.txt'.format(qName, today)
    outfile = open(o, 'w+')
    proteininfo = ("Protein/id: " + qName + " Alignment Size: " +
                   str(len(sequenceOrder)) + " Length of protein: " +
                   str(seqLength) + " Cutoff: " + str(cutoff) + "\n")
    outfile.write(proteininfo)
    outfile.write("#OfClusters\tAUC\tRunTime\n")
    for key in sorted(outDict.keys()):
        outfile.write(outDict[key])


def heatmapPlot(name, dataMat):
    print('In Heatmapping')
    dmMax = np.max(dataMat)
    print('Identified max')
    dmMin = np.min(dataMat)
    print('Identified min')
    plotMax = max([dmMax, abs(dmMin)])
    print('Determined highest value')
    heatMap = heatmap(data=dataMat, cmap=cm.jet, center=0.0, vmin=-1 * plotMax,
                      vmax=plotMax, cbar=True, square=True)
    print('Plotted heat map')
    plt.title(name)
    print('Altered title')
    plt.savefig(name.replace(' ', '_') + '.pdf')
    print('Saved figure')
    plt.clf()
    print('Cleared figure')


def surfacePlot(name, dataMat):
    dmMax = np.max(dataMat)
    dmMin = np.min(dataMat)
    plotMax = max([dmMax, abs(dmMin)])
    X = Y = np.arange(max(dataMat.shape))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, dataMat, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1 * plotMax, plotMax)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(name.replace(' ', '_') + '.pdf')
    plt.clf()


####--------------------------------------------------------#####
### BODY OF CODE ##
####--------------------------------------------------------#####
if __name__ == '__main__':
    start = time.time()
    # Read input from the command line
    args = parseArguments()
    ###########################################################################
    # Set up global variables
    ###########################################################################
    today = str(datetime.date.today())
#     neighbor_list = []
    gap_list = ["-", ".", "_"]
#     aa_list = []
#     # comment out for actual dataset
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
#     aa_gap_list = aa_list + gap_list
#     i_j_list = []
    # {seq1_seq2} = distancescore
#     seq12_distscore_dict = {}
#     key = ''
#     temp_aa = ''
    ###########################################################################
    # Set up input variables
    ###########################################################################
    files = open(args['alignment'][0], 'rb')
    processes = args['processes']
    pCount = cpu_count()
    if(processes > pCount):
        processes = pCount
    ###########################################################################
    # Set up output location
    ###########################################################################
    startDir = os.getcwd()
    createFolder = (args['output'] + str(today) + "/" + args['query'][0])
    if not os.path.exists(createFolder):
        os.makedirs(createFolder)
        print "creating new folder"
    os.chdir(createFolder)

    print 'Starting ETMIP'
    # Import alignment information: this will be our alignment
    alignment_dict, seqOrder = importAlignment(files, 'alignment_dict.pkl')
    # Remove gaps from aligned query sequences
    query_name, fixed_alignment_dict = removeGaps(alignment_dict,
                                                  args['query'][0],
                                                  'ungapped_alignment.pkl')
    query_sequence = fixed_alignment_dict[query_name]
    writeOutAlignment(fixed_alignment_dict, seqOrder, 'UngappedAlignment.fa')
    print('Query Sequence:')
    print(query_sequence)
    seq_length = len(query_sequence)
    # I will get a corr_dict for method x for all residue pairs FOR ONE PROTEIN
    X, sequence_order = distanceMatrix(fixed_alignment_dict, aa_dict,
                                       ('X', 'seq_order.pkl'))
    # Generate MIP Matrix
    wholeMIP_Matrix = wholeAnalysis(fixed_alignment_dict, aa_dict, 'wholeMIP')
    ###########################################################################
    # Set up for remaining analysis
    ###########################################################################
    if(args['pdb']):
        print(os.getcwd())
        os.chdir(startDir)
        pdbfilename = open(args['pdb'], 'rb')
        os.chdir(createFolder)
        residuedictionary, pdb_residue_list, ResidueDict, pdbSeq = importPDB(
            pdbfilename, 'pdbData.pkl')
        mapAToP = mapAlignmentToPDBSeq(query_sequence, pdbSeq)
        sortedPDBDist = findDistance(query_sequence, mapAToP,
                                     residuedictionary, pdb_residue_list,
                                     'PDBdistances')
    else:
        pdb_residue_list = None
        ResidueDict = None
        sortedPDBDist = None
        pdbSeq = None
#     from IPython import embed
#     embed()
    print('PDB Sequence')
    print(pdbSeq)

#     exit()
    ###########################################################################
    # Perform multiprocessing of clustering method
    ###########################################################################
    clusters = [2, 3, 5, 7, 10, 25]
    resMats = {}
    resTimes = {}
    rawClusters = {}
    if(processes == 1):
        poolInit(aa_dict, X, sequence_order, fixed_alignment_dict, seqOrder,
                 seq_length)
        for c in clusters:
            clus, mat, times, rawCScores = etMIPWorker((c,))
            resMats[clus] = mat
            resTimes[clus] = times
            rawClusters[clus] = rawCScores
    else:
        pool = Pool(processes=processes, initializer=poolInit,
                    initargs=(aa_dict, X, sequence_order, fixed_alignment_dict,
                              seqOrder, seq_length))
        res1 = pool.map_async(etMIPWorker, [(x,) for x in clusters])
        pool.close()
        pool.join()
        res1 = res1.get()
        for r in res1:
            resMats[r[0]] = r[1]
            resTimes[r[0]] = r[2]
            rawClusters[r[0]] = r[3]
#     summedMatrices = [np.zeros(wholeMIP_Matrix.shape)] * len(clusters)
    summedMatrices = {i: np.zeros(wholeMIP_Matrix.shape)
                      for i in clusters}
    for i in clusters:
        #     for i in range(len(clusters)):
        #         summedMatrices[i] = summedMatrices[i] + wholeMIP_Matrix
        summedMatrices[i] += wholeMIP_Matrix
#         for j in range(0, i):
        for j in [c for c in clusters if c <= i]:
            #             summedMatrices[i] += resMats[clusters[j]]
            #             summedMatrices[i] += resMats[i]
            summedMatrices[i] += resMats[j]
    resRawScore = {}
    resCoverage = {}
    resScorePos = {}
    resAUCROC = {}
    if(processes == 1):
        #         for i in range(len(clusters)):
        for clus in clusters:
            poolInit2(args['threshold'], seq_length, pdb_residue_list,
                      sortedPDBDist)
            r = etMIPWorker2((clus, summedMatrices[clus]))
#             r = et_mip_worker2((clusters[i], summedMatrices[i]))
            resRawScore[r[0]] = r[1]
            resCoverage[r[0]] = r[2]
            resScorePos[r[0]] = r[3]
            resTimes[r[0]] += r[4]
            resAUCROC[r[0]] = r[5:]
    else:
        pool2 = Pool(processes=processes, initializer=poolInit2,
                     initargs=(args['threshold'], seq_length,
                               pdb_residue_list, sortedPDBDist))
        res2 = pool2.map_async(etMIPWorker2, [(clus, summedMatrices[clus])
                                              for clus in clusters])
#         res2 = pool2.map_async(et_mip_worker2, [(clusters[i], summedMatrices[i])
#                                               for i in range(len(clusters))])
        pool2.close()
        pool2.join()
        res2 = res2.get()
        for r in res2:
            resRawScore[r[0]] = r[1]
            resCoverage[r[0]] = r[2]
            resScorePos[r[0]] = r[3]
            resTimes[r[0]] += r[4]
            resAUCROC[r[0]] = r[5:]
    outDict = {}
    for c in clusters:
        clusterDir = '{}/'.format(c)
        if(not os.path.exists(clusterDir)):
            print('making dir')
            os.mkdir(clusterDir)
        else:
            print('not making dir')
        os.chdir(clusterDir)
        cStart = time.time()
        if(args['pdb']):
            plotAUC(resAUCROC[c][0], resAUCROC[c][1], resAUCROC[c][2],
                    args['query'][0], c, today, args['threshold'])
        writeOutClusterScoring(today, args['query'][0], query_sequence, c,
                               resScorePos[c], resRawScore[c], resCoverage[c],
                               rawClusters[c], wholeMIP_Matrix, resMats[c],
                               summedMatrices[c])
        heatmapPlot('Raw Score Heatmap K {}'.format(c), summedMatrices[c])
        surfacePlot('Raw Score Surface K {}'.format(c), summedMatrices[c])
#         heatmap_plot('Coverage Heatmap K {}'.format(c), resCoverage[c])
#         surface_plot('Coverage Surface K {}'.format(c), resCoverage[c])
        writeOutClusteringResults(today, args['query'][0],
                                  args['threshold'], c, resScorePos[c],
                                  resRawScore[c], resCoverage[c],
                                  query_sequence, sortedPDBDist,
                                  pdb_residue_list)
        cEnd = time.time()
        timeElapsed = cEnd - cStart
        resTimes[c] += timeElapsed
        try:
            outDict[c] = "\t{0}\t{1}\t{2}\n".format(c,
                                                    round(resAUCROC[c][2], 2),
                                                    round(resTimes[c], 2))
        except(TypeError):
            outDict[c] = "\t{0}\t{1}\t{2}\n".format(c, '-',
                                                    round(resTimes[c], 2))
        os.chdir('..')
    writeFinalResults(args['query'][0], today, sequence_order,
                      seq_length, args['threshold'], outDict)
    print "Generated results in", createFolder
    os.chdir(startDir)
    end = time.time()
    print('ET MIP took {} minutes to run!'.format((end - start) / 60.0))
