'''
Created on Mar 10, 2017

@author: Benu Atri
'''
from multiprocessing import Pool, cpu_count, Manager, Semaphore
from sklearn.metrics import roc_curve, auc, mutual_info_score
from sklearn.cluster import AgglomerativeClustering
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
import datetime
import argparse
import time
import sys
import re
import os


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
    parser.add_argument('--pdb', metavar='P', type=str, nargs=1,
                        help='The file path to the PDB structure associated with the provided alignment.')
    parser.add_argument('--query', metavar='Q', type=str, nargs=1,
                        help='The name of the protein being queried in this analysis.')
    parser.add_argument('--threshold', metavar='T', type=float, nargs=1,
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

    This method imports alignments into an existing dictionary.

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
    '''
    start = time.time()
    if((saveFile is not None) and (os.path.exists(saveFile))):
        alignment = pickle.load(open(saveFile, 'rb'))
    else:
        alignment = {}
        for line in faFile:
            if line.startswith(">"):
                key = line.rstrip()
                alignment[key] = ''
            else:
                alignment[key] += line.rstrip()
        faFile.close()
        if(saveFile is not None):
            pickle.dump(alignment, open(saveFile, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Importing alignment took {} min'.format((end - start) / 60.0))
    return alignment


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
                for i in range(1, len(queryGapInd) - 1):
                    newAlignment[key] += value[queryGapInd[i] + 1:
                                               queryGapInd[i + 1]]
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
    aaDict: dict
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
    # Generate distanceMatrix: Calculating Sequence Identity
    start = time.time()
    if((saveFiles is not None) and os.path.exists(saveFiles[0]) and
       os.path.exists(saveFiles[1])):
        valuematrix = np.load(saveFiles[0] + '.npz')['X']
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
    return valuematrix, keyList


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
    aaDict: list
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
        MMI = np.zeros(seq_length)
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
    importPDB

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
    if((saveFile is not None) and os.path.exists(saveFile)):
        residue3D, pdbResidueList, residuePos = pickle.load(
            open(saveFile, 'rb'))
    else:
        residue3D = {}
        pdbResidueList = []
        residuePos = {}
        prevRes = None
        pdbPattern = r'ATOM\s*(\d+)\s*(\w*)\s*([A-Z]{3})\s*([A-Z])\s*(\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*([A-Z])'
        for line in pdbFile:  # for a line in the pdb
            res = re.match(pdbPattern, line)
            if not res:
                continue
            resName = res.group(3)
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
        residue3D[prevRes] = np.vstack(residue3D[prevRes])
        # list of sorted residues - necessary for those where res1 is not 1
        pdbResidueList = sorted(pdbResidueList)
        pdbFile.close()
        if(saveFile is not None):
            pickle.dump((residue3D, pdbResidueList, residuePos),
                        open(saveFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Importing the PDB file took {} min'.format((end - start) / 60.0))
    return residue3D, pdbResidueList, residuePos


def findDistance(residue3D, pdbResidueList, saveFile=None):
    '''
    Find distance

    This code takes in an input of a pdb file and outputs a dictionary with the
    nearest atom distance between two residues.

    Parameters:
    -----------
    residue3D: dictionary
        A dictionary mapping a residue position from the PDB file to its three
        dimensional position.
    pdbResidueList: list
        A sorted list of residue numbers from the PDB file.
    saveFile: str
        File name and/or location of file containing a previously computed set
        of distance data for a PDB structure.
    Returns:
    list
        List of minimum distances between residues, sorted by the ordering of
        residues in pdbResidueList.
    --------
    '''
    start = time.time()
    if((saveFile is not None) and os.path.exists(saveFile)):
        sortedPDBDist = np.load(saveFile + '.npz')['pdbDists']
    else:
        sortedPDBDist = []
        # Loop over all residues in the pdb
        for i in range(len(pdbResidueList)):
            # Loop over residues to calculate distance between all residues i
            # and j
            for j in range(i + 1, len(pdbResidueList)):
                # Getting the 3d coordinates for every atom in each residue.
                # iterating over all pairs to find all distances
                key1 = pdbResidueList[i]
                key2 = pdbResidueList[j]
                # finding the minimum value from the distance array
                # Making dictionary of all min values indexed by the two residue
                # names
                res1 = residue3D[key2] - residue3D[key1][:, np.newaxis]
                norms = np.linalg.norm(res1, axis=2)
                sortedPDBDist.append(np.min(norms))
        sortedPDBDist = np.asarray(sortedPDBDist)
        if(saveFile is not None):
            np.savez(saveFile, pdbDists=sortedPDBDist)
    end = time.time()
    print('Computing the distance matrix based on the PDB file took {} min'.format(
        (end - start) / 60.0))
    # return PDBresidueList, ResidueDict, sortedPDBDist
    return sortedPDBDist


def poolInit(a, seqDists, sK, fAD):
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
    '''
    global aa_dict
    aaDict = a
    global X
    X = seqDists
    global sortedKeys
    sortedKeys = sK
    global fixedAlignment
    fixedAlignment = fAD


def etMIPWorker(inTup):
    '''
    ETMIP Worker

    Performs clustering and calculation ofcluster dependent sequence distances.
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
    resMatrix = None
    start = time.time()
    print "Starting clustering: K={}".format(clus)
    clusterDict, clusterDet = aggClustering(clus, X, sortedKeys)
    for c in clusterDet:
        newAlignment = {}
        for key in clusterDict[c]:
            newAlignment[key] = fixedAlignment[key]
        clusteredMIP_matrix = wholeAnalysis(newAlignment, aaDict)
        if(resMatrix is None):
            resMatrix = clusteredMIPMatrix
        else:
            resMatrix += clusteredMIPMatrix
    end = time.time()
    timeElapsed = end - start
    print('ETMIP worker took {} min'.format(time_elapsed / 60.0))
    return clus, resMatrix, timeElapsed


def poolInit2(c, PDBRL, sPDBD):
    '''
    poolInit2

    A function which initializes processes spawned in a worker pool performing
    the etMIPWorker2 function.  This provides a set of variables to all working
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
    global PDBresidueList
    pdbResidueList = PDBRL
    global sortedPDBDist
    sortedPDBDist = sPDBD


def etMIPWorker2(inTup):
    '''
    ETMIP Worker

    Performs clustering and calculation ofcluster dependent sequence distances.
    This function requires initialization of threads with poolInit, or setting
    of global variables as described in that function.

    Parameters:
    -----------
    inTup: tuple
        Tuple containing the number of clusters to form during agglomerative
        clustering and a matric which is the result of summing the original
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
    scorePositions = []
    etmipResScoreList = []
    for i in range(0, len(pdbResidueList)):
        for j in range(i + 1, len(pdbResidueList)):
            newkey1 = "{}_{}".format(
                pdbResidueList[i], pdbResidueList[j])
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
    if len(etmiplistCoverage) != len(sortedPDBDist):
        print "lengths do not match"
        sys.exit()
    y_true1 = ((sortedPDBDist <= cutoff) * 1)
    fpr, tpr, _thresholds = roc_curve(y_true1, etmiplistCoverage,
                                      pos_label=1)
    roc_auc = auc(fpr, tpr)
    return clus, etmiplistCoverage, scorePositions, fpr, tpr, roc_auc


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


def writeOutClusteringResults(today, qName, cutoff, clus, scorePositions,
                              etmiplistCoverage, sortedPDBDist, pdbResidueList,
                              residues3D):
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
    pdbResidueList: list
        Sorted list of residues from the PDB structure
    '''
    start = time.time()
    e = "{}_{}_{}.etmipCVG.clustered.txt".format(today, qName, clus)
    etMIPOutFile = open(e, "w+")
    counter = 0
    for i in range(0, len(pdbResidueList)):
        for j in range(i + 1, len(pdbResidueList)):
            res1 = pdbResidueList[i]
            res2 = pdbResidueList[j]
            key = '{}_{}'.format(res1, res2)
            if sortedPDBDist[counter] <= cutoff:
                r = 1
            else:
                r = 0
            ind = scorePositions.index(key)
            etMIPOutputLine = '{} ({}) {} ({}) {} {} {} {}\n'.format(
                res1, residues3D[res1], res2, residues3D[res2],
                round(etmiplistCoverage[ind], 2),
                round(sortedPDBDist[counter], 2), r, clus)
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
    seqLength: int
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


####--------------------------------------------------------#####
    ### BODY OF CODE ##
####--------------------------------------------------------#####
if __name__ == '__main__':
    start = time.time()
    args = parseArguments()
    ###########################################################################
    # Set up global variables
    ###########################################################################
    today = str(datetime.date.today())
    neighbor_list = []
    gap_list = ["-", ".", "_"]
    aa_list = []
    # comment out for actual dataset
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    aa_gap_list = aa_list + gap_list
    i_j_list = []
    # {seq1_seq2} = distancescore
    seq12_distscore_dict = {}
    key = ''
    temp_aa = ''
    ###########################################################################
    # Set up input variables
    ###########################################################################
    files = open(args['alignment'][0], 'rb')
    pdbfilename = open(args['pdb'][0], 'rb')
    try:
        processes = args['processes']
        pCount = cpu_count()
        if(processes > pCount):
            processes = pCount
    except:
        processes = 1
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
    alignment_dict = importAlignment(files, 'alignment_dict.pkl')
    # Remove gaps from aligned query sequences
    query_name, fixed_alignment_dict = removeGaps(alignment_dict,
                                                  args['query'][0],
                                                  'ungapped_alignment.pkl')
    # I will get a corr_dict for method x for all residue pairs FOR ONE PROTEIN
    X, sequence_order = distanceMatrix(fixed_alignment_dict, aa_dict,
                                       ('X', 'seq_order.pkl'))
    # Generate MIP Matrix
    wholeMIP_Matrix = wholeAnalysis(fixed_alignment_dict, aa_dict, 'wholeMIP')
    ###########################################################################
    # Set up for remaining analysis
    ###########################################################################
    seq_length = len(fixed_alignment_dict[fixed_alignment_dict.keys()[0]])
    # pdbData = importPDB(pdbfilename, 'pdbData.pkl')
    residuedictionary, PDBresidueList, ResidueDict = importPDB(
        pdbfilename, 'pdbData.pkl')
    # PDBresidueList, ResiduesDict, sortedPDBDist = findDistance(
    sortedPDBDist = findDistance(residuedictionary, PDBresidueList,
                                 'PDBdistances')
    #   pdbData, ('PDBdist.pkl', 'PDBdistances'))
    ###########################################################################
    # Perform multiprocessing of clustering method
    ###########################################################################
    clusters = [2, 3, 5, 7, 10, 25]
    resMats = {}
    resTimes = {}
    if(processes == 1):
        poolInit(aa_dict, X, sequence_order, fixed_alignment_dict)
        for c in clusters:
            clus, mat, times = etMIPWorker((c,))
            resMats[clus] = mat
            resTimes[clus] = times
    else:
        pool = Pool(processes=processes, initializer=poolInit, initargs=(
            aa_dict, X, sequence_order, fixed_alignment_dict))
        res1 = pool.map_async(etMIPWorker, [(x,) for x in clusters])
        pool.close()
        pool.join()
        es1 = res1.get()
        for r in res1:
            resMats[r[0]] = r[1]
            resTimes[r[0]] = r[2]
    summedMatrices = [np.zeros(wholeMIP_Matrix.shape)] * len(clusters)
    for i in range(len(clusters)):
        summedMatrices[i] = summedMatrices[i] + wholeMIP_Matrix
        for j in range(0, i):
            summedMatrices[i] += resMats[clusters[j]]
    resCoverage = {}
    resScorePos = {}
    resAUCROC = {}
    if(processes == 1):
        for i in range(len(clusters)):
            poolInit2(args['threshold'][0], PDBresidueList, sortedPDBDist)
            clus, coverage, score, auc, fpr, tpr = etMIPWorker2(clusters[i],
                                                                summedMatrices[i])
            resCoverage[clus] = coverage
            resScorePos[clsu] = score
            resAUCROC[clus] = (auc, fpr, tpr)
    else:
        pool2 = Pool(processes=processes, initializer=poolInit2, initargs=(
            args['threshold'][0], PDBresidueList, sortedPDBDist))
        res2 = pool2.map_async(etMIPWorker2, [(clusters[i], summedMatrices[i])
                                              for i in range(len(clusters))])
        pool2.close()
        pool2.join()
        res2 = res2.get()
        for r in res2:
            resCoverage[r[0]] = r[1]
            resScorePos[r[0]] = r[2]
            resAUCROC[r[0]] = r[3:]
    outDict = {}
    for c in clusters:
        plotAUC(resAUCROC[c][0], resAUCROC[c][1], resAUCROC[c][2],
                args['query'][0], c, today, args['threshold'][0])
        writeOutClusteringResults(today, args['query'][0], args['threshold'][0],
                                  c, resScorePos[c], resCoverage[c],
                                  sortedPDBDist, PDBresidueList, ResidueDict)
        outDict[c] = "\t{0}\t{1}\t{2}\n".format(c, round(resAUCROC[c][2], 2),
                                                round(resTimes[c], 2))
    writeFinalResults(args['query'][0], today, sequence_order,
                      seq_length, args['threshold'][0], outDict)
    print "Generated results in", createFolder
    os.chdir(startDir)
    end = time.time()
    print('ET MIP took {} minutes to run!'.format((end - start) / 60.0))
