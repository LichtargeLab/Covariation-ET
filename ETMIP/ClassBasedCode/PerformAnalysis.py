'''
Created on Aug 17, 2017

@author: daniel
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
import numpy as np
import pylab as pl
import datetime
import argparse
import time
import csv
import sys
import os
from SeqAlignment import SeqAlignment
from PDBReference import PDBReference
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


def wholeAnalysis(alignment, saveFile=None):
    '''
    Whole Analysis

    Generates the MIP matrix.

    Parameters:
    -----------
    alignment: SeqAlignment
        A class containing the query sequence alignment in different formats,
        as well as summary values.
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
        # generate an MI matrix for each cluster
        miMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
        # Vector of 1 column
        MMI = np.zeros(alignment.seqLength)
        apcMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
        mipMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
        # Create matrix converting sequences of amino acids to sequences of integers
        # representing sequences of amino acids
#         alignment2Num = alignment2num(
#             alignment, keyList, alignment.seqLength, aaDict)
        # Generate MI matrix from alignment2Num matrix, the MMI matrix,
        # and overallMMI
#         usablePositions = alignment.determineUsablePositions(ratio=0.2)
        for i in range(alignment.seqLength):
            for j in range(i + 1, alignment.seqLength):
                #                 columnI = alignment2Num[:, i]
                #                 columnJ = alignment2Num[:, j]
                columnI = alignment.alignmentMatrix[:, i]
                columnJ = alignment.alignmentMatrix[:, j]
                currMIS = mutual_info_score(
                    columnI, columnJ, contingency=None)
                # AW: divides by individual entropies to normalize.
                miMatrix[i, j] = miMatrix[j, i] = currMIS
                overallMMI += currMIS
        MMI += np.sum(miMatrix, axis=1)
        MMI -= miMatrix[np.arange(alignment.seqLength),
                        np.arange(alignment.seqLength)]
        MMI /= (alignment.seqLength - 1)
        overallMMI = 2.0 * \
            (overallMMI / (alignment.seqLength - 1)) / alignment.seqLength
        # Calculating APC
        apcMatrix += np.outer(MMI, MMI)
        apcMatrix[np.arange(alignment.seqLength),
                  np.arange(alignment.seqLength)] = 0
        apcMatrix /= overallMMI
        # Defining MIP matrix
        mipMatrix += miMatrix - apcMatrix
        mipMatrix[np.arange(alignment.seqLength),
                  np.arange(alignment.seqLength)] = 0
        if(saveFile is not None):
            np.savez(saveFile, wholeMIP=mipMatrix)
    end = time.time()
    print('Whole analysis took {} min'.format((end - start) / 60.0))
    return mipMatrix


def poolInit(a, seqDists, qAlignment):
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
    qAlignment: SeqAlignment
        Object holding the alignment for the current analysis.
    '''
    global aaDict
    aaDict = a
    global X
    X = seqDists
    global poolAlignment
    poolAlignment = qAlignment


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
    start = time.time()
    print "Starting clustering: K={}".format(clus)
    clusterDict, clusterDet = aggClustering(clus, X, poolAlignment.seqOrder)
    rawScores = np.zeros((clus, poolAlignment.seqLength,
                          poolAlignment.seqLength))
    for c in clusterDet:
        newAlignment = poolAlignment.generateSubAlignment(clusterDict[c])
        newAlignment.alignment2num(aaDict)
        newAlignment.writeOutAlignment(
            fileName='AligmentForK{}_{}.fa'.format(clus, c))
        clusteredMIPMatrix = wholeAnalysis(newAlignment)
        rawScores[c] = clusteredMIPMatrix
    resMatrix = np.sum(rawScores, axis=0)
#     resMatrix = np.mean(rawScores, axis=0)
    end = time.time()
    timeElapsed = end - start
    print('ETMIP worker took {} min'.format(timeElapsed / 60.0))
    return clus, resMatrix, timeElapsed, rawScores


def poolInit2(c, qAlignment, qStructure, sPDBD):
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
    qAlignment: SeqAlignment
        Object containing the sequence alignment for this analysis.
    aStructure: PDBReference
        Object containing the PDB information for this analysis.
    sPDBD: list
        List of distances between PDB residues sorted by pairwise residue
        numbers.
    '''
    global cutoff
    cutoff = c
    global seqLen
    seqLen = qAlignment.seqLength
    global pdbResidueList
    if(qStructure is None):
        pdbResidueList = None
    else:
        pdbResidueList = qStructure.pdbResidueList
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


if __name__ == '__main__':
    ###########################################################################
    # BODY OF CODE ##
    ###########################################################################
    start = time.time()
    # Read input from the command line
    args = parseArguments()
    ###########################################################################
    # Set up global variables
    ###########################################################################
    today = str(datetime.date.today())
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aaDict = {aa_list[i]: i for i in range(len(aa_list))}
    ###########################################################################
    # Set up input variables
    ###########################################################################
    processes = args['processes']
    pCount = cpu_count()
    if(processes > pCount):
        processes = pCount
    ###########################################################################
    # Set up output location
    ###########################################################################
    startDir = os.getcwd()
    print startDir
    createFolder = (args['output'] + str(today) + "/" + args['query'][0])
    if not os.path.exists(createFolder):
        os.makedirs(createFolder)
        print "creating new folder"
    os.chdir(createFolder)
    ###########################################################################
    # Import alignment and perform initial analysis
    ###########################################################################
    print 'Starting ETMIP'
    # Import alignment information: this will be our alignment
    queryAlignment = SeqAlignment(fileName=startDir + '/' + args['alignment'][0],
                                  queryID=args['query'][0])
    queryAlignment.importAlignment(saveFile='alignment_dict.pkl')
    # Remove gaps from aligned query sequences
    queryAlignment.removeGaps(saveFile='ungapped_alignment.pkl')
    queryAlignment.alignment2num(aaDict)
    queryAlignment.writeOutAlignment(fileName='UngappedAlignment.fa')
    print('Query Sequence:')
    print(queryAlignment.querySequence)
    # I will get a corr_dict for method x for all residue pairs FOR ONE PROTEIN
    X = queryAlignment.distanceMatrix(saveFile='X')
    # Generate MIP Matrix
    wholeMIP_Matrix = wholeAnalysis(queryAlignment, 'wholeMIP')
    ###########################################################################
    # Set up for remaining analysis
    ###########################################################################
    if(args['pdb']):
        queryStructure = PDBReference(startDir + '/' + args['pdb'])
        queryStructure.importPDB(saveFile='pdbData.pkl')
        queryStructure.mapAlignmentToPDBSeq(queryAlignment.querySequence)
        sortedPDBDist = queryStructure.findDistance(
            queryAlignment.querySequence, saveFile='PDBdistances')
    else:
        queryStructure = None
    print('PDB Sequence')
    print(queryStructure.seq)
    ###########################################################################
    # Perform multiprocessing of clustering method
    ###########################################################################
    clusters = [2, 3, 5, 7, 10, 25]
    resMats = {}
    resTimes = {}
    rawClusters = {}
    if(processes == 1):
        poolInit(aaDict, X, queryAlignment)
        for c in clusters:
            clus, mat, times, rawCScores = etMIPWorker((c,))
            resMats[clus] = mat
            resTimes[clus] = times
            rawClusters[clus] = rawCScores
    else:
        pool = Pool(processes=processes, initializer=poolInit,
                    initargs=(aaDict, X, queryAlignment))
        res1 = pool.map_async(etMIPWorker, [(x,) for x in clusters])
        pool.close()
        pool.join()
        res1 = res1.get()
        for r in res1:
            resMats[r[0]] = r[1]
            resTimes[r[0]] = r[2]
            rawClusters[r[0]] = r[3]
    summedMatrices = {i: np.zeros(wholeMIP_Matrix.shape)
                      for i in clusters}
    for i in range(len(clusters)):
        currClus = clusters[i]
        summedMatrices[currClus] += wholeMIP_Matrix
        for j in [c for c in clusters if c <= currClus]:
            summedMatrices[currClus] += resMats[j]
        summedMatrices[currClus] /= (i + 2)
    resRawScore = {}
    resCoverage = {}
    resScorePos = {}
    resAUCROC = {}
    if(processes == 1):
        for clus in clusters:
            poolInit2(args['threshold'], queryAlignment,
                      queryStructure, sortedPDBDist)
            r = etMIPWorker2((clus, summedMatrices[clus]))
            resRawScore[r[0]] = r[1]
            resCoverage[r[0]] = r[2]
            resScorePos[r[0]] = r[3]
            resTimes[r[0]] += r[4]
            resAUCROC[r[0]] = r[5:]
    else:
        pool2 = Pool(processes=processes, initializer=poolInit2,
                     initargs=(args['threshold'], queryAlignment,
                               queryStructure, sortedPDBDist))
        res2 = pool2.map_async(etMIPWorker2, [(clus, summedMatrices[clus])
                                              for clus in clusters])
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
        writeOutClusterScoring(today, args['query'][0],
                               queryAlignment.querySequence, c,
                               resScorePos[c], resRawScore[c], resCoverage[c],
                               rawClusters[c], wholeMIP_Matrix, resMats[c],
                               summedMatrices[c])
        heatmapPlot('Raw Score Heatmap K {}'.format(c), summedMatrices[c])
        surfacePlot('Raw Score Surface K {}'.format(c), summedMatrices[c])
#         heatmapPlot('Coverage Heatmap K {}'.format(c), resCoverage[c])
#         surfacePlot('Coverage Surface K {}'.format(c), resCoverage[c])
        writeOutClusteringResults(today, args['query'][0],
                                  args['threshold'], c, resScorePos[c],
                                  resRawScore[c], resCoverage[c],
                                  queryAlignment.querySequence, sortedPDBDist,
                                  queryStructure.pdbResidueList)
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
    writeFinalResults(args['query'][0], today, queryAlignment.seqOrder,
                      queryAlignment.seqLength, args['threshold'], outDict)
    print "Generated results in", createFolder
    os.chdir(startDir)
    end = time.time()
    print('ET MIP took {} minutes to run!'.format((end - start) / 60.0))
