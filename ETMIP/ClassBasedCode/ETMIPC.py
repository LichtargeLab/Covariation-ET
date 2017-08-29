'''
Created on Aug 21, 2017

@author: dmkonecki
'''
import os
import csv
import sys
import numpy as np
import pylab as pl
from time import time
from multiprocessing import Pool
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mutual_info_score, auc, roc_curve
from IPython import embed
from nose.plugins import cover


class ETMIPC(object):
    '''
    classdocs
    '''

    def __init__(self, alignment, clusters, pdb, outputDir, processes):
        '''
        Constructor
        '''
        self.alignment = alignment
        self.clusters = clusters
        self.pdb = pdb
        self.outputDir = outputDir
        self.processes = processes
        self.wholeMIPMatrix = None
        self.wholeEvidenceMatrix = None
        self.resultTimes = {c: 0.0 for c in self.clusters}
        self.rawScores = {c: np.zeros((c, self.alignment.seqLength,
                                       self.alignment.seqLength))
                          for c in self.clusters}
        self.resultMatrices = {c: None for c in self.clusters}
        self.evidenceCounts = {c: np.zeros((c, self.alignment.seqLength,
                                            self.alignment.seqLength))
                               for c in self.clusters}
        self.summaryMatrices = {c: np.zeros((self.alignment.seqLength,
                                             self.alignment.seqLength))
                                for c in self.clusters}
        self.coverage = {c: np.zeros((self.alignment.seqLength,
                                      self.alignment.seqLength))
                         for c in self.clusters}
#         self.scorePositions = {}
        self.aucs = {}

    def determineWholeMIP(self):
        mipMatrix, evidenceCounts = wholeAnalysis(self.alignment,
                                                  saveFile='wholeMIP')
        self.wholeMIPMatrix = mipMatrix
        self.evidenceCounts = evidenceCounts

    def calculateClusteredMIPScores(self, aaDict, wCC='evidence_weighted'):
        # wCC options: 'sum', 'average', 'size_weighted', 'evidence_weighted'
        if(self.processes == 1):
            poolInit(aaDict, self.alignment, wCC, self.outputDir)
            res1 = []
            for c in self.clusters:
                clus, mat, times, rawCScores = etMIPWorker((c,))
                res1.append((clus, mat, times, rawCScores))
        else:
            pool = Pool(processes=self.processes, initializer=poolInit,
                        initargs=(aaDict, self.alignment, wCC, self.outputDir))
            res1 = pool.map_async(etMIPWorker, [(x,) for x in self.clusters])
            pool.close()
            pool.join()
            res1 = res1.get()
        for r in res1:
            self.resultMatrices[r[0]] = r[1]
            self.resultTimes[r[0]] = r[2]
            self.rawScores[r[0]] = r[3]

    def combineClusteringResults(self, combination='average'):
        for i in range(len(self.clusters)):
            currClus = self.clusters[i]
            self.summaryMatrices[currClus] += self.wholeMIPMatrix
            for j in [c for c in self.clusters if c <= currClus]:
                self.summaryMatrices[currClus] += self.resultMatrices[j]
            if(combination == 'average'):
                self.summaryMatrices[currClus] /= (i + 2)

    def computeCoverageAndAUC(self, sortedPDBDist, threshold):
        #         if(self.processes == 1):
        #
        #
        #
        #
        if(True):
            res2 = []
            for clus in self.clusters:
                poolInit2(threshold, self.alignment,
                          self.pdb, sortedPDBDist)
                r = etMIPWorker2((clus, self.summaryMatrices[clus]))
                res2.append(r)
        else:
            pool2 = Pool(processes=self.processes, initializer=poolInit2,
                         initargs=(threshold, self.alignment,
                                   self.pdb, sortedPDBDist))
            res2 = pool2.map_async(etMIPWorker2, [(clus, self.summaryMatrices[clus])
                                                  for clus in self.clusters])
            pool2.close()
            pool2.join()
            res2 = res2.get()
        for r in res2:
            self.coverage[r[0]] = r[1]
            self.resultTimes[r[0]] += r[2]
            self.aucs[r[0]] = r[3:]

    def plotAUC(self, qName, clus, today, cutoff):
        '''
        Plot AUC

        This function plots and saves the AUCROC.  The image will be stored in the
        eps format with dpi=1000 using a name specified by the query name, cutoff,
        clustering constant, and date.

        Parameters:
        qName: str
            Name of the query protein
        clus: int
            Number of clusters created
        today: date
            The days date
        cutoff: int
            The distance used for proximity cutoff in the PDB structure.
        '''
        start = time()
        pl.plot(self.aucs[clus][0], self.aucs[clus][1],
                label='(AUC = {0:.2f})'.format(self.aucs[clus][2]))
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
        end = time()
        print('Plotting the AUC plot took {} min'.format((end - start) / 60.0))

    def writeOutClusterScoring(self, today, qName, clus):
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
        start = time()
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
                             ['Cluster_Score', 'Summed_Score', 'ETMIp_Coverage'])
        for i in range(0, self.alignment.seqLength):
            for j in range(i + 1, self.alignment.seqLength):
                res1 = i + 1
                res2 = j + 1
                rowP1 = [res1, convertAA[self.alignment.querySequence[i]], res2,
                         convertAA[self.alignment.querySequence[j]],
                         round(self.wholeMIPMatrix[i, j], 2)]
                rowP2 = [round(self.rawScores[clus][c, i, j], 2)
                         for c in range(clus)]
                rowP3 = [round(self.resultMatrices[clus][i, j], 2),
                         round(self.summaryMatrices[clus][i, j], 2),
                         round(self.coverage[clus][i, j], 2)]
                etMIPWriter.writerow(rowP1 + rowP2 + rowP3)
        etMIPOutFile.close()
        end = time()
        print('Writing the ETMIP worker data to file took {} min'.format(
            (end - start) / 60.0))

    def writeOutClusteringResults(self, today, qName, cutoff, clus,
                                  sortedPDBDist):
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
        start = time()
        convertAA = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'B': 'ASX',
                     'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'Z': 'GLX', 'G': 'GLY',
                     'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET',
                     'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
                     'Y': 'TYR', 'V': 'VAL'}
        e = "{}_{}_{}.etmipCVG.clustered.txt".format(today, qName, clus)
        etMIPOutFile = open(e, "w+")
        etMIPWriter = csv.writer(etMIPOutFile, delimiter='\t')
        header = ['Pos1', '(AA1)', 'Pos2', '(AA2)', 'ETMIp_Score',
                  'ETMIp_Coverage', 'Residue_Dist', 'Within_Threshold',
                  'Cluster']
        etMIPWriter.writerow(header)
        counter = 0
        for i in range(0, self.alignment.seqLength):
            for j in range(i + 1, self.alignment.seqLength):
                if(sortedPDBDist is None):
                    res1 = i + 1
                    res2 = j + 1
                    r = '-'
                    dist = '-'
                else:
                    res1 = self.pdb.pdbResidueList[i]
                    res2 = self.pdb.pdbResidueList[j]
                    dist = round(sortedPDBDist[counter], 2)
                    if(sortedPDBDist[counter] <= cutoff):
                        r = 1
                    elif(np.isnan(sortedPDBDist[counter])):
                        r = '-'
                    else:
                        r = 0
                etMIPOutputLine = [res1, '({})'.format(
                    convertAA[self.alignment.querySequence[i]]),
                    res2, '({})'.format(
                    convertAA[self.alignment.querySequence[j]]),
                    round(self.summaryMatrices[clus][i, j], 2),
                    round(self.coverage[clus][i, j], 2), dist, r, clus]
                etMIPWriter.writerow(etMIPOutputLine)
                counter += 1
        etMIPOutFile.close()
        end = time()
        print('Writing the ETMIP worker data to file took {} min'.format(
            (end - start) / 60.0))
###############################################################################
#
###############################################################################


def wholeAnalysis(alignment, evidence=False, ratioCutOff=None, alterInput=False,
                  saveFile=None):
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
    ratioCutOff: float or None
        A cutoff for the percentage of sequences in the alignment which
        must have information (no gaps) for the MIP score at that position
        to be considered relevant.  Positions which fall below this
        threshold will be set to a MIP score of zero and given an evidence
        count of 0.
    alterInput: bool
        Whether or not to restrict the input to the mutual information
        computation to only those sequences which have gaps in neither of
        the considered positions.
    saveFile: str
        File path to a previously stored MIP matrix (.npz should be excluded as
        it will be added automatically).
    Returns:
    --------
    matrix
        Matrix of MIP scores which has dimensions seq_length by seq_length.
    '''
    start = time()
    if((saveFile is not None) and os.path.exists(saveFile + '.npz')):
        loadedData = np.load(saveFile + '.npz')
        mipMatrix = loadedData['wholeMIP']
        evidenceMatrix = loadedData['evidence']
    else:
        overallMMI = 0.0
        # generate an MI matrix for each cluster
        miMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
        evidenceMatrix = np.zeros(
            (alignment.seqLength, alignment.seqLength))
        # Vector of 1 column
        MMI = np.zeros(alignment.seqLength)
        apcMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
        mipMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
        # Generate MI matrix from alignment2Num matrix, the MMI matrix,
        # and overallMMI
        t1 = time()
        for i in range(alignment.seqLength):
            for j in range(i + 1, alignment.seqLength):
                if(evidence or alterInput):
                    #                     t11 = time()
                    colSubI, colSubJ, _pos, ev = alignment.identifyComparableSequences(
                        i, j)
#                     t22 = time()
#                     print('Time to identifyComp: {} min'.format(
#                         (t22 - t11) / 60.0))
                else:
                    ev = 0
                if(alterInput):
                    colI = colSubI
                    colJ = colSubJ
                else:
                    colI = alignment.alignmentMatrix[:, i]
                    colJ = alignment.alignmentMatrix[:, j]
                currMIS = mutual_info_score(colI, colJ, contingency=None)
#                 if(((alterInput) and (ev == 0)) or
#                    ((ratioCutOff is not None) and (r >= 0.8))):
#                     currMIS = 0
#                 else:
#                     try:
#                         currMIS = mutual_info_score(
#                             colI, colJ, contingency=None)
#                     except:
#                         print 'wholeAnalysis'
#                         embed()
#                         exit()
                # AW: divides by individual entropies to normalize.
                miMatrix[i, j] = miMatrix[j, i] = currMIS
                evidenceMatrix[i, j] = evidenceMatrix[j, i] = ev
                overallMMI += currMIS
        t2 = time()
        print('Time to in loop: {} min'.format((t2 - t1) / 60.0))
        t1 = time()
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
        t2 = time()
        print('Time to perform normalization: {} min'.format((t2 - t1) / 60.0))
        if(saveFile is not None):
            np.savez(saveFile, wholeMIP=mipMatrix, evidence=evidenceMatrix)
    end = time()
    print('Whole analysis took {} min'.format((end - start) / 60.0))
    return mipMatrix, evidenceMatrix


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
    start = time()
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
    end = time()
    print('Performing agglomerative clustering took {} min'.format(
        (end - start) / 60.0))
    return clusterDict, set(clusterList)


def poolInit(a, qAlignment, wCC, oDir):
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
    global poolAlignment
    poolAlignment = qAlignment
    global withinClusterCombi
    withinClusterCombi = wCC
    global outDir
    outDir = oDir


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
    start = time()
    resultDir = outDir + '/{}/'.format(clus)
    if(not os.path.exists(resultDir)):
        os.mkdir(resultDir)
    os.chdir(resultDir)
    print "Starting clustering: K={}".format(clus)
    clusterDict, clusterDet = aggClustering(clus, poolAlignment.distanceMatrix,
                                            poolAlignment.seqOrder)
    rawScores = np.zeros((clus, poolAlignment.seqLength,
                          poolAlignment.seqLength))
    evidenceCounts = np.zeros((clus, poolAlignment.seqLength,
                               poolAlignment.seqLength))
    clusterSizes = {}
    for c in clusterDet:
        newAlignment = poolAlignment.generateSubAlignment(clusterDict[c])
        clusterSizes[c] = newAlignment.size
        # Create matrix converting sequences of amino acids to sequences of
        # integers representing sequences of amino acids
        newAlignment.alignment2num(aaDict)
        newAlignment.writeOutAlignment(
            fileName='AligmentForK{}_{}.fa'.format(clus, c))
#         clusteredMIPMatrix = wholeAnalysis(newAlignment)
        if(withinClusterCombi == 'evidence_weighted'):
            clusteredMIPMatrix, evidenceMat = wholeAnalysis(newAlignment, True)
        else:
            clusteredMIPMatrix, evidenceMat = wholeAnalysis(
                newAlignment, False)
        rawScores[c] = clusteredMIPMatrix
        evidenceCounts[c] = evidenceMat
    # Additive clusters
    if(withinClusterCombi == 'sum'):
        resMatrix = np.sum(rawScores, axis=0)
    # Normal average over clusters
    elif(withinClusterCombi == 'average'):
        resMatrix = np.mean(rawScores, axis=0)
    # Weighted average over clusters based on cluster sizes
    elif(withinClusterCombi == 'size_weighted'):
        weighting = np.array([clusterSizes[c]
                              for c in sorted(clusterSizes.keys())])
        resMatrix = weighting[:, None, None] * rawScores
        resMatrix = np.sum(resMatrix, axis=0) / poolAlignment.size
    # Weighted average over clusters based on evidence counts at each pair
    elif(withinClusterCombi == 'evidence_weighted'):
        resMatrix = (np.sum(rawScores * evidenceCounts, axis=0) /
                     float(poolAlignment.size))
#                      np.sum(evidenceCounts, axis=0))
    else:
        print 'Combination method not yet implemented'
        raise NotImplementedError()
    resMatrix[np.isnan(resMatrix)] = 0.0
    os.chdir(outDir)
    end = time()
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
    start = time()
    coverage = np.zeros(summedMatrix.shape)
    testMat = np.triu(summedMatrix)
    mask = np.triu(np.ones(summedMatrix.shape), k=1)
    normalization = ((summedMatrix.shape[0]**2 - summedMatrix.shape[0]) / 2.0)
    for i in range(summedMatrix.shape[0]):
        for j in range(i + 1, summedMatrix.shape[0]):
            print('{} : {}'.format(i, j))
            boolMat = (testMat[i, j] >= testMat) * 1.0
            correctedMat = boolMat * mask
            computeCoverage2 = (((np.sum(correctedMat) - 1) * 100) /
                                normalization)
            coverage[i, j] = coverage[j, i] = computeCoverage2
    etmiplistCoverage = coverage[np.triu_indices(summedMatrix.shape[0], 1)]
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
    end = time()
    timeElapsed = end - start
    print('ETMIP worker 2 took {} min'.format(timeElapsed / 60.0))
    return (clus, coverage, timeElapsed, fpr, tpr, roc_auc)
