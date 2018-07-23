'''
Created on Aug 21, 2017

@author: dmkonecki
'''
import os
import csv
import sys
import Queue
import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pylab as pl
from seaborn import heatmap
from multiprocessing import Manager, Pool
from sklearn.metrics import mutual_info_score, auc, roc_curve
from IPython import embed


class ETMIPC(object):
    '''
    classdocs
    '''

    def __init__(self, alignment, clusters, pdb, outputDir, processes,
                 lowMemoryMode=False):
        '''
        Constructor

        Initiates an instance of the ETMIPC class which stores the
        following data:

        alignment : SeqAlignment
            The SeqAlignment object containing relevant information for this
            ETMIPC analysis.
        clusters : list
            The k's for which to create different clusterings.
        subAlignments : dict
            A dictionary mapping a clustering constant (k) to another dictionary
            which maps a cluster label (0 to k-1) to a SeqAlignment object
            containing only the sequences for that specific cluster.
        pdb : PDBReference
            The PDBReference object containing relevant information for this
            ETMIPC analysis.
        outputDir : str
            Directory name or path to directory where results from this analysis
            should be stored.
        processes : int
            The number of processes to spawn for more intense computations
            performed during this analysis.  If the number is higher than the
            number of jobs required to quickly perform this analysis the number
            of jobs is used as the number of processes.  If the number of
            processes is higher than the number of processors available to the
            program, the number of processors available is used instead.
        wholeMIPMatrix : np.array
            Matrix scoring the coupling between all positions in the query
            sequence, as computed over all sequences in the input alignment.
        wholeEvidenceMatrix : np.array
            Matrix containing the number of sequences which are not gaps in
            either position used for scoring the wholeMIPMatrix.
        resultTimes : dict
            Dictionary mapping k constant to the amount of time it took to
            perform analysis for that k constant.
        rawScores : dict
            This dictionary maps clustering constant to an k x m x n matrix.
            This matrix has coupling scores for all positions in the query
            sequences for each of the clusters created by hierarchical
            clustering.
        resultMatrices : dict
            This dictionary maps clustering constants to a matrix scoring the
            coupling between all residues in the query sequence over all of the
            clusters created at that constant.
        evidenceCounts : dict
            This dictionary maps clustering constants to a matrix which has
            counts for the number of sequences which are not gaps in
            either position used for scoring at that position.
        summaryMatrices : dict
            This dictionary maps clustering constants to a matrix which combines
            the scores from the wholeMIPMatrix, all lower clustering constants,
            and this clustering constant.
        coverage : dict
            This dictionary maps clustering constants to a matrix of normalized
            coupling scores between 0 and 100, computed from the
            summaryMatrices.
        aucs : dict
            This dictionary maps clustering constants to a tuple containing the
            auc, tpr, and fpr for comparing the coupled scores to the PDB
            reference (if provided) at a specified distance threshold.
        lowMemoryMode: bool
            This boolean specifies whether or not to run in low memory mode. If
            True is specified a majority of the class variables are set to None
            and the data is saved to disk at run time and loaded when needed for
            downstream analysis. The intermediate files generated in this way
            can be removed using clearIntermediateFiles. The default value is
            False, in which case all variables are kept in memory.
        '''
        self.alignment = alignment
        self.clusters = clusters
        self.subAlignments = {c: {} for c in self.clusters}
        self.pdb = pdb
        self.outputDir = outputDir
        self.processes = processes
        self.wholeMIPMatrix = None
        self.wholeEvidenceMatrix = None
        self.resultTimes = {c: 0.0 for c in self.clusters}
        if(lowMemoryMode):
            self.rawScores = None
            self.evidenceCounts = None
            self.resultMatrices = None
            self.summaryMatrices = None
            self.coverage = None
        else:
            self.rawScores = {c: np.zeros((c, self.alignment.seqLength,
                                           self.alignment.seqLength))
                              for c in self.clusters}
            self.evidenceCounts = {c: np.zeros((c, self.alignment.seqLength,
                                                self.alignment.seqLength))
                                   for c in self.clusters}
            self.resultMatrices = {c: None for c in self.clusters}
            self.summaryMatrices = {c: np.zeros((self.alignment.seqLength,
                                                 self.alignment.seqLength))
                                    for c in self.clusters}
            self.coverage = {c: np.zeros((self.alignment.seqLength,
                                          self.alignment.seqLength))
                             for c in self.clusters}
        self.aucs = {}
        self.lowMem = lowMemoryMode

    def determineWholeMIP(self, evidence):
        '''
        determineWholeMIP

        Paramters:
        evidence : bool
            Whether or not to normalize using the evidence using the evidence
            counts computed while performing the coupling scoring.

        This method performs the wholeAnalysis method on all sequences in the
        sequence alignment. This method updates the wholeMIPMatrix and
        wholeEvidenceMatrix class variables.
        '''
        mipMatrix, evidenceCounts = wholeAnalysis(self.alignment, evidence,
                                                  saveFile='wholeMIP')
        self.wholeMIPMatrix = mipMatrix
        self.wholeEvidenceMatrix = evidenceCounts

    def calculateClusteredMIPScores(self, aaDict, wCC):
        '''
        Calculate Clustered MIP Scores

        This method calculates the coupling scores for subsets of sequences
        from the alignment as determined by hierarchical clustering on the
        distance matrix between sequences of the alignment. This method updates
        the resultMatrices, resultTimes, and rawScores class variables.

        Parameters:
        -----------
        aaDict : dict
            A dictionary mapping amino acids to numerical representations.
        wCC : str
            Method by which to combine individual matrices from one round of
            clustering. The options supported now are: sum, average,
            size_weighted, evidence_weighted, and evidence_vs_size.
        '''
        cetmipManager = Manager()
        kQueue = cetmipManager.Queue()
        subAlignmentQueue = cetmipManager.Queue()
        resQueue = cetmipManager.Queue()
        alignmentLock = cetmipManager.Lock()
        for k in self.clusters:
            kQueue.put(k)
        if(self.processes == 1):
            poolInit1(aaDict, wCC, self.alignment, self.outputDir,
                      alignmentLock, kQueue, subAlignmentQueue, resQueue,
                      self.lowMem)
            clusterSizes, subAlignments, clusterTimes = etMIPWorker1((1, 1))
            print subAlignmentQueue.qsize()
            self.resultTimes = clusterTimes
            self.subAlignments = subAlignments
        else:
            pool = Pool(processes=self.processes, initializer=poolInit1,
                        initargs=(aaDict, wCC, self.alignment, self.outputDir,
                                  alignmentLock, kQueue, subAlignmentQueue,
                                  resQueue, self.lowMem))
            poolRes = pool.map_async(etMIPWorker1,
                                     [(x + 1, self.processes)
                                      for x in range(self.processes)])
            pool.close()
            pool.join()
            clusterDicts = poolRes.get()
            clusterSizes = {}
            for cD in clusterDicts:
                for c in cD[0]:
                    if(c not in clusterSizes):
                        clusterSizes[c] = {}
                    for s in cD[0][c]:
                        clusterSizes[c][s] = cD[0][c][s]
                for c in cD[1]:
                    for s in cD[1][c]:
                        self.subAlignments[c][s] = cD[1][c][s]
                for c in cD[2]:
                    self.resultTimes[c] += cD[2][c]
        # Retrieve results
        while((not self.lowMem) and (not resQueue.empty())):
            r = resQueue.get_nowait()
            self.rawScores[r[0]][r[1]] = r[2]
            self.evidenceCounts[r[0]][r[1]] = r[3]
#             self.resultTimes[r[0]] += r[4]
        # Combine results
        for c in self.clusters:
            if(self.lowMem):
                currRawScores, currEvidence = loadRawScoreMatrix(
                    self.alignment.seqLength, c, self.outputDir)
            else:
                currRawScores = self.rawScores[c]
                currEvidence = self.evidenceCounts[c]
            start = time()
            # Additive clusters
            if(wCC == 'sum'):
                resMatrix = np.sum(currRawScores, axis=0)
            # Normal average over clusters
            elif(wCC == 'average'):
                resMatrix = np.mean(currRawScores, axis=0)
            # Weighted average over clusters based on cluster sizes
            elif(wCC == 'size_weighted'):
                weighting = np.array([clusterSizes[c][s]
                                      for s in sorted(clusterSizes[c].keys())])
                resMatrix = weighting[:, None, None] * currRawScores
                resMatrix = np.sum(resMatrix, axis=0) / self.alignment.size
            # Weighted average over clusters based on evidence counts at each
            # pair vs. the number of sequences with evidence for that pairing.
            elif(wCC == 'evidence_weighted'):
                resMatrix = (np.sum(currRawScores * currEvidence,
                                    axis=0) / np.sum(currEvidence, axis=0))
            # Weighted average over clusters based on evidence counts at each
            # pair vs. the entire size of the alignment.
            elif(wCC == 'evidence_vs_size'):
                resMatrix = (np.sum(currRawScores * currEvidence,
                                    axis=0) / float(self.alignment.size))
            else:
                print 'Combination method not yet implemented'
                raise NotImplementedError()
            resMatrix[np.isnan(resMatrix)] = 0.0
            if(self.lowMem):
                saveSingleMatrix('Result', c, resMatrix, self.outputDir)
            else:
                self.resultMatrices[c] = resMatrix
            end = time()
            self.resultTimes[c] += end - start

    def combineClusteringResults(self, combination):
        '''
        Combine Clustering Result

        This method combines data from wholeMIPMatrix and resultMatrices to
        populate the summaryMatrices.  The combination occurs by simple addition
        but if average is specified it is normalized by the number of elements
        added.

        Parameters:
        -----------
        combination: str
            Method by which to combine scores across clustering constants. By
            default only a sum is performed, the option average is also
            supported.
        '''
        start = time()
        for i in range(len(self.clusters)):
            currClus = self.clusters[i]
            if(self.lowMem):
                currSummary = np.zeros((self.alignment.seqLength,
                                        self.alignment.seqLength))
            else:
                currSummary = self.summaryMatrices[currClus]
            currSummary += self.wholeMIPMatrix
            for j in [c for c in self.clusters if c <= currClus]:
                if(self.lowMem):
                    currSummary += loadSingleMatrix('Result',
                                                    j, self.outputDir)
                else:
                    currSummary += self.resultMatrices[j]
            if(combination == 'average'):
                currSummary /= (i + 2)
            if(self.lowMem):
                saveSingleMatrix('Summary', currClus, currSummary,
                                 self.outputDir)
        end = time()
        print('Combining data across clusters took {} min'.format(
            (end - start) / 60.0))

    def computeCoverageAndAUC(self, threshold):
        '''
        Compute Coverage And AUC

        This method computes the coverage/normalized coupling scores between
        residues in the query sequence as well as the AUC summary values
        determined when comparing the coupling score to the distance between
        residues in the PDB file. This method updates the coverage, resultTimes,
        and aucs variables.

        Parameters:
        -----------
        threshold : float
            Distance in Angstroms between residues considered as a preliminary
            positive set of coupled residues based on spatial positions in
            the PDB file if provided.
        '''
        start = time()
        if(self.processes == 1):
            res2 = []
            for clus in self.clusters:
                poolInit2(threshold, self.alignment, self.pdb, self.outputDir)
                r = etMIPWorker2((clus, self.summaryMatrices))
                res2.append(r)
        else:
            pool2 = Pool(processes=self.processes, initializer=poolInit2,
                         initargs=(threshold, self.alignment, self.pdb,
                                   self.outputDir))
            res2 = pool2.map_async(etMIPWorker2,
                                   [(clus, self.summaryMatrices)
                                    for clus in self.clusters])
            pool2.close()
            pool2.join()
            res2 = res2.get()
        for r in res2:
            if(not self.lowMem):
                self.coverage[r[0]] = r[1]
            self.resultTimes[r[0]] += r[2]
            self.aucs[r[0]] = r[3:]
        end = time()
        print('Computing coverage and AUC took {} min'.format((end - start) / 60.0))

    def produceFinalFigures(self, today, cutOff, verbosity):
        '''
        Produce Final Figures

        This method writes out clustering scores and additional results, as well
        as plotting heatmaps and surface plots of the coupling data for the
        query sequence. This method updates the resultTimes class variable.

        Parameters:
        -----------
        today : str
            The current date which will be used for identifying the proper
            directory to store files in.
        cutOff : float
            Distance in Angstroms between residues considered as a preliminary
            positive set of coupled residues based on spatial positions in
            the PDB file if provided.
        verbosity : int
            How many figures to produce.1 = ROC Curves, ETMIP Coverage file,
            and final AUC and Timing file. 2 = files with all scores at each
            clustering. 3 = sub-alignment files and plots. 4 = surface plots
            and heatmaps of ETMIP raw and coverage scores.'
        '''
        begin = time()
        qName = self.alignment.queryID.split('_')[1]
        poolManager = Manager()
        clusterQueue = poolManager.Queue()
        outputQueue = poolManager.Queue()
        for c in self.clusters:
            clusterQueue.put_nowait(c)
        if(self.processes == 1):
            poolInit3(clusterQueue, outputQueue, qName, today, cutOff,
                      verbosity, self.wholeMIPMatrix, self.rawScores,
                      self.resultMatrices, self.coverage, self.summaryMatrices,
                      self.subAlignments, self.alignment, self.aucs, self.pdb,
                      self.outputDir)
            etMIPWorker3((1, 1))
        else:
            pool = Pool(processes=self.processes, initializer=poolInit3,
                        initargs=(clusterQueue, outputQueue, qName, today,
                                  cutOff, verbosity, self.wholeMIPMatrix,
                                  self.rawScores,  self.resultMatrices,
                                  self.coverage, self.summaryMatrices,
                                  self.subAlignments, self.alignment,
                                  self.aucs, self.pdb, self.outputDir))
            res = pool.map_async(etMIPWorker3, [(x + 1, self.processes)
                                                for x in range(self.processes)])
            pool.close()
            pool.join()
            for times in res.get():
                for c in times:
                    self.resultTimes[c] += times[c]
        finish = time()
        print('Producing final figures took {} min'.format(
            (finish - begin) / 60.0))

    def writeFinalResults(self, today, cutoff):
        '''
        Write final results

        This method writes the final results to file for an analysis.  In this case
        that consists of the cluster numbers, the resulting AUCs, and the time
        spent in processing.

        Parameters:
        -----------
        today: str
            The current date in string format.
        cutoff: float
            The distance threshold for interaction between two residues in a
            protein structure.
        '''
        start = time()
        qName = self.alignment.queryID.split('_')[1]
        o = '{}_{}etmipAUC_results.txt'.format(qName, today)
        outfile = open(o, 'w+')
        outfile.write(
            "Protein/id: {} Alignment Size: {} Length of protein: {} Cutoff: {}\n".format(
                qName, self.alignment.size, self.alignment.seqLength, cutoff))
        outfile.write("#OfClusters\tAUC\tRunTime\n")
        for c in self.clusters:
            if(self.pdb):
                outfile.write("\t{0}\t{1}\t{2}\n".format(
                    c, round(self.aucs[c][2], 4),
                    round(self.resultTimes[c], 4)))
            else:
                outfile.write("\t{0}\t{1}\t{2}\n".format(
                    c, '-',
                    round(self.resultTimes[c], 4)))
        end = time()
        print('Writing final results took {} min'.format((end - start) / 60.0))

    def clearIntermediateFiles(self):
        '''
        Clear Intermediate Files

        This method is intended to be used only if the ETMIPC lowMem variable is
        set to True. If this is the case and the complete analysis has been
        performed then this function will remove all intermediate file generated
        during execution.
        '''
        for k in self.clusters:
            resPath = os.path.join(self.outputDir, str(k),
                                   'K{}_Result.npz'.format(k))
            os.remove(resPath)
            summaryPath = os.path.join(self.outputDir, str(k),
                                       'K{}_Summary.npz'.format(k))
            os.remove(summaryPath)
            coveragePath = os.path.join(self.outputDir, str(k),
                                        'K{}_Coverage.npz'.format(k))
            os.remove(coveragePath)
            for sub in range(k):
                currPath = os.path.join(self.outputDir, str(k),
                                        'K{}_Sub{}.npz'.format(k, sub))
                os.remove(currPath)
###############################################################################
#
###############################################################################


def saveRawScoreMatrix(k, sub, mat, evidence, outDir):
    '''
    Save Raw Score Matrix

    This function can be used to save the rawScore and evidenceCounts matrices
    which need to be saved to disk in order to reduce the memory footprint when
    the ETMIPC variable lowMem is set to true.

    Parameters:
    -----------
    k : int
        An integer specifying which clustering constant to load data for.
    sub : int
        An integer specifying the the cluster for which to save data (expected
        values are in range(0, k)).
    mat : np.array
        The array for the rawScore data to save for the specified cluster.
    evidence : np.array
        The array for the evidenceCounts data to save for the specified cluster.
    outDir : str
        The top level directory where data are being stored, where directories
        for each k can be found.
    '''
    cOutDir = os.path.join(outDir, str(k))
    if(not os.path.exists(cOutDir)):
        os.mkdir(cOutDir)
    np.savez(os.path.join(cOutDir, 'K{}_Sub{}.npz'.format(k, sub)), mat=mat,
             evidence=evidence)


def loadRawScoreMatrix(seqLen, k, outDir):
    '''
    Load Raw Score Matrix

    This function can be used to load the rawScore and evidenceCounts matrices
    which need to be saved to disk in order to reduce the memory footprint when
    the ETMIPC variable lowMem is set to true.

    Parameters:
    -----------
    k : int
        An integer specifying which clustering constant to load data for.
    sub : int
        An integer specifying the the cluster for which to save data (expected
        values are in range(0, k)).
    outDir : str
        The top level directory where data are being stored, where directories
        for each k can be found.
    Returns:
    --------
    np.array
        The array for the rawScore data to save for the specified cluster.
    np.array
        The array for the evidenceCounts data to save for the specified cluster.
    '''
    mat = np.zeros((k, seqLen, seqLen))
    evidence = np.zeros((k, seqLen, seqLen))
    for sub in range(k):
        loadPath = os.path.join(outDir, str(k), 'K{}_Sub{}.npz'.format(k, sub))
        data = np.load(loadPath)
        cMat = data['mat']
        eMat = data['evidence']
        mat[sub] = cMat
        evidence[sub] = eMat
    return mat, evidence


def saveSingleMatrix(name, k, mat, outDir):
    '''
    Save Single Matrix

    This function can be used to save any of the several matrices which need to
    be saved to disk in order to reduce the memory footprint when the ETMIPC
    variable lowMem is set to true.

    Parameters:
    -----------
    name : str
        A string specifying what kind of data is being stored, expected values
        include:
            Result
            Summary
            Coverage
    k : int
        An integer specifying which clustering constant to load data for.
    mat : np.array
        The array for the given type of data to save for the specified cluster.
    outDir : str
        The top level directory where data are being stored, where directories
        for each k can be found.
    '''
    cOutDir = os.path.join(outDir, str(k))
    if(not os.path.exists(cOutDir)):
        os.mkdir(cOutDir)
    np.savez(os.path.join(cOutDir, 'K{}_{}.npz'.format(k, name)), mat=mat)


def loadSingleMatrix(name, k, outDir):
    '''
    Load Single Matrix

    This function can be used to load any of the several matrices which are
    saved to disk in order to reduce the memory footprint when the ETMIPC
    variable lowMem is set to true.

    Parameters:
    -----------
    name : str
        A string specifying what kind of data is being stored, expected values
        include:
            Result
            Summary
            Coverage
    k : int
        An integer specifying which clustering constant to load data for.
    outDir : str
        The top level directory where data are being stored, where directories
        for each k can be found.
    Returns:
    --------
    np.array
        The array for the given type of data loaded for the specified cluster.
    '''
    data = np.load(os.path.join(outDir, str(k), 'K{}_{}.npz'.format(k, name)))
    return data['mat']


def wholeAnalysis(alignment, evidence, saveFile=None):
    '''
    Whole Analysis

    Generates the MIP matrix.

    Parameters:
    -----------
    alignment: SeqAlignment
        A class containing the query sequence alignment in different formats,
        as well as summary values.
    evidence : bool
        Whether or not to normalize using the evidence using the evidence
        counts computed while performing the coupling scoring.
    saveFile: str
        File path to a previously stored MIP matrix (.npz should be excluded as
        it will be added automatically).
    Returns:
    --------
    matrix
        Matrix of MIP scores which has dimensions seq_length by seq_length.
    matrix
        Matrix containing the number of sequences which are not gaps in either
        position used for scoring the wholeMIPMatrix.
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
        for i in range(alignment.seqLength):
            for j in range(i + 1, alignment.seqLength):
                if(evidence):
                    _I, _J, _pos, ev = alignment.identifyComparableSequences(
                        i, j)
                else:
                    ev = 0
                colI = alignment.alignmentMatrix[:, i]
                colJ = alignment.alignmentMatrix[:, j]
                try:
                    currMIS = mutual_info_score(colI, colJ, contingency=None)
                except:
                    print colI
                    print colJ
                    exit()
                # AW: divides by individual entropies to normalize.
                miMatrix[i, j] = miMatrix[j, i] = currMIS
                evidenceMatrix[i, j] = evidenceMatrix[j, i] = ev
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
            np.savez(saveFile, wholeMIP=mipMatrix, evidence=evidenceMatrix)
    end = time()
    print('Whole analysis took {} min'.format((end - start) / 60.0))
    return mipMatrix, evidenceMatrix


def plotAUC(qName, clus, today, cutoff, aucs, outputDir=None):
    '''
    Plot AUC

    This function plots and saves the AUCROC.  The image will be stored in
    the eps format with dpi=1000 using a name specified by the query name,
    cutoff, clustering constant, and date.

    Parameters:
    -----------
    qName: str
        Name of the query protein
    clus: int
        Number of clusters created
    today: date
        The days date
    cutoff: int
        The distance used for proximity cutoff in the PDB structure.
    aucs : dictionary
        AUC values stored in the ETMIPC class, used to identify the specific
        values for the specified clustering constant (clus).
    outputDir : str
        The full path to where the AUC plot image should be stored. If None
        (default) the plot will be stored in the current working directory.
    '''
    start = time()
    pl.plot(aucs[clus][0], aucs[clus][1],
            label='(AUC = {0:.2f})'.format(aucs[clus][2]))
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    title = 'Ability to predict positive contacts in {}, Cluster = {}'.format(
        qName, clus)
    pl.title(title)
    pl.legend(loc="lower right")
    imagename = '{0}{1}A_C{2}_{3}roc.eps'.format(
        qName, cutoff, clus, today)
    if(outputDir):
        imagename = outputDir + imagename
    pl.savefig(imagename, format='eps', dpi=1000, fontsize=8)
    pl.close()
    end = time()
    print('Plotting the AUC plot took {} min'.format((end - start) / 60.0))


def heatmapPlot(name, relData, cluster, outputDir=None):
    '''
    Heatmap Plot

    This method creates a heatmap using the Seaborn plotting package. The
    data used can come from the summaryMatrices or coverage data.

    Parameters:
    -----------
    name : str
        Name used as the title of the plot and the filename for the saved
        figure.
    relData : dict
        A dictionary of integers (k) mapped to matrices (scores). This input
        should either be the coverage or summaryMatrices from the ETMIPC class.
    cluster : int
        The clustering constant for which to create a heatmap.
    outputDir : str
        The full path to where the heatmap plot image should be stored. If None
        (default) the plot will be stored in the current working directory.
    '''
    start = time()
    if(relData):
        dataMat = relData[cluster]
    else:
        if('Coverage' in name):
            dataMat = loadSingleMatrix('Coverage', cluster, outputDir)
        else:
            dataMat = loadSingleMatrix('Summary', cluster, outputDir)
    dmMax = np.max(dataMat)
    dmMin = np.min(dataMat)
    plotMax = max([dmMax, abs(dmMin)])
    heatmap(data=dataMat, cmap='jet', center=0.0, vmin=-1 * plotMax,
            vmax=plotMax, cbar=True, square=True)
    plt.title(name)
    imageName = name.replace(' ', '_') + '.pdf'
    if(outputDir):
        imageName = os.path.join(outputDir, str(cluster), imageName)
    plt.savefig(imageName)
    plt.clf()
    end = time()
    print('Plotting ETMIp-C heatmap took {} min'.format((end - start) / 60.0))


def surfacePlot(name, relData, cluster, outputDir=None):
    '''
    Surface Plot

    This method creates a surface plot using the matplotlib plotting
    package. The data used can come from the summaryMatrices or coverage
    data.

    Parameters:
    -----------
    name : str
        Name used as the title of the plot and the filename for the saved
        figure.
    relData : dict
        A dictionary of integers (k) mapped to matrices (scores). This input
        should either be the coverage or summaryMatrices from the ETMIPC class.
    cluster : int
        The clustering constant for which to create a heatmap.
    outputDir : str
        The full path to where the AUC plot image should be stored. If None
        (default) the plot will be stored in the current working directory.
    '''
    start = time()
    if(relData):
        dataMat = relData[cluster]
    else:
        if('Coverage' in name):
            dataMat = loadSingleMatrix('Coverage', cluster, outputDir)
        else:
            dataMat = loadSingleMatrix('Summary', cluster, outputDir)
    dmMax = np.max(dataMat)
    dmMin = np.min(dataMat)
    plotMax = max([dmMax, abs(dmMin)])
    X = Y = np.arange(max(dataMat.shape))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, dataMat, cmap='jet', linewidth=0,
                           antialiased=False)
    ax.set_zlim(-1 * plotMax, plotMax)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    imageName = name.replace(' ', '_') + '.pdf'
    if(outputDir):
        imageName = os.path.join(outputDir, str(cluster), imageName)
    plt.savefig(imageName)
    plt.clf()
    end = time()
    print('Plotting ETMIp-C surface plot took {} min'.format((end - start) / 60.0))


def writeOutClusteringResults(today, qName, cutoff, clus, alignment, pdb,
                              summary, coverage, outputDir):
    '''
    Write out clustering results

    This method writes the results of the clustering to file.

    Parameters:
    today: date
        Todays date.
    qName: str
        The name of the query protein
    cutoff : float
        The distance used for proximity cutoff in the PDB structure.
    clus: int
        The number of clusters created
    alignment: SeqAlignment
        The sequence alignment object associated with the ETMIPC instance
        calling this method.
    pdb: PDBReference
        Object representing the pdb structure used in the current
        analysis.  This object is passed in to enable access to the
        sortedPDBDist variable.
    summary : dict
        A dictionary of the clustering constants mapped to a matrix of the raw
        values from the whole MIp matrix through all clustering constants <=
        clus. See ETMIPC class description.
    coverage : dict
        A dictionary of the clustering constants mapped to a matrix of the
        coverage values computed on the summary matrices. See ETMIPC class
        description.
    outputDir : str
        The full path to where the output file should be stored. If None
        (default) the plot will be stored in the current working directory.
    '''
    start = time()
    convertAA = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'B': 'ASX',
                 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'Z': 'GLX', 'G': 'GLY',
                 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET',
                 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
                 'Y': 'TYR', 'V': 'VAL'}
    e = "{}_{}_{}.etmipCVG.clustered.txt".format(today, qName, clus)
    if(summary and coverage):
        cSummary = summary[clus]
        cCoverage = coverage[clus]
    else:
        cSummary = loadSingleMatrix('Summary', clus, outputDir)
        cCoverage = loadSingleMatrix('Coverage', clus, outputDir)
    if(outputDir):
        e = os.path.join(outputDir, str(clus), e)
    etMIPOutFile = open(e, "w+")
    etMIPWriter = csv.writer(etMIPOutFile, delimiter='\t')
    header = ['Pos1', '(AA1)', 'Pos2', '(AA2)', 'ETMIp_Score',
              'ETMIp_Coverage', 'Residue_Dist', 'Within_Threshold',
              'Cluster']
    etMIPWriter.writerow(header)
    if(pdb):
        mappedChain = pdb.fastaToPDBMapping[0]
    else:
        mappedChain = None
    for i in range(0, alignment.seqLength):
        for j in range(i + 1, alignment.seqLength):
            if(pdb is None):
                res1 = i + 1
                res2 = j + 1
                r = '-'
                dist = '-'
            else:
                if((i in pdb.fastaToPDBMapping[1]) or
                   (j in pdb.fastaToPDBMapping[1])):
                    if(i in pdb.fastaToPDBMapping[1]):
                        mapped1 = pdb.fastaToPDBMapping[1][i]
                        res1 = pdb.pdbResidueList[mappedChain][mapped1]
                    else:
                        res1 = '-'
                    if(j in pdb.fastaToPDBMapping[1]):
                        mapped2 = pdb.fastaToPDBMapping[1][j]
                        res2 = pdb.pdbResidueList[mappedChain][mapped2]
                    else:
                        res2 = '-'
                    if((i in pdb.fastaToPDBMapping[1]) and
                       (j in pdb.fastaToPDBMapping[1])):
                        dist = round(
                            pdb.residueDists[mappedChain][mapped1, mapped2], 4)
                    else:
                        dist = float('NaN')
                else:
                    res1 = '-'
                    res2 = '-'
                    dist = float('NaN')
                if(dist <= cutoff):
                    r = 1
                elif(np.isnan(dist)):
                    r = '-'
                else:
                    r = 0
            etMIPOutputLine = [res1, '({})'.format(
                convertAA[alignment.querySequence[i]]),
                res2, '({})'.format(
                convertAA[alignment.querySequence[j]]),
                round(cSummary[i, j], 4),
                round(cCoverage[i, j], 4), dist, r, clus]
            etMIPWriter.writerow(etMIPOutputLine)
    etMIPOutFile.close()
    end = time()
    print('Writing the ETMIP worker data to file took {} min'.format(
        (end - start) / 60.0))


def writeOutClusterScoring(today, qName, clus, alignment, mipMatrix, rawScores,
                           resMat, coverage, summary, outputDir):
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
    alignment : SeqAlignment
        The SeqAlignment object containing relevant information for this
        ETMIPC analysis.
    mipMatrix : np.ndarray
        Matrix scoring the coupling between all positions in the query
        sequence, as computed over all sequences in the input alignment.
    rawScores : dict
        The dictionary mapping clustering constant to coupling scores for all
        positions in the query sequences at the specified clustering constant
        created by hierarchical clustering.
    resMat : dict
        A dictionary mapping clustering constants to matrices which represent
        the integration of coupling scores across all clusters defined at that
        clustering constant.
    coverage : dict
        This dictionary maps clustering constants to a matrix of normalized
        coupling scores between 0 and 100, computed from the
        summaryMatrices.
    summary : dict
        This dictionary maps clustering constants to a matrix which combines
        the scores from the wholeMIPMatrix, all lower clustering constants,
        and this clustering constant.
    outputDir : str
        The full path to where the output file should be stored. If None
        (default) the plot will be stored in the current working directory.
    '''
    start = time()
    convertAA = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'B': 'ASX',
                 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'Z': 'GLX', 'G': 'GLY',
                 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET',
                 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
                 'Y': 'TYR', 'V': 'VAL'}
    if(rawScores and resMat and summary and coverage):
        cRawScores = rawScores[clus]
        cResMat = resMat[clus]
        cSummary = summary[clus]
        cCoverage = coverage[clus]
    else:
        cRawScores, _ = loadRawScoreMatrix(
            alignment.seqLength, clus, outputDir)
        cResMat = loadSingleMatrix('Result', clus, outputDir)
        cSummary = loadSingleMatrix('Summary', clus, outputDir)
        cCoverage = loadSingleMatrix('Coverage', clus, outputDir)
    e = "{}_{}_{}.all_scores.txt".format(today, qName, clus)
    if(outputDir):
        e = os.path.join(outputDir, str(clus), e)
    etMIPOutFile = open(e, "wb")
    etMIPWriter = csv.writer(etMIPOutFile, delimiter='\t')
    etMIPWriter.writerow(['Pos1', 'AA1', 'Pos2', 'AA2', 'OriginalScore'] +
                         ['C.' + i for i in map(str, range(1, clus + 1))] +
                         ['Cluster_Score', 'Summed_Score', 'ETMIp_Coverage'])
    for i in range(0, alignment.seqLength):
        for j in range(i + 1, alignment.seqLength):
            res1 = i + 1
            res2 = j + 1
            rowP1 = [res1, convertAA[alignment.querySequence[i]], res2,
                     convertAA[alignment.querySequence[j]],
                     round(mipMatrix[i, j], 4)]
            rowP2 = [round(cRawScores[c, i, j], 4)
                     for c in range(clus)]
            rowP3 = [round(cResMat[i, j], 4),
                     round(cSummary[i, j], 4),
                     round(cCoverage[i, j], 4)]
            etMIPWriter.writerow(rowP1 + rowP2 + rowP3)
    etMIPOutFile.close()
    end = time()
    print('Writing the ETMIP worker data to file took {} min'.format(
        (end - start) / 60.0))


def poolInit1(aaReference, wCC, originalAlignment, saveDir, alignLock,
              kQueue, subAlignmentQueue, resQueue, lowMem):
    '''
    poolInit

    A function which initializes processes spawned in a worker pool performing
    the etMIPWorker function.  This provides a set of variables to all working
    processes which are shared.

    Parameters:
    -----------
    aaReference : dict
        Dictionary mapping amino acid abbreviations.
    wCC : str
        Method by which to combine individual matrices from one round of
        clustering. The options supported now are: sum, average, size_weighted,
        and evidence_weighted.
    originialAlignment : SeqAlignment
        Alignment held by the instance of ETMIPC which called this method.
    saveDir : str
        The caching directory used to save results from agglomerative
        clustering.
    alignLock : multiprocessing.Manager.Lock()
        Lock used to regulate access to the alignment object for the purpose
        of setting the tree order.
    kQueue : multiprocessing.Manager.Queue()
        Queue used for tracking the k's for which clustering still needs to be
        performed.
    subAlignmentQueue : multiprocessing.Manager.Queue()
        Queue used to track the subalignments generated by clustering based on
        the k's in the kQueue.
    resQueue : multiprocessing.Manager.Queue()
        Queue used to track final results generated by this method.
    lowMem : bool
        Whether or not low memory mode should be used.
    '''
    global aaDict
    aaDict = aaReference
    global withinClusterCombi
    withinClusterCombi = wCC
    global initialAlignment
    initialAlignment = originalAlignment
    global cacheDir
    cacheDir = saveDir
    global kLock
    kLock = alignLock
    global queue1
    queue1 = kQueue
    global queue2
    queue2 = subAlignmentQueue
    global queue3
    queue3 = resQueue
    global pool1MemMode
    pool1MemMode = lowMem


def etMIPWorker1(inTup):
    '''
    ETMIP Worker

    Performs clustering and calculation of cluster dependent sequence distances.
    This function requires initialization of threads with poolInit, or setting
    of global variables as described in that function.

    Parameters:
    -----------
    inTup: tuple
        Tuple containing the one int specifying which process this is,
        and a second int specifying the number of active processes.
    Returns:
    --------
    dict
        Mapping of k, to sub-cluster, to size of sub-cluster.
    dict
        Mapping of k, to sub-cluster, to the SeqAlignment object reprsenting
        the sequence IDs present in that sub-cluster.
    dict
        Mapping of k to the time spent working on data from that k by this
        process.
    '''
    currProcess, totalProcesses = inTup
    clusterSizes = {}
    clusterTimes = {}
    subAlignments = {}
    while((not queue1.empty()) or (not queue2.empty())):
        try:
            print('Processes {}:{} acquiring sub alignment!'.format(
                currProcess, totalProcesses))
            clus, sub, newAlignment = queue2.get_nowait()
            print('Current alignment has {} sequences'.format(newAlignment.size))
            start = time()
            if('evidence' in withinClusterCombi):
                clusteredMIPMatrix, evidenceMat = wholeAnalysis(
                    newAlignment, True)
            else:
                clusteredMIPMatrix, evidenceMat = wholeAnalysis(
                    newAlignment, False)
            end = time()
            timeElapsed = end - start
            if(clus in clusterTimes):
                clusterTimes[clus] += timeElapsed
            else:
                clusterTimes[clus] = timeElapsed
            print('ETMIP worker took {} min'.format(timeElapsed / 60.0))
            if(pool1MemMode):
                saveRawScoreMatrix(clus, sub, clusteredMIPMatrix, evidenceMat,
                                   cacheDir)
            else:
                queue3.put((clus, sub, clusteredMIPMatrix, evidenceMat))
            print('Processes {}:{} pushing cET-MIp scores!'.format(
                currProcess, totalProcesses))
            continue
        except Queue.Empty:
            print('Processes {}:{} failed to acquire-sub alignment!'.format(
                currProcess, totalProcesses))
            pass
        try:
            print('Processes {}:{} acquiring k to generate clusters!'.format(
                currProcess, totalProcesses))
            kLock.acquire()
            print('Lock acquired by: {}'.format(currProcess))
            c = queue1.get_nowait()
            print('K: {} acquired setting tree'.format(c))
            start = time()
            clusterSizes[c] = {}
            clusDict, clusDet = initialAlignment.aggClustering(nCluster=c,
                                                               cacheDir=cacheDir)
            treeOrdering = []
            subAlignments[c] = {}
            for sub in clusDet:
                newAlignment = initialAlignment.generateSubAlignment(
                    clusDict[sub])
                clusterSizes[c][sub] = newAlignment.size
                # Create matrix converting sequences of amino acids to sequences of
                # integers representing sequences of amino acids
                newAlignment.alignment2num(aaDict)
                queue2.put((c, sub, newAlignment))
                treeOrdering += newAlignment.treeOrder
                subAlignments[c][sub] = newAlignment
            initialAlignment.setTreeOrdering(tOrder=treeOrdering)
            end = time()
            kLock.release()
            if(c in clusterTimes):
                clusterTimes[c] += (end - start)
            else:
                clusterTimes[c] = (end - start)
            print('Processes {}:{} pushing new sub-alignment!'.format(
                currProcess, totalProcesses))
            continue
        except Queue.Empty:
            kLock.release()
            print('Processes {}:{} failed to acquire k!'.format(
                currProcess, totalProcesses))
            pass
    print('Process: {} completed and returning!'.format(currProcess))
    return clusterSizes, subAlignments, clusterTimes


def poolInit2(c, qAlignment, qStructure, outDir):
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
    qStructure: PDBReference
        Object containing the PDB information for this analysis.
    '''
    global cutoff
    cutoff = c
    global seqLen
    seqLen = qAlignment.seqLength
    global pdbResidueList
    global pdbDist
    global seqToPDB
    global pdbStructure
    if(qStructure is None):
        pdbResidueList = None
        pdbDist = None
        seqToPDB = None
        pdbStructure = None
    else:
        mappedChain = qStructure.fastaToPDBMapping[0]
        pdbResidueList = qStructure.pdbResidueList[mappedChain]
        pdbDist = qStructure.residueDists[mappedChain]
        seqToPDB = qStructure.fastaToPDBMapping[1]
        pdbStructure = qStructure
    global w2OutDir
    w2OutDir = outDir


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
    float
        The time in seconds which it took to perform clustering.
    list
        List of false positive rates.
    list
        List of true positive rates.
    float
        The ROCAUC value for this clustering.
    '''
    clus, allSummedMatrix = inTup
    if(allSummedMatrix):
        summedMatrix = allSummedMatrix[clus]
    else:
        summedMatrix = loadSingleMatrix('Summary', clus, w2OutDir)
    start = time()
    coverage = np.zeros(summedMatrix.shape)
    testMat = np.triu(summedMatrix)
    mask = np.triu(np.ones(summedMatrix.shape), k=1)
    normalization = ((summedMatrix.shape[0]**2 - summedMatrix.shape[0]) / 2.0)
    for i in range(summedMatrix.shape[0]):
        for j in range(i + 1, summedMatrix.shape[0]):
            boolMat = (testMat[i, j] >= testMat) * 1.0
            correctedMat = boolMat * mask
            computeCoverage2 = (((np.sum(correctedMat) - 1) * 100) /
                                normalization)
            coverage[i, j] = coverage[j, i] = computeCoverage2
    # Defining which of the values which there are ETMIPC scores for have
    # distance measurements in the PDB Structure
    indices = np.triu_indices(summedMatrix.shape[0], 1)
    if(seqToPDB is not None):
        mappablePos = np.array(seqToPDB.keys())
        xMappable = np.in1d(indices[0], mappablePos)
        yMappable = np.in1d(indices[1], mappablePos)
        finalMappable = xMappable & yMappable
        indices = (indices[0][finalMappable], indices[1][finalMappable])
#     etmiplistCoverage = coverage[np.triu_indices(summedMatrix.shape[0], 1)]
    etmiplistCoverage = coverage[indices]
    # Mapping indices used for ETMIPC coverage list so that it can be used to
    # retrieve correct distances from PDB distances matrix.
    if(seqToPDB is not None):
        keys = sorted(seqToPDB.keys())
        values = [seqToPDB[k] for k in keys]
        replace = np.array([keys, values])
        mask1 = np.in1d(indices[0], replace[0, :])
        indices[0][mask1] = replace[1, np.searchsorted(replace[0, :],
                                                       indices[0][mask1])]
        mask2 = np.in1d(indices[1], replace[0, :])
        indices[1][mask2] = replace[1, np.searchsorted(replace[0, :],
                                                       indices[1][mask2])]
        sortedPDBDist = pdbDist[indices]
    else:
        sortedPDBDist = None
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
    if(allSummedMatrix == None):
        saveSingleMatrix('Coverage', clus, coverage, w2OutDir)
        coverage = None
    return (clus, coverage, timeElapsed, fpr, tpr, roc_auc)


def poolInit3(clusterQueue, outputQueue, qName, today, cutOff, verbosity,
              classMIPMatrix, classRawScores, classResultMatrices,
              classCoverage, classSummary, classSubalignments, classAlignment,
              classAUCs, classPDB, outputDir):
    '''
    poolInit3

    A function which initializes processes spawned in a worker pool performing
    the etMIPWorker3 function.  This provides a set of variables to all working
    processes which are shared.

    Parameters:
    -----------
    clusterQueue : multiprocessing.Manager.Queue()
        Queue used for tracking the k's for which output still needs to be
        generated.
    outputQueue : multiprocessing.Manager.Queue()
        Queue used for tracking the types of output to be generated and the
        inputs for the dependent methods.
    qName : str
        The name of the query string.
    today : str
        The current date which will be used for identifying the proper directory
        to store files in.
    cutOff : float
        Distance in Angstroms between residues considered as a preliminary
        positive set of coupled residues based on spatial positions in the PDB
        file if provided.
    verbosity : int
        How many figures to produce.1 = ROC Curves, ETMIP Coverage file,
        and final AUC and Timing file. 2 = files with all scores at each
        clustering. 3 = sub-alignment files and plots. 4 = surface plots
        and heatmaps of ETMIP raw and coverage scores.'
    classMIPMatrix : np.ndarray
        Matrix scoring the coupling between all positions in the query
        sequence, as computed over all sequences in the input alignment.
    classRawScores : dict
        The dictionary mapping clustering constant to coupling scores for all
        positions in the query sequences at the specified clustering constant
        created by hierarchical clustering.
    classResultMatrices : dict
        A dictionary mapping clustering constants to matrices which represent
        the integration of coupling scores across all clusters defined at that
        clustering constant.
    classCoverage : dict
        This dictionary maps clustering constants to a matrix of normalized
        coupling scores between 0 and 100, computed from the
        summaryMatrices.
    classSummary : dict
        This dictionary maps clustering constants to a matrix which combines
        the scores from the wholeMIPMatrix, all lower clustering constants,
        and this clustering constant.
    classSubalignments : dict
            A dictionary mapping a clustering constant (k) to another dictionary
            which maps a cluster label (0 to k-1) to a SeqAlignment object
            containing only the sequences for that specific cluster.
    classAlignment : SeqAlignment
        The SeqAlignment object containing relevant information for this
        ETMIPC analysis.
    classAUCs : dictionary
        AUC values stored in the ETMIPC class, used to identify the specific
        values for the specified clustering constant (clus).
    classPDB : PDBReference
        Object representing the pdb structure used in the current
        analysis.
    outputDir : str
        The full path to where the output generated by this process should be
        stored. If None (default) the plot will be stored in the current working
        directory.
    '''
    global queue1
    queue1 = clusterQueue
    global queue2
    queue2 = outputQueue
    global queryN
    queryN = qName
    global date
    date = today
    global threshold
    threshold = cutOff
    global ver
    ver = verbosity
    global mipMatrix
    mipMatrix = classMIPMatrix
    global rawScores
    rawScores = classRawScores
    global resMat
    resMat = classResultMatrices
    global subAlignments
    subAlignments = classSubalignments
    global alignment
    alignment = classAlignment
    global coverage
    coverage = classCoverage
    global summary
    summary = classSummary
    global aucs
    aucs = classAUCs
    global pdb
    pdb = classPDB
    global outDir
    outDir = outputDir


def etMIPWorker3(inputTuple):
    '''
    ETMIP Worker 3

    This method uses queues to generate the jobs necessary to create the final
    output of the ETMIPC class ProduceFinalFigures method (figures and 
    output files). One queue is used to hold the clustering constants to be
    processed (producer) while another queue is used to hold the functions
    to call and the input data to provide (producer). This method directs a
    process to preferentially pull jobs from the second queue, unless none are
    available, in which case it directs the process to generate additional jobs
    using queue 1. If both queues are empty the method terminates.

    Parameters:
    -----------
    inTup: tuple
        Tuple containing the one int specifying which process this is,
        and a second int specifying the number of active processes.
    Returns:
    --------
    dict
        Mapping of k to the time spent working on data from that k by this
        process.
    '''
    currProcess, totalProcesses = inputTuple
    times = {}
    functionDict = {'heatmap': heatmapPlot, 'surfacePlot': surfacePlot,
                    'writeClusterResults': writeOutClusteringResults,
                    'writeClusterScoring': writeOutClusterScoring,
                    'plotAUC': plotAUC, 'subAlignment': None}
    while((not queue1.empty()) or (not queue2.empty())):
        try:
            qFunc, qParam = queue2.get_nowait()
            print('Calling: {} in processes {}:{}'.format(qFunc, currProcess,
                                                          totalProcesses))
            if(qFunc == 'subAlignment'):
                c, sub, curOutDir = qParam
                subAlignments[c][sub].setTreeOrdering(alignment.treeOrder)
                subAlignments[c][sub].writeOutAlignment(
                    curOutDir + 'AlignmentForK{}_{}.fa'.format(c, sub))
                subAlignments[c][sub].heatmapPlot(
                    'Alignment For K {} {}'.format(c, sub), curOutDir)
            else:
                start = time()
                functionDict[qFunc](*qParam)
                end = time()
                if(qFunc == 'writeClusterResults'):
                    timeElapsed = end - start
                    c = qParam[3]
                    if(c not in times):
                        times[c] = timeElapsed
                    else:
                        times[c] += timeElapsed
        except Queue.Empty:
            pass
        try:
            c = queue1.get_nowait()
            currOutDir = '{}/{}/'.format(outDir, c)
            if(not os.path.exists(currOutDir)):
                os.mkdir(currOutDir)
            if(ver >= 1):
                queue2.put_nowait(('writeClusterResults',
                                   (date, queryN, threshold, c, alignment,
                                    pdb, summary, coverage,
                                    outDir)))
                if(pdb):
                    queue2.put_nowait(('plotAUC',
                                       (queryN, c, date, threshold, aucs,
                                        currOutDir)))
            if(ver >= 2):
                queue2.put_nowait(
                    ('writeClusterScoring',
                     (date, queryN, c, alignment, mipMatrix, rawScores, resMat,
                      coverage, summary, outDir)))
            if(ver >= 3):
                for sub in range(c):
                    queue2.put_nowait(('subAlignment', (c, sub, currOutDir)))
            if(ver >= 4):
                queue2.put_nowait(
                    ('heatmap', ('Raw Score Heatmap K {}'.format(c), summary, c, outDir)))
                queue2.put_nowait(
                    ('heatmap', ('Coverage Heatmap K {}'.format(c), coverage, c, outDir)))
                queue2.put_nowait(
                    ('surfacePlot', ('Raw Score Surface K {}'.format(c), summary, c, outDir)))
                queue2.put_nowait(
                    ('surfacePlot', ('Coverage Surface K {}'.format(c), coverage, c, outDir)))
        except Queue.Empty:
            pass
    print('Function completed by {}:{}'.format(currProcess, totalProcesses))
    return times
