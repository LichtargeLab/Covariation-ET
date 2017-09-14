'''
Created on Aug 21, 2017

@author: dmkonecki
'''
import os
import csv
import sys
import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pylab as pl
from seaborn import heatmap
from multiprocessing import Pool
# from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mutual_info_score, auc, roc_curve
from IPython import embed


class ETMIPC(object):
    '''
    classdocs
    '''

    def __init__(self, alignment, clusters, pdb, outputDir, processes):
        '''
        Constructor

        Initiates an instance of the ETMIPC class which stores the
        following data:

        alignment : SeqAlignment
            The SeqAlignment object containing relevant information for this
            ETMIPC analysis.
        clusters : list
            The k's for which to create different clusterings.
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
        self.aucs = {}

    def determineWholeMIP(self, evidence, alterInput):
        '''
        determineWholeMIP

        Paramters:
        evidence : bool
            Whether or not to normalize using the evidence using the evidence
            counts computed while performing the coupling scoring.
        alterInput: bool
            Whether or not to restrict the input to the mutual information
            computation to only those sequences which have gaps in neither of
            the considered positions.

        This method performs the wholeAnalysis method on all sequences in the
        sequence alignment. This method updates the wholeMIPMatrix and
        wholeEvidenceMatrix class variables.
        '''
        mipMatrix, evidenceCounts = wholeAnalysis(self.alignment, evidence,
                                                  alterInput,
                                                  saveFile='wholeMIP')
        self.wholeMIPMatrix = mipMatrix
        self.wholeEvidenceMatrix = evidenceCounts

    def calculateClusteredMIPScores(self, aaDict, wCC, alterInput):
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
        alter: bool
            Whether or not to restrict the input to the mutual information
            computation to only those sequences which have gaps in neither of
            the considered positions.
        '''
        # Generate clusters and jobs to perform
        inputs = []
        clusterSizes = {}
#         subAlignments = []
        for c in self.clusters:
            start = time()
            clusterSizes[c] = {}
            resultDir = self.outputDir + '/{}/'.format(c)
            if(not os.path.exists(resultDir)):
                os.mkdir(resultDir)
            os.chdir(resultDir)
            clusDict, clusDet = self.alignment.aggClustering(nCluster=c,
                                                             cacheDir=self.outputDir)
            treeOrdering = []
            for sub in clusDet:
                newAlignment = self.alignment.generateSubAlignment(
                    clusDict[sub])
                clusterSizes[c][sub] = newAlignment.size
                # Create matrix converting sequences of amino acids to sequences of
                # integers representing sequences of amino acids
                newAlignment.alignment2num(aaDict)
                inputs.append((c, sub, newAlignment))
                treeOrdering += newAlignment.treeOrder
                self.subAlignments[c][sub] = newAlignment
            self.alignment.setTreeOrdering(tOrder=treeOrdering)
            os.chdir('..')
            end = time()
            self.resultTimes[c] += end - start
        # Perform jobs
        if(self.processes == 1):
            poolInitTemp(wCC, alterInput)
            res1 = []
            for i in inputs:
                res = etMIPWorkerTemp(i)
                res1.append(res)
        else:
            pool = Pool(processes=self.processes, initializer=poolInitTemp,
                        initargs=(wCC, alterInput))
            res1 = pool.map_async(etMIPWorkerTemp, inputs)
            pool.close()
            pool.join()
            res1 = res1.get()
        # Retrieve results
        for r in res1:
            self.rawScores[r[0]][r[1]] = r[2]
            self.evidenceCounts[r[0]][r[1]] = r[3]
            self.resultTimes[r[0]] += r[4]
        # Combine results
        for c in self.clusters:
            start = time()
            # Additive clusters
            if(wCC == 'sum'):
                resMatrix = np.sum(self.rawScores[c], axis=0)
            # Normal average over clusters
            elif(wCC == 'average'):
                resMatrix = np.mean(self.rawScores[c], axis=0)
            # Weighted average over clusters based on cluster sizes
            elif(wCC == 'size_weighted'):
                weighting = np.array([clusterSizes[c][s]
                                      for s in sorted(clusterSizes[c].keys())])
                resMatrix = weighting[:, None, None] * self.rawScores[c]
                resMatrix = np.sum(resMatrix, axis=0) / self.alignment.size
            # Weighted average over clusters based on evidence counts at each
            # pair vs. the number of sequences with evidence for that pairing.
            elif(wCC == 'evidence_weighted'):
                resMatrix = (np.sum(self.rawScores[c] * self.evidenceCounts[c],
                                    axis=0) / np.sum(self.evidenceCounts[c], axis=0))
            # Weighted average over clusters based on evidence counts at each
            # pair vs. the entire size of the alignment.
            elif(wCC == 'evidence_vs_size'):
                resMatrix = (np.sum(self.rawScores[c] * self.evidenceCounts[c],
                                    axis=0) / float(self.alignment.size))
            else:
                print 'Combination method not yet implemented'
                raise NotImplementedError()
            resMatrix[np.isnan(resMatrix)] = 0.0
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
            self.summaryMatrices[currClus] += self.wholeMIPMatrix
            for j in [c for c in self.clusters if c <= currClus]:
                self.summaryMatrices[currClus] += self.resultMatrices[j]
            if(combination == 'average'):
                self.summaryMatrices[currClus] /= (i + 2)
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
                poolInit2(threshold, self.alignment, self.pdb)
                r = etMIPWorker2((clus, self.summaryMatrices[clus]))
                res2.append(r)
        else:
            pool2 = Pool(processes=self.processes, initializer=poolInit2,
                         initargs=(threshold, self.alignment, self.pdb))
            res2 = pool2.map_async(etMIPWorker2,
                                   [(clus, self.summaryMatrices[clus])
                                    for clus in self.clusters])
            pool2.close()
            pool2.join()
            res2 = res2.get()
        for r in res2:
            self.coverage[r[0]] = r[1]
            self.resultTimes[r[0]] += r[2]
            self.aucs[r[0]] = r[3:]
        end = time()
        print('Computing coverage and AUC took {} min'.format((end - start) / 60.0))

    def produceFinalFigures(self, today, cutOff):
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
        '''
        start = time()
        self.alignment.heatmapPlot('Overall Alignment')
        for c in self.clusters:
            clusterDir = '{}/'.format(c)
            if(not os.path.exists(clusterDir)):
                os.mkdir(clusterDir)
            os.chdir(clusterDir)
            start = time()
            qName = self.alignment.queryID.split('_')[1]
            if(self.pdb):
                self.plotAUC(qName, c, today, cutOff)
            for sub in self.subAlignments[c].keys():
                self.subAlignments[c][sub].writeOutAlignment(
                    fileName='AlignmentForK{}_{}.fa'.format(c, sub))
                self.subAlignments[c][sub].setTreeOrdering(
                    tOrder=self.alignment.treeOrder)
                self.subAlignments[c][sub].heatmapPlot(
                    name='Aligment For K {} {}'.format(c, sub))
            self.writeOutClusterScoring(today, qName, c)
            self.writeOutClusteringResults(today, qName, cutOff, c)
            self.heatmapPlot('Raw Score Heatmap K {}'.format(c),
                             normalized=False, cluster=c)
            self.surfacePlot('Raw Score Surface K {}'.format(c),
                             normalized=False, cluster=c)
            self.heatmapPlot('Coverage Heatmap K {}'.format(c), normalized=True,
                             cluster=c)
            self.surfacePlot('Coverage Surface K {}'.format(c), normalized=True,
                             cluster=c)
            end = time()
            timeElapsed = end - start
            self.resultTimes[c] += timeElapsed
            os.chdir('..')
        end = time()
        print('Producing final figures took {} min'.format((end - start) / 60.0))

    def plotAUC(self, qName, clus, today, cutoff):
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

    def heatmapPlot(self, name, normalized, cluster):
        '''
        Heatmap Plot

        This method creates a heatmap using the Seaborn plotting package. The
        data used can come from the summaryMatrices or coverage data.

        Parameters:
        -----------
        name : str
            Name used as the title of the plot and the filename for the saved
            figure.
        normalized : bool
            Whether or not to use the coverage data or not. If not the summary
            Matrices data will be used.
        cluster : int
            The clustering constant for which to create a heatmap.
        '''
        start = time()
        if(normalized):
            relData = self.coverage
        else:
            relData = self.summaryMatrices
        dataMat = relData[cluster]
        dmMax = np.max(dataMat)
        dmMin = np.min(dataMat)
        plotMax = max([dmMax, abs(dmMin)])
        heatmap(data=dataMat, cmap='jet', center=0.0, vmin=-1 * plotMax,
                vmax=plotMax, cbar=True, square=True)
        plt.title(name)
        plt.savefig(name.replace(' ', '_') + '.pdf')
        plt.clf()
        end = time()
        print('Plotting ETMIp-C heatmap took {} min'.format((end - start) / 60.0))

    def surfacePlot(self, name, normalized, cluster):
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
        normalized : bool
            Whether or not to use the coverage data or not. If not the summary
            Matrices data will be used.
        cluster : int
            The clustering constant for which to create a heatmap.
        '''
        start = time()
        if(normalized):
            relData = self.coverage
        else:
            relData = self.summaryMatrices
        dataMat = relData[cluster]
        dmMax = np.max(dataMat)
        dmMin = np.min(dataMat)
        plotMax = max([dmMax, abs(dmMin)])
        X = Y = np.arange(max(dataMat.shape))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, dataMat, cmap='jet',  # cm.jet,
                               linewidth=0, antialiased=False)
        ax.set_zlim(-1 * plotMax, plotMax)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(name.replace(' ', '_') + '.pdf')
        plt.clf()
        end = time()
        print('Plotting ETMIp-C surface plot took {} min'.format((end - start) / 60.0))

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
                         round(self.wholeMIPMatrix[i, j], 4)]
                rowP2 = [round(self.rawScores[clus][c, i, j], 4)
                         for c in range(clus)]
                rowP3 = [round(self.resultMatrices[clus][i, j], 4),
                         round(self.summaryMatrices[clus][i, j], 4),
                         round(self.coverage[clus][i, j], 4)]
                etMIPWriter.writerow(rowP1 + rowP2 + rowP3)
        etMIPOutFile.close()
        end = time()
        print('Writing the ETMIP worker data to file took {} min'.format(
            (end - start) / 60.0))

    def writeOutClusteringResults(self, today, qName, cutoff, clus):
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
        pdb: PDBReference
            Object representing the pdb structure used in the current
            analysis.  This object is passed in to enable access to the
            sortedPDBDist variable.
        sortedPDBDist: numpy nd array
            Array of the distances between sequences, sorted by sequence indices.
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
        for i in range(0, self.alignment.seqLength):
            for j in range(i + 1, self.alignment.seqLength):
                if(self.pdb is None):
                    res1 = i + 1
                    res2 = j + 1
                    r = '-'
                    dist = '-'
                else:
                    if((i in self.pdb.fastaToPDBMapping) or
                       (j in self.pdb.fastaToPDBMapping)):
                        if(i in self.pdb.fastaToPDBMapping):
                            mapped1 = self.pdb.fastaToPDBMapping[i]
                            res1 = self.pdb.pdbResidueList[mapped1]
                        else:
                            res1 = '-'
                        if(j in self.pdb.fastaToPDBMapping):
                            mapped2 = self.pdb.fastaToPDBMapping[j]
                            res2 = self.pdb.pdbResidueList[mapped2]
                        else:
                            res2 = '-'
                        if((i in self.pdb.fastaToPDBMapping) and
                           (j in self.pdb.fastaToPDBMapping)):
                            dist = round(
                                self.pdb.residueDists[mapped1, mapped2], 4)
                        else:
                            dist = float('NaN')
                    if(dist <= cutoff):
                        r = 1
                    elif(np.isnan(dist)):
                        r = '-'
                    else:
                        r = 0
                etMIPOutputLine = [res1, '({})'.format(
                    convertAA[self.alignment.querySequence[i]]),
                    res2, '({})'.format(
                    convertAA[self.alignment.querySequence[j]]),
                    round(self.summaryMatrices[clus][i, j], 4),
                    round(self.coverage[clus][i, j], 4), dist, r, clus]
                etMIPWriter.writerow(etMIPOutputLine)
        etMIPOutFile.close()
        end = time()
        print('Writing the ETMIP worker data to file took {} min'.format(
            (end - start) / 60.0))

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
###############################################################################
#
###############################################################################


def wholeAnalysis(alignment, evidence, alterInput, saveFile=None):
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
                if(evidence or alterInput):
                    colSubI, colSubJ, _pos, ev = alignment.identifyComparableSequences(
                        i, j)
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


def poolInitTemp(wCC, alterInput):
    '''
    poolInit

    A function which initializes processes spawned in a worker pool performing
    the etMIPWorker function.  This provides a set of variables to all working
    processes which are shared.

    Parameters:
    -----------
    wCC : str
        Method by which to combine individual matrices from one round of
        clustering. The options supported now are: sum, average, size_weighted,
        and evidence_weighted.
    alterInput: bool
        Whether or not to restrict the input to the mutual information
        computation to only those sequences which have gaps in neither of
        the considered positions.
    '''
    global withinClusterCombi
    withinClusterCombi = wCC
    global alterMIInput
    alterMIInput = alterInput


def etMIPWorkerTemp(inTup):
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
        The clustering constant K.
    int
        The subalignment analyzed in this alignment.
    numpy.array
        A matrix of pairwise distances between sequences based on subalignments
        formed during clustering.
    numpy.array
        A matrix containing not only the overall clustering scores for the
        specified clustering constants, but all of the scores computed in the
        subalignments identified by the hierarchical clustering.
    float
        The time in seconds which it took to perform clustering.
    '''
    clus, sub, newAlignment = inTup
    start = time()
    if('evidence' in withinClusterCombi):
        clusteredMIPMatrix, evidenceMat = wholeAnalysis(newAlignment, True,
                                                        alterMIInput)
    else:
        clusteredMIPMatrix, evidenceMat = wholeAnalysis(newAlignment, False,
                                                        alterMIInput)
    end = time()
    timeElapsed = end - start
    print('ETMIP worker took {} min'.format(timeElapsed / 60.0))
    return clus, sub, clusteredMIPMatrix, evidenceMat, timeElapsed


def poolInit2(c, qAlignment, qStructure):
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
    '''
    global cutoff
    cutoff = c
    global seqLen
    seqLen = qAlignment.seqLength
    global pdbResidueList
    global pdbDist
    global seqToPDB
    if(qStructure is None):
        pdbResidueList = None
        pdbDist = None
        seqToPDB = None
    else:
        pdbResidueList = qStructure.pdbResidueList
        pdbDist = qStructure.residueDists
        seqToPDB = qStructure.fastaToPDBMapping


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
    clus, summedMatrix = inTup
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
    mappablePos = np.array(seqToPDB.keys())
    xMappable = np.in1d(indices[0], mappablePos)
    yMappable = np.in1d(indices[1], mappablePos)
    finalMappable = xMappable & yMappable
    indices = (indices[0][finalMappable], indices[1][finalMappable])
#     etmiplistCoverage = coverage[np.triu_indices(summedMatrix.shape[0], 1)]
    etmiplistCoverage = coverage[indices]
    # Mapping indices used for ETMIPC coverage list so that it can be used to
    # retrieve correct distances from PDB distances matrix.
    replace = np.array([list(seqToPDB.keys()), list(seqToPDB.values())])
    mask1 = np.in1d(indices[0], replace[0, :])
    indices[0][mask1] = replace[1, np.searchsorted(replace[0, :],
                                                   indices[0][mask1])]
    mask2 = np.in1d(indices[1], replace[0, :])
    indices[1][mask2] = replace[1, np.searchsorted(replace[0, :],
                                                   indices[1][mask2])]
    sortedPDBDist = pdbDist[indices]
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
