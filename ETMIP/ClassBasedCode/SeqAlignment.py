'''
Created on Aug 17, 2017

@author: daniel
'''
from sklearn.cluster import AgglomerativeClustering
import cPickle as pickle
from time import time
import pandas as pd
import numpy as np
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seaborn import heatmap, clustermap
from IPython import embed


class SeqAlignment(object):
    '''
    classdocs
    '''

    def __init__(self, fileName, queryID):
        '''
        Constructor

        Initiates an instance of the SeqAlignment class which stores the
        following data:

        fileName: str
            The file path to the file from which the alignment can be parsed.
        queryID: str
            The provided queryID prepended with >query_, which should be the
            identifier for query sequence in the alignment file.
        size: int
            The number of sequences in the alignment represented by this object.
        seqOrder: list
            List of sequence ids in the order in which they were parsed from the
            alignment file.
        seqLength: int
            The length of the query sequence.
        querySequence: str
            The sequence matching the sequence identifier give by the queryID
            variable.
        alignmentDict: dict
            A dictionary mapping sequence IDs with their sequences as parsed
            from the alignment file.
        alignmentMatrix: np.array
            A numerical representation of alignment, every amino acid has been
            assigned a numerical representation as has the gap symbol.  All
            rows are different sequences as described in seqOrder, while each
            column in the matrix is a position in the sequence.
        distanceMatrix: np.array
            A matrix with the identity scores between sequences in the
            alignment.
        '''
        self.fileName = fileName
        self.queryID = '>query_' + queryID
        self.alignmentDict = None
        self.seqOrder = None
        self.querySequence = None
        self.alignmentMatrix = None
        self.seqLength = None
        self.size = None
        self.distanceMatrix = None
        self.treeOrder = None

    def importAlignment(self, saveFile=None):
        '''
        Import alignments:

        This method imports the alignments into the class and forces all
        non-amino acids to take on the standard gap character "-".  This
        updates the alignmentDict, seqOrder, querySequence, seqLength, and size
        class variables.

        Parameters:
        -----------
        saveFile: str
            Path to file in which the desired alignment was stored previously.
        '''
        start = time()
        if((saveFile is not None) and (os.path.exists(saveFile))):
            alignment, seqOrder = pickle.load(open(saveFile, 'rb'))
        else:
            faFile = open(self.fileName, 'rb')
            alignment = {}
            seqOrder = []
            for line in faFile:
                if line.startswith(">"):
                    key = line.rstrip()
                    alignment[key] = ''
                    seqOrder.append(key)
                else:
                    alignment[key] += re.sub(r'[^a-zA-Z]', '-', line.rstrip())
            faFile.close()
            if(saveFile is not None):
                pickle.dump((alignment, seqOrder), open(saveFile, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
        end = time()
        print('Importing alignment took {} min'.format((end - start) / 60.0))
        self.alignmentDict = alignment
        self.seqOrder = seqOrder
        self.querySequence = self.alignmentDict[self.queryID]
        self.seqLength = len(self.querySequence)
        self.size = len(self.alignmentDict)

    def writeOutAlignment(self, fileName):
        '''
        This method writes out the alignment in the standard fa format.  Any
        sequence which is longer than 60 positions will be split over multiple
        lines with 60 characters per line.

        Parameters:
        fileName: str
            Path to file where the alignment should be written.
        '''
        outFile = open(fileName, 'wb')
        for seqId in self.seqOrder:
            if(seqId in self.alignmentDict):
                outFile.write(seqId + '\n')
                seqLen = len(self.alignmentDict[seqId])
                breaks = seqLen / 60
                if((seqLen % 60) != 0):
                    breaks += 1
                for i in range(breaks):
                    startPos = 0 + i * 60
                    endPos = 60 + i * 60
                    outFile.write(
                        self.alignmentDict[seqId][startPos: endPos] + '\n')
            else:
                pass
        outFile.close()

    def heatmapPlot(self, name):
        '''
        Heatmap Plot

        This method creates a heatmap using the Seaborn plotting package. The
        data used can come from the summaryMatrices or coverage data.

        Parameters:
        -----------
        name : str
            Name used as the title of the plot and the filename for the saved
            figure.
        cluster : int
            The clustering constant for which to create a heatmap.
        '''
#         embed()
#         exit()
        reIndexing = [self.seqOrder.index(x) for x in self.treeOrder]
        hm = heatmap(data=self.alignmentMatrix[reIndexing, :], cmap='jet',
                     center=10.0, vmin=0.0, vmax=20.0, cbar=True, square=False)
        hm.set_xticklabels(list(self.querySequence), fontsize=6, rotation=0)
        hm.set_yticklabels(self.treeOrder, fontsize=8, rotation=0)
        plt.title(name)
        plt.savefig(name.replace(' ', '_') + '.pdf')
        plt.clf()

    def removeGaps(self, saveFile=None):
        '''
        Remove Gaps

        Removes all gaps from the query sequence and removes characters at the
        corresponding positions in all other sequences. This method updates the
        class variables alignmentDict, querySequence, and seqLength.

        Parameters:
        -----------
        saveFile: str
            Path to a file where the alignment with gaps in the query sequence
            removed was stored previously.
        '''
        start = time()
        if((saveFile is not None) and os.path.exists(saveFile)):
            newAlignment = pickle.load(
                open(saveFile, 'rb'))
        else:
            queryArr = np.array(list(self.querySequence))
            queryUngappedInd = np.where(queryArr != '-')[0]
            if(len(queryUngappedInd) > 0):
                newAlignment = {}
                for key, value in self.alignmentDict.iteritems():
                    currArr = np.array(list(value))[queryUngappedInd]
                    newAlignment[key] = currArr.tostring()
            else:
                pass
            if(saveFile is not None):
                pickle.dump((newAlignment), open(saveFile, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
        self.alignmentDict = newAlignment
        self.querySequence = self.alignmentDict[self.queryID]
        self.seqLength = len(self.querySequence)
        end = time()
        print('Removing gaps took {} min'.format((end - start) / 60.0))

    def alignment2num(self, aaDict):
        '''
        Alignment2num

        Converts an alignment dictionary to a numerical representation.  This
        method updates the alignmentMatrix class variable.

        Parameters:
        -----------
        key_list: list
            Ordered set of sequence ids which specifies the ordering of the
            sequences along the row dimension of the resulting matrix.
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
        alignment2Num = np.zeros((self.size, self.seqLength))
        for i in range(self.size):
            for j in range(self.seqLength):
                currSeq = self.alignmentDict[self.seqOrder[i]]
                alignment2Num[i, j] = aaDict[currSeq[j]]
        self.alignmentMatrix = alignment2Num

    def computeDistanceMatrix(self, saveFile=None):
        '''
        Distance matrix

        Computes the sequence identity distance between a set of sequences and
        returns a matrix of the pairwise distances.  This method updates the
        distanceMatrix class variable.

        Parameters:
        -----------
        saveFile: str
            The path for an .npz file containing distances between sequences in
            the alignment (leave out the .npz as it will be added automatically.
        '''
        start = time()
        if((saveFile is not None) and os.path.exists(saveFile + '.npz')):
            valueMatrix = np.load(saveFile + '.npz')['X']
        else:
            valueMatrix = np.zeros([self.size, self.size])
            for i in range(self.size):
                check = self.alignmentMatrix - self.alignmentMatrix[i]
                valueMatrix[i] = np.sum(check == 0, axis=1)
            valueMatrix[np.arange(self.size), np.arange(self.size)] = 0
            valueMatrix /= self.seqLength
            if(saveFile is not None):
                np.savez(saveFile, X=valueMatrix)
        end = time()
        print('Computing the distance matrix took {} min'.format(
            (end - start) / 60.0))
        self.distanceMatrix = valueMatrix

    def setTreeOrdering(self, cacheDir, precomputed=False):
        '''
        Determine the ordering of the sequences from the full clustering tree
        used when separating the alignment into sub-clusters.

        cacheDir : str
            The path to the directory where the clustering model can be stored
            for access later when identifying different numbers of clusters.
        precomputed: boolean
            Whether or not to use the distances from X as the distances to
            cluster on, the alternative is to compute a new distance matrix
            based on X using Euclidean distance.
        '''
        if(self.treeOrder is None):
            df = pd.DataFrame(self.alignmentMatrix,
                              columns=list(self.querySequence),
                              index=self.treeOrder)
            hm = clustermap(df, method='ward', metric='euclidean',
                            z_score=None, standard_scale=None, row_cluster=True,
                            col_cluster=False, cmap='jet')
            reIndexing = hm.dendrogram_row.reordered_ind
            plt.clf()
#             clusterDict, _clusterLables = self.aggClustering(nCluster=self.size,
#                                                              cacheDir=cacheDir,
#                                                              precomputed=precomputed)
            self.treeOrder = [self.seqOrder[i] for i in reIndexing]
        else:
            pass

    def aggClustering(self, nCluster, cacheDir, precomputed=False):
        '''
        Agglomerative clustering

        Performs agglomerative clustering on a matrix of pairwise distances
        between sequences in the alignment being analyzed.

        Parameters:
        -----------
        nCluster: int
            The number of clusters to separate sequences into.
        cacheDir : str
            The path to the directory where the clustering model can be stored
            for access later when identifying different numbers of clusters.
        precomputed: boolean
            Whether or not to use the distances from X as the distances to
            cluster on, the alternative is to compute a new distance matrix
            based on X using Euclidean distance.
        Returns:
        --------
        dict
            A dictionary with cluster number as the key and the sub-alignment
            of this alignment created by that clustering.
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
                                        n_clusters=nCluster, memory=cacheDir,
                                        compute_full_tree=True)
        model.fit(self.distanceMatrix)
        # unique and sorted list of cluster ids e.g. for n_clusters=2, g=[0,1]
        clusterList = model.labels_.tolist()
        ####---------------------------------------#####
        #       Mapping Clusters to Sequences
        ####---------------------------------------#####
        clusterDict = {}
        for i in range(len(clusterList)):
            key = clusterList[i]
            if(key not in clusterDict):
                clusterDict[key] = []
            clusterDict[key].append(self.seqOrder[i])
        end = time()
        print('Performing agglomerative clustering took {} min'.format(
            (end - start) / 60.0))
        return clusterDict, set(clusterList)
#         return subAlignments

    def generateSubAlignment(self, sequenceIDs):
        '''
        Initializes a new alignment which is a subset of the current alignment.

        This method creates a new alignment which contains only sequences
        relating to a set of provided sequence ids.

        Parameters:
        -----------
        sequenceIDs: list
            A list of strings which are sequence identifiers for sequences in
            the current alignment.  Other sequence ids will be skipped.

        Returns:
        --------
        SeqAlignment
            A new SeqAlignment object containing the same fileName, queryID,
            seqLength, and query sequence.  The seqOrder will be updated to
            only those passed in ids which are also in the current alignment,
            preserving their ordering from the current SeqAlignment object.
            The alignmentDict will contain only the subset of sequences
            represented by ids which are present in the new seqOrder.  The size
            is set to the length of the new seqOrder.
        '''
        newAlignment = SeqAlignment(self.fileName, self.queryID.split('_')[1])
        newAlignment.queryID = self.queryID
        newAlignment.querySequence = self.querySequence
        newAlignment.seqLength = self.seqLength
        newAlignment.seqOrder = [x for x in self.seqOrder if x in sequenceIDs]
        newAlignment.alignmentDict = {x: self.alignmentDict[x]
                                      for x in newAlignment.seqOrder}
        newAlignment.size = len(newAlignment.seqOrder)
        newAlignment.treeOrder = [x for x in self.treeOrder
                                  if x in sequenceIDs]
        return newAlignment

    def determineUsablePositions(self, ratio):
        '''
        Determine which positions in the alignment can be used for analysis.

        Parameters:
        -----------
        ratio: float
            The maximum percentage of sequences which can have a gap at a
            specific position before it can no longer be used for analysis.

        Returns:
        --------
        numpy ndarray:
            The positions for which this alignment meets the specified ratio.
        numpy ndarray:
            The number of sequences which do not have gaps at each position in
            the sequence alignment.
        '''
        gaps = (self.alignmentMatrix == 21) * 1.0
        perColumn = np.sum(gaps, axis=0)
        percentGaps = perColumn / self.alignmentMatrix.shape[0]
        usablePositions = np.where(percentGaps <= ratio)[0]
        evidence = (np.ones(self.seqLength) * self.size) - perColumn
        return usablePositions, evidence

    def identifyComparableSequences(self, pos1, pos2):
        '''
        For two specified sequence positions identify the sequences which are
        not gaps in either and return them.

        Parameters:
        -----------
        pos1: int
            First position to check in the sequence alignment.
        pos2: int
            Second position to check in the sequence alignment.

        Returns:
        --------
        np.array
            The column for position 1 which was specified, where the amino acids
            are not gaps in position 1 or position 2.
        np.array
            The column for position 2 which was specified, where the amino acids
            are not gaps in position 1 or position 2.
        np.array
            The array of indices for which the positions were not gapped.  This
            corresponds to the sequences where there were no gaps in the
            alignment at those positions.
        int
            Number of comparable positions, this will be less than or equal to
            the SeqAlignment.size variable.
        '''
        columnI = self.alignmentMatrix[:, pos1]
        indices1 = (columnI != 20.0) * 1
        columnJ = self.alignmentMatrix[:, pos2]
        indices2 = (columnJ != 20.0) * 1
        check = np.where((indices1 + indices2) == 2)[0]
        return (columnI[check], columnJ[check], check, check.shape[0])
