'''
Created on Aug 21, 2017

@author: dmkonecki
'''
import os
import numpy as np
from time import time
from sklearn.metrics import mutual_info_score


class ETMIPC(object):
    '''
    classdocs
    '''

    def __init__(self, alignment, dists, clusters, pdb, outputDir, processes):
        '''
        Constructor
        '''
        self.alignment = alignment
        self.dists = dists
        self.clusters = clusters
        self.pdb = pdb
        self.outputDir = outputDir
        self.processes = processes
        self.wholeMIPMatrix = None
        self.resultTimes = {c: 0.0 for c in self.clusters}
        self.rawScores = {c: np.zeros((c, self.alignment.seqLength,
                                       self.alignment.seqLength))
                          for c in self.clusters}

    def wholeAnalysis(self, alignment, saveFile=None):
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
            loadedData = np.load(saveFile + '.npz')
            mipMatrix = loadedData['wholeMIP']
    #         evidenceMatrix = loadedData['evidence']
        else:
            overallMMI = 0.0
            # generate an MI matrix for each cluster
            miMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
    #         evidenceMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
            # Vector of 1 column
            MMI = np.zeros(alignment.seqLength)
            apcMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
            mipMatrix = np.zeros((alignment.seqLength, alignment.seqLength))
            # Generate MI matrix from alignment2Num matrix, the MMI matrix,
            # and overallMMI
            for i in range(alignment.seqLength):
                for j in range(i + 1, alignment.seqLength):
                    colI = alignment.alignmentMatrix[:, i]
                    colJ = alignment.alignmentMatrix[:, j]
                    currMIS = mutual_info_score(colI, colJ, contingency=None)
    #                 colI, colJ, _pos, ev, r = alignment.identifyComparableSequences(
    #                     i, j)
    #                 if(ev == 0):
    #                     #                 if((ev == 0) or (r >= 0.8)):
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
    #                 evidenceMatrix[i, j] = evidenceMatrix[j, i] = ev
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
                # , evidence=evidenceMatrix)
                np.savez(saveFile, wholeMIP=mipMatrix)
        end = time.time()
        print('Whole analysis took {} min'.format((end - start) / 60.0))
        return mipMatrix
    #     return mipMatrix, evidenceMatrix
