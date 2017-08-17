'''
Created on Aug 17, 2017

@author: daniel
'''
import cPickle as pickle
from time import time
import numpy as np
import os
import re
from IPython import embed


class SeqAlignment(object):
    '''
    classdocs
    '''

    def __init__(self, fileName, queryID):
        '''
        Constructor
        '''
        self.fileName = fileName
        self.queryID = '>query_' + queryID
        self.alignmentDict = None
        self.seqOrder = None
        self.querySequence = None
        self.alignmentMatrix = None
        self.seqLength = None
        self.size = None

    def importAlignment(self, saveFile=None):
        '''
        Import alignments:

        This method imports the alignments into the class and forces all
        non-amino acids to take on the standard gap character "-".

        Parameters:
        -----------
        saveFile: str
            Path to file in which the desired alignment was stored previously.
        Returns:
        --------
        alignment_dict: dict    
            Dictionary which will be used to store alignments from the file.
        seqOrder: list
            List of sequence ids in the order in which they were parsed from the
            alignment file.
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
                    # .replace('.', '-').replace('_', '-')
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

    def removeGaps(self, saveFile=None):
        '''
        Remove Gaps

        Removes all gaps from the query sequence and removes characters at the
        corresponding positions in all other sequences.

        Parameters:
        -----------
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

        Converts an alignment dictionary to a numerical representation.

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

    def distanceMatrix(self, saveFile=None):
        '''
        Distance matrix

        Computes the sequence identity distance between a set of sequences and
        returns a matrix of the pairwise distances.

        Parameters:
        -----------
        saveFile: str
            The path for an .npz file containing distances between sequences in
            the alignment (leave out the .npz as it will be added automatically.
        Returns:
        --------
        matrix
            A symmetric matrix of pairwise distance computed between two sequences
            using the sequence identity metric.
        list
            List of the sequence identifiers in the order in which they appear in
            the matrix.
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
        return valueMatrix

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
            is set to the lenght of the new seqOrder.
        '''
        newAlignment = SeqAlignment(self.fileName, self.queryID.split('_')[1])
        newAlignment.queryID = self.queryID
        newAlignment.querySequence = self.querySequence
        newAlignment.seqLength = self.seqLength
        newAlignment.seqOrder = [x for x in self.seqOrder if x in sequenceIDs]
        newAlignment.alignmentDict = {x: self.alignmentDict[x]
                                      for x in newAlignment.seqOrder}
        newAlignment.size = len(newAlignment.seqOrder)
        return newAlignment
