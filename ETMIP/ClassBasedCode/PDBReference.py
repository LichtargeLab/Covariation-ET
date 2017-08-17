'''
Created on Aug 17, 2017

@author: daniel
'''
from Bio import pairwise2
import cPickle as pickle
from time import time
import numpy as np
import os
import re


class PDBReference(object):
    '''
    classdocs
    '''

    def __init__(self, pdbFile):
        '''
        Constructor
        '''
        self.fileName = pdbFile
        self.residue3D = None
        self.pdbResidueList = None
        self.residuePos = None
        self.seq = None
        self.fastaToPDBMapping = None

    def importPDB(self, saveFile=None):
        '''
        importPDB

        This method imports a PDB files information generating a list of lists. Each
        list contains the Amino Acid 3-letter abbreviation, residue number, x, y,
        and z coordinate.

        Parameters:
        -----------
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
        start = time()
        convertAA = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'ASX': 'B',
                     'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLX': 'Z', 'GLY': 'G',
                     'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M',
                     'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
                     'TYR': 'Y', 'VAL': 'V'}
        if((saveFile is not None) and os.path.exists(saveFile)):
            residue3D, pdbResidueList, residuePos, seq = pickle.load(
                open(saveFile, 'rb'))
        else:
            pdbFile = open(self.fileName, 'rb')
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
        self.residue3D = residue3D
        self.pdbResidueList = pdbResidueList
        self.residuePos = residuePos
        self.seq = seq
        end = time()
        print('Importing the PDB file took {} min'.format((end - start) / 60.0))

    def mapAlignmentToPDBSeq(self, fastaSeq):
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
        alignments = pairwise2.align.globalxs(fastaSeq, self.seq, -1, 0)
        from Bio.pairwise2 import format_alignment
        print(format_alignment(*alignments[0]))
        fCounter = 0
        pCounter = 0
        fToPMap = {}
        for i in range(len(alignments[0][0])):
            if((alignments[0][0][i] != '-') and (alignments[0][1][i] != '-')):
                fToPMap[fCounter] = pCounter
            if(alignments[0][0][i] != '-'):
                fCounter += 1
            if(alignments[0][1][i] != '-'):
                pCounter += 1
        self.fastaToPDBMapping = fToPMap

    def findDistance(self, querySequence, saveFile=None):
        '''
        Find distance

        This code takes in an input of a pdb file and outputs a dictionary with the
        nearest atom distance between two residues.

        Parameters:
        -----------
        querySequence: str
            A string providing the amino acid (single letter abbreviations)
            sequence for the protein.
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
            qLen = len(querySequence)
            sortedPDBDist = []
            # Loop over all residues in the pdb
    #         for i in range(len(pdbResidueList)):
            for i in range(qLen):
                # Loop over residues to calculate distance between all residues i
                # and j
                for j in range(i + 1, qLen):
                    # Getting the 3d coordinates for every atom in each residue.
                    # iterating over all pairs to find all distances
                    try:
                        key1 = self.pdbResidueList[self.fastaToPDBMapping[i]]
                        key2 = self.pdbResidueList[self.fastaToPDBMapping[j]]
                        # finding the minimum value from the distance array
                        # Making dictionary of all min values indexed by the two residue
                        # names
                        res1 = (self.residue3D[key2] -
                                self.residue3D[key1][:, np.newaxis])
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
