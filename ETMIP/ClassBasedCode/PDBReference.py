'''
Created on Aug 17, 2017

@author: daniel
'''
from Bio.pairwise2 import format_alignment
from Bio import pairwise2
import cPickle as pickle
from time import time
import numpy as np
import os
import re
from IPython import embed


class PDBReference(object):
    '''
    classdocs
    '''

    def __init__(self, pdbFile):
        '''
        Constructor

        Initiates an instance of the PDBReference class which stores the
        following data:

        fileName: str
            The file name or path to the desired PDB file.
        residue3D : dict
            A dictionary mapping a residue number to its spatial position in 3D.
        pdbResidueList : list
            A sorted list of residue numbers from the PDB file.
        residuePos : dict
            A dictionary mapping residue number to the name of the residue at that
            position.
        seq:
            Sequence of the structure parsed in from the PDB file.
        fastaToPDBMapping : dict
            A structure mapping the index of the positions in the fasta sequence
            which align to positions in the PDB sequence based on a local alignment
            with no mismatches allowed.
        residueDists : list
            List of minimum distances between residues, sorted by the ordering
            of residues in pdbResidueList.
        chains : set
            The chains which are present in this proteins structure.
        size : int
            The length of the amino acid chain defining this structure.
        '''
        self.fileName = pdbFile
        self.residue3D = None
        self.pdbResidueList = None
        self.residuePos = None
        self.seq = None
        self.fastaToPDBMapping = None
        self.residueDists = None
        self.chains = None
        self.size = 0

    def importPDB(self, saveFile=None):
        '''
        importPDB

        This method imports a PDB files information generating a list of lists.
        Each list contains the Amino Acid 3-letter abbreviation, residue number,
        x, y, and z coordinate. This method updates the following class
        variables: residue3D, pdbResidueList, residuePos, and seq.

        Parameters:
        -----------
        saveFile: str
            The file path to a previously stored PDB file data structure.
        '''
        start = time()
        convertAA = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'ASX': 'B',
                     'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLX': 'Z', 'GLY': 'G',
                     'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M',
                     'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
                     'TYR': 'Y', 'VAL': 'V'}
        if((saveFile is not None) and os.path.exists(saveFile)):
            residue3D, pdbResidueList, residuePos, seq, chains = pickle.load(
                open(saveFile, 'rb'))
        else:
            pdbFile = open(self.fileName, 'rb')
            chains = set()
            residue3D = {}
            pdbResidueList = {}
            residuePos = {}
            seq = {}
            prevRes = None
            prevChain = None
            pdbPattern = r'ATOM\s*(\d+)\s*(\w*)\s*([A-Z]{3})\s*([A-Z])\s*(\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*([A-Z])'
            for line in pdbFile:

                res = re.match(pdbPattern, line)
                if not res:
                    continue
                resName = convertAA[res.group(3)]
                resChain = res.group(4)
                resNum = int(res.group(5))
                resAtomList = np.asarray([float(res.group(6)),
                                          float(res.group(7)),
                                          float(res.group(8))])
                # New conditional to manage prev res and prev chain
                if(prevRes != resNum):
                    if(prevChain != resChain):
                        if(prevChain is not None):
                            residue3D[prevChain][prevRes] = np.vstack(
                                residue3D[prevChain][prevRes])
                        prevChain = resChain
                        chains.add(resChain)
                        residue3D[resChain] = {}
                        pdbResidueList[resChain] = []
                        residuePos[resChain] = {}
                        seq[resChain] = []
                    elif(prevRes is not None):
                        residue3D[resChain][prevRes] = np.vstack(
                            residue3D[resChain][prevRes])
                    else:
                        pass
                    prevRes = resNum
                    residue3D[resChain][resNum] = [resAtomList]
                    pdbResidueList[resChain].append(resNum)
                    residuePos[resChain][resNum] = resName
                    seq[resChain].append(resName)
                else:
                    residue3D[resChain][resNum].append(resAtomList)
            residue3D[prevChain][prevRes] = np.vstack(
                residue3D[prevChain][prevRes])
            # list of sorted residues - necessary for those where res1 is not 1
            for chain in chains:
                pdbResidueList[chain] = sorted(pdbResidueList[chain])
                seq[chain] = ''.join(seq[chain])
            pdbFile.close()
            if(saveFile is not None):
                pickle.dump((residue3D, pdbResidueList, residuePos, seq, chains),
                            open(saveFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        self.chains = chains
        self.residue3D = residue3D
        self.pdbResidueList = pdbResidueList
        self.residuePos = residuePos
        self.seq = seq
        self.size = {chain: len(seq[chain]) for chain in self.chains}
        end = time()
        print('Importing the PDB file took {} min'.format((end - start) / 60.0))

    def mapAlignmentToPDBSeq(self, fastaSeq):
        '''
        Map sequence positions between query from the alignment and residues in
        PDB file. This method updates the fastaToPDBMapping class variable.

        Parameters:
        -----------
        fastaSeq: str
            A string providing the amino acid (single letter abbreviations)
            sequence for the protein.
        pdbSeq: str
            A string providing the amino acid (single letter abbreviations)
            sequence for the protein.
        '''
        start = time()
        chain = None
        if(len(self.chains) == 1):
            chain, = self.chains
            alignments = pairwise2.align.globalxs(fastaSeq, self.seq[chain],
                                                  -1, 0)
        else:
            alignments = None
            for ch in self.chains:
                currAlign = pairwise2.align.globalxs(fastaSeq, self.seq[ch],
                                                     -1, 0)
                print currAlign[0][2]
                if((alignments is None) or
                   (alignments[0][2] < currAlign[0][2])):
                    alignments = currAlign
                    chain = ch
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
        end = time()
        print('Mapping query sequence and pdb took {} min'.format(
            (end - start) / 60.0))
        self.fastaToPDBMapping = (chain, fToPMap)

    def findDistance(self, saveFile=None):
        '''
        Find distance

        This code takes in an input of a pdb file and outputs a dictionary with the
        nearest atom distance between two residues. This method updates the
        resideuDists class variables.

        Parameters:
        -----------
        saveFile: str
            File name and/or location of file containing a previously computed set
            of distance data for a PDB structure.
        --------
        '''
        start = time()
        if((saveFile is not None) and os.path.exists(saveFile)):
            pdbDist = {}
            for chain in self.chains:
                pdbDist[chain] = np.load(
                    saveFile + '_' + chain + '.npz')[chain]
        else:
            pdbDist = {}
            for chain in self.chains:
                pdbDist[chain] = np.zeros((self.size[chain], self.size[chain]))
                # Loop over all residues in the pdb
                for i in range(self.size[chain]):
                    # Loop over residues to calculate distance between all residues
                    # i and j
                    for j in range(i + 1, self.size[chain]):
                        # Getting the 3d coordinates for every atom in each residue.
                        # iterating over all pairs to find all distances
                        key1 = self.pdbResidueList[chain][i]
                        key2 = self.pdbResidueList[chain][j]
                        # finding the minimum value from the distance array
                        # Making dictionary of all min values indexed by the two residue
                        # names
                        res1 = (self.residue3D[chain][key2] -
                                self.residue3D[chain][key1][:, np.newaxis])
                        norms = np.linalg.norm(res1, axis=2)
                        pdbDist[chain][i, j] = pdbDist[chain][j, i] = np.min(
                            norms)
            if(saveFile is not None):
                for chain in self.chains:
                    np.savez(saveFile + '_' + chain, chain=pdbDist[chain])
        end = time()
        print('Computing the distance matrix based on the PDB file took {} min'.format(
            (end - start) / 60.0))
        self.residueDists = pdbDist
