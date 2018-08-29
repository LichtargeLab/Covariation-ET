"""
Created on Aug 17, 2017

@author: daniel
"""
import cPickle as pickle
import os
from time import time

import numpy as np
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa


class PDBReference(object):
    """
    classdocs
    """

    def __init__(self, pdb_file):
        """
        Constructor

        Initiates an instance of the PDBReference class which stores the
        following data:

        file_name: str
            The file name or path to the desired PDB file.
        residue_3d : dict
            A dictionary mapping a residue number to its spatial position in 3D.
        pdb_residue_list : list
            A sorted list of residue numbers from the PDB file.
        residue_pos : dict
            A dictionary mapping residue number to the name of the residue at that
            position.
        seq:
            Sequence of the structure parsed in from the PDB file.
        fasta_to_pdb_mapping : dict
            A structure mapping the index of the positions in the fasta sequence
            which align to positions in the PDB sequence based on a local alignment
            with no mismatches allowed.
        residue_dists : list
            List of minimum distances between residues, sorted by the ordering
            of residues in pdb_residue_list.
        chains : set
            The chains which are present in this proteins structure.
        size : int
            The length of the amino acid chain defining this structure.
        """
        self.file_name = pdb_file
        self.residue_3d = None
        self.pdb_residue_list = None
        self.residue_pos = None
        self.seq = None
        self.fasta_to_pdb_mapping = None
        self.residue_dists = None
        self.chains = None
        self.size = 0
        self.structure = None

    def import_pdb(self, structure_id, save_file=None):
        """
        import_pdb

        This method imports a PDB files information generating a list of lists.
        Each list contains the Amino Acid 3-letter abbreviation, residue number,
        x, y, and z coordinate. This method updates the following class
        variables: residue_3d, pdb_residue_list, residue_pos, and seq.

        Parameters:
        -----------
        save_file: str
            The file path to a previously stored PDB file data structure.
        """
        start = time()
        if (save_file is not None) and os.path.exists(save_file):
            structure, seq, chains, pdb_residue_list, residue_3d, residue_pos = pickle.load(open(save_file, 'rb'))
        else:
            parser = PDBParser(PERMISSIVE=0)  # strict
            parser = PDBParser(PERMISSIVE=1)  # corrective
            structure = parser.get_structure(structure_id, self.file_name)
            seq = {}
            chains = set([])
            pdb_residue_list = {}
            residue_3d = {}
            residue_pos = {}
            for model in structure:
                for chain in model:
                    chains.add(chain.id)
                    pdb_residue_list[chain.id] = []
                    seq[chain.id] = ''
                    residue_3d[chain.id] = {}
                    residue_pos[chain.id] = {}
                    for residue in chain:
                        if is_aa(residue.get_resname(), standard=True):
                            res_name = three_to_one(residue.get_resname())
                        else:
                            res_name = 'X'
                        seq[chain.id] += res_name
                        res_num = residue.get_id()[1]
                        residue_pos[chain.id][res_num] = res_name
                        pdb_residue_list[chain.id].append(res_num)
                        coords = []
                        for atom in residue:
                            coords.append(atom.get_coord())
                        residue_3d[chain.id][res_num] = np.vstack(coords)
            print(seq)
            if save_file is not None:
                pickle.dump((structure, seq, chains, pdb_residue_list, residue_3d, residue_pos), open(save_file, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
        self.structure = structure
        self.chains = chains
        self.seq = seq
        self.pdb_residue_list = pdb_residue_list
        self.residue_3d = residue_3d
        self.residue_pos = residue_pos
        self.size = {chain: len(seq[chain]) for chain in self.chains}
        end = time()
        print('Importing the PDB file took {} min'.format((end - start) / 60.0))


    def map_alignment_to_pdb_seq(self, fasta_seq):
        """
        Map sequence positions between query from the alignment and residues in
        PDB file. This method updates the fasta_to_pdb_mapping class variable.

        Parameters:
        -----------
        fasta_seq: str
            A string providing the amino acid (single letter abbreviations)
            sequence for the protein.
        """
        start = time()
        chain = None
        if len(self.chains) == 1:
            chain, = self.chains
            alignments = pairwise2.align.globalxs(fasta_seq, self.seq[chain],
                                                  -1, 0)
        else:
            alignments = None
            for ch in self.chains:
                curr_align = pairwise2.align.globalxs(fasta_seq, self.seq[ch], -1, 0)
                print curr_align[0][2]
                if (alignments is None) or (alignments[0][2] < curr_align[0][2]):
                    alignments = curr_align
                    chain = ch
        print(format_alignment(*alignments[0]))
        f_counter = 0
        p_counter = 0
        f_to_p_map = {}
        for i in range(len(alignments[0][0])):
            if (alignments[0][0][i] != '-') and (alignments[0][1][i] != '-'):
                f_to_p_map[f_counter] = p_counter
            if alignments[0][0][i] != '-':
                f_counter += 1
            if alignments[0][1][i] != '-':
                p_counter += 1
        end = time()
        print('Mapping query sequence and pdb took {} min'.format(
            (end - start) / 60.0))
        self.fasta_to_pdb_mapping = (chain, f_to_p_map)

    def find_distance(self, save_file=None):
        """
        Find distance

        This code takes in an input of a pdb file and outputs a dictionary with the
        nearest atom distance between two residues. This method updates the
        resideuDists class variables.

        Parameters:
        -----------
        save_file: str
            File name and/or location of file containing a previously computed set
            of distance data for a PDB structure.
        """
        start = time()
        if (save_file is not None) and os.path.exists(save_file):
            pdb_dist = {}
            for chain in self.chains:
                pdb_dist[chain] = np.load(save_file + '_' + chain + '.npz')[chain]
        else:
            pdb_dist = {}
            for chain in self.chains:
                pdb_dist[chain] = np.zeros((self.size[chain], self.size[chain]))
                # Loop over all residues in the pdb
                for i in range(self.size[chain]):
                    # Loop over residues to calculate distance between all residues
                    # i and j
                    for j in range(i + 1, self.size[chain]):
                        # Getting the 3d coordinates for every atom in each residue.
                        # iterating over all pairs to find all distances
                        key1 = self.pdb_residue_list[chain][i]
                        key2 = self.pdb_residue_list[chain][j]
                        # finding the minimum value from the distance array
                        # Making dictionary of all min values indexed by the two residue
                        # names
                        res1 = (self.residue_3d[chain][key2] -
                                self.residue_3d[chain][key1][:, np.newaxis])
                        norms = np.linalg.norm(res1, axis=2)
                        pdb_dist[chain][i, j] = pdb_dist[chain][j, i] = np.min(norms)
            if save_file is not None:
                for chain in self.chains:
                    np.savez(save_file + '_' + chain, chain=pdb_dist[chain])
        end = time()
        print('Computing the distance matrix based on the PDB file took {} min'.format(
            (end - start) / 60.0))
        self.residue_dists = pdb_dist
