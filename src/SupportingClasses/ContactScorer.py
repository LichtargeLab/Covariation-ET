"""
Created on Sep 1, 2018

@author: dmkonecki
"""
from time import time
from Bio import pairwise2
from Bio.pairwise2 import format_alignment


class ContactScorer(object):
    """

    """

    def __init__(self, query_sequence, pdb_sequences):
        """

        """
        self.query_seq = query_sequence
        self.pbd_chains = pdb_sequences
        self.best_chain = None
        self.query_pdb_mapping = None

    def __str__(self):
        """

        :return:
        """
        raise NotImplemented()

    def fit(self):
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
        best_chain = None
        best_alignment = None
        best_score = 0
        for c in self.pdb_chains:
            alignment = pairwise2.align.globalxs(self.query_seq, self.pdb_chains[c], -1, 0)
            if alignment[0][2] > best_score:
                best_score = alignment[0][2]
                best_chain = c
                best_alignment = alignment
        print(format_alignment(*best_alignment[0]))
        print('Best Chain: {}'.format(best_chain))
        f_counter = 0
        p_counter = 0
        f_to_p_map = {}
        for i in range(len(best_alignment[0][0])):
            if (best_alignment[0][0][i] != '-') and (best_alignment[0][1][i] != '-'):
                f_to_p_map[f_counter] = p_counter
            if best_alignment[0][0][i] != '-':
                f_counter += 1
            if best_alignment[0][1][i] != '-':
                p_counter += 1
        end = time()
        print('Mapping query sequence and pdb took {} min'.format((end - start) / 60.0))
        self.best_chain = best_chain
        self.query_pdb_mapping = f_to_p_map

    def measure_distance(self, pdb_structure, method='Any'):
        """

        :param method:
        :return:
        """
        raise NotImplemented()

    def score_auc(self, predictions):
        """

        :param predictions:
        :return:
        """
        raise NotImplemented()

    def score_precision(self, predictions, k):
        """

        :param predictions:
        :param k:
        :return:
        """
        raise NotImplemented()

    def clustering_z_score(self):
        """

        :return:
        """
        raise NotImplemented()