"""
Created on May 22, 2021

@author: dmkonecki
"""
import numpy as np
import pandas as pd
from time import time
from Evaluation.Scorer import Scorer


class SinglePositionScorer(Scorer):
    """
    SinglePositionScorer

    This class is meant to abstract the process of scoring a set of residue importance predictions.  This is being done
    for two main reasons. First, residue importance predictions are being made with several different methods and so
    their scoring should be performed consistently by another object or function. Second, this class will support
    several scoring methods such as the hypergoemtric test scoring of overlap with key sites as well as methods internal
    to the lab like the Selection Cluster Weighting Z-score derived from prior ET work.

    Attributes:
        query_pdb_mapper (SequencePDBMap): A mapping from the index of the query sequence to the index of the
        pdb chain's sequence for those positions which match according to a pairwise global alignment and vice versa.
        data (panda.DataFrame): A data frame containing a mapping of the sequence, structure, and predictions as well as
        labels to indicate which sequence separation category and top prediction category specific scores belong to.
        This is used for most evaluation methods.
    """

    def __init__(self, query, seq_alignment, pdb_reference, chain=None):
        """
        __init__

        This function initializes a new SinglePositionScorer, it accepts paths to an alignment and if available a pdb
        file to be used in the assessment of residue importance predictions for a given query.

        Args:
            query (str): The name of the query sequence/structure.
            seq_alignment (str/path, SeqAlignment): Path to the alignment being evaluated in this contact scoring
            prediction task or an already initialized SeqAlignment object.
            pdb_reference (str/path, PDBReference): The path to a PDB structure file, or an already initialized
            PDBReference object.
            chain (str): Which chain in the PDB structure to use for comparison and evaluation. If left blank the best
            chain will be identified by aligning the query sequence from seq_alignment against the chains in
            pdb_reference and the closest match will be selected.
        """
        super().__init__(query, seq_alignment, pdb_reference, chain)

    def fit(self):
        """
        Fit

        This function maps sequence positions between the query sequence from the alignment and residues in PDB file.
        If there are multiple chains for the structure in the PDB, the one which matches the query sequence best
        (highest global alignment score) is used and recorded in the seq_pdb_mapper attribute. This method updates the
        query_pdb_mapper and data class attributes.
        """
        super().fit()
        if (self.data is None) or (not self.data.columns.isin(['Seq Pos', 'Seq AA', 'Struct Pos', 'Struct AA']).all()):
            start = time()
            data_dict = {'Seq Pos': [], 'Seq AA': [], 'Struct Pos': [], 'Struct AA': []}
            for pos in range(self.query_pdb_mapper.seq_aln.seq_length):
                char = self.query_pdb_mapper.seq_aln.query_sequence[pos]
                struct_pos, struct_char = self.query_pdb_mapper.map_seq_position_to_pdb_res(pos)
                if struct_pos is None:
                    struct_pos, struct_char = '-', '-'
                data_dict['Seq Pos'].append(pos)
                data_dict['Seq AA'].append(char)
                data_dict['Struct Pos'].append(struct_pos)
                data_dict['Struct AA'].append(struct_char)
            data_df = pd.DataFrame(data_dict)
            self.data = data_df
            end = time()
            print('Constructing internal representation took {} min'.format((end - start) / 60.0))

    def map_predictions_to_pdb(self, ranks, predictions, coverages, threshold=0.5):
        """
        Map Predictions To PDB

        This method accepts a set of predictions and uses the mapping between the query sequence and the best PDB chain
        to extract the comparable predictions.

        Args:
            ranks (np.array): An array of ranks for predicted residue importance with size n where n is the length of
            the query sequence used when initializing the SinglePositionScorer.
            predictions (np.array): An array of prediction scores for residue importance with size n where n is the
            length of the query sequence used when initializing the SinglePositionScorer.
            coverages (np.array): An array of coverage scores for predicted residue importance with size n where n is
            the length of the query sequence used when initializing the SinglePositionScorer.
            threshold (float): The cutoff for coverage scores up to which scores (inclusive) are considered true.
        """
        # Add predictions to the internal data representation, simultaneously mapping them to the structural data.
        if self.data is None:
            self.fit()
        assert len(ranks) == len(self.data) and len(ranks) == len(predictions) and len(ranks) == len(coverages)
        self.data['Rank'] = ranks
        self.data['Score'] = predictions
        self.data['Coverage'] = coverages
        self.data['True Prediction'] = self.data['Coverage'].apply(lambda x: x <= threshold)
