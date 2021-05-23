"""
Created on May 22, 2021

@author: dmkonecki
"""
import numpy as np
import pandas as pd
from time import time
from math import floor
from SupportingClasses.utils import compute_rank_and_coverage
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

    def _identify_relevant_data(self, n=None, k=None, coverage_cutoff=None):
        """
        Identify Relevant Data

        This method accepts parameters describing a subset of the data stored in the SinglePositionScorer instance and
        subsets it. The order of operations is:
            2. Filter to only predicted residues which map to the best_chain of the provided PDB structure.
            3. Filter to the top predictions as specified by n, k, or coverage_cutoff

        Args:
            k (int): This value should only be specified if n and coverage_cutoff are not specified. This is the number
            that L, the length of the query sequence, will be divided by to give the number of predictions to test.
            n (int): This value should only be specified if k and coverage_cutoff are not specified. This is the number
            of predictions to test.
            coverage_cutoff (float): This value should only be specified if n and k are not specified. This number
            determines how many predictions will be tested by considering predictions up to the point that the specified
            percentage of residues in best_chain are covered. If predictions are tied their residues are added to this
            calculation together, so if many residues are added by considering the next group of predictions, and the
            number of unique residues exceeds the specified percentage, none of the predictions in that group will be
            added.
        Returns:
            pd.DataFrame: A set of predictions which can be scored because they map successfully to the PDB and meet the
            criteria set by n, k, or coverage cutoff for top predictions.
        """
        # Defining for which of the pairs of residues there are both cET-MIp  scores and distance measurements from
        # the PDB Structure.
        if self.data is None:
            self.fit()
        if not pd.Series(['Rank', 'Score', 'Coverage']).isin(self.data.columns).all():
            raise ValueError('Ranks, Scores, and Coverage values must be provided through map_predictions_to_pdb,'
                             'before a specific evaluation can be made.')
        parameter_count = (k is not None) + (n is not None) + (coverage_cutoff is not None)
        if parameter_count > 1:
            raise ValueError('Only one parameter should be set, either: n, k, or coverage_cutoff.')
        unmappable_index = self.data.index[(self.data['Struct Pos'] == '-')]
        final_subset = self.data.drop(unmappable_index)
        final_df = final_subset.sort_values(by='Coverage')
        final_df['Top Predictions'] = final_df['Coverage'].rank(method='dense')
        if coverage_cutoff:
            assert isinstance(coverage_cutoff, float), 'coverage_cutoff must be a float!'
            mappable_res = sorted(self.query_pdb_mapper.query_pdb_mapping.keys())
            top_sequence_ranks = np.ones(self.query_pdb_mapper.seq_aln.seq_length) * np.inf
            residues_visited = set()
            groups = final_df.groupby('Top Predictions')
            for curr_rank in sorted(groups.groups.keys()):
                curr_group = groups.get_group(curr_rank)
                curr_residues = set(curr_group['Seq Pos'])
                new_positions = curr_residues - residues_visited
                top_sequence_ranks[list(new_positions)] = curr_rank
                residues_visited |= new_positions
            top_residue_ranks = top_sequence_ranks[mappable_res]
            assert len(top_residue_ranks) <= len(mappable_res)
            _, top_coverage = compute_rank_and_coverage(seq_length=len(mappable_res), pos_size=1, rank_type='min',
                                                        scores=top_residue_ranks)
            final_df['Pos Coverage'] = final_df['Seq Pos'].apply(lambda x: top_coverage[x])
            n = final_df.loc[(final_df['Pos Coverage'] > coverage_cutoff), 'Top Predictions'].min() - 1
            if np.isnan(n):
                n = final_df['Top Predictions'].max()
                if np.isnan(n):
                    n = 0
        elif k:
            n = int(floor(self.query_pdb_mapper.seq_aln.seq_length / float(k)))
        elif n is None:
            n = self.data.shape[0]
        else:
            pass
        ind = final_df['Top Predictions'] <= n
        final_df = final_df.loc[ind, :]
        return final_df
