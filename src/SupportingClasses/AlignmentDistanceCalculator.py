"""
Created on May 15, 2019

@author: Daniel Konecki
"""
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceMatrix
from Bio.Align import MultipleSeqAlignment
from itertools import combinations
import pandas as pd
import numpy as np


class AlignmentDistanceCalculator(DistanceCalculator):
    """
    This class is meant to perform the calculation of sequence distance between sequences in an alignment. This
    information can then be used for other uses such as constructing a phylogenetic tree.

    Attributes:
        aln_type (str): Whether this is a protein or DNA alignment.
        alphabet (list): The corresponding amino or nucleic acid alphabet.
        model (str): The distance/substitution model to use when calculating sequence distance.
        mapping (dict): A mapping from the character alphabet to their indices (used when computing identity distance or
        referencing subsitution scores).
        scoring_matrix (np.ndarray): The corresponding scoring/substitution matrix for the chosen model.
    """

    def __init__(self, protein=True, model='identity', skip_letters=None):
        """
        Args:
            protein (bool): Whether or not this calculator will be used on protein alignments or not.
            model (str): The distance/substitution model to use when calculating sequence distance.
            skip_letters (list): Which characters to skip when scoring sequences in the alignment.
        """
        if protein:
            self.aln_type = 'protein'
            self.alphabet = DistanceCalculator.protein_alphabet
            possible_models = DistanceCalculator.protein_models
        else:
            self.aln_type = 'dna'
            self.alphabet = DistanceCalculator.dna_alphabet
            possible_models = DistanceCalculator.dna_models
        possible_models.append('identity')
        if model not in possible_models:
            raise ValueError("Model '{}' not in list of possible models:\n{}".format(model, ', '.join(possible_models)))
        else:
            self.model = model
        super(AlignmentDistanceCalculator, self).__init__(model=model, skip_letters=skip_letters)
        self.gap_characters = list(set(['*', '-', '.']) - set(self.skip_letters))
        self.mapping = self._build_mapping()
        self.scoring_matrix = self._update_scoring_matrix()

    def _build_mapping(self):
        """
        Build Mapping

        Constructs a dictionary mapping characters in the given alphabet to their position in the alphabet (which
        corresponds to their index in substitution matrices). It also maps gap characters and skip letters to positions
        outside of that range.

        Returns:
            dict: Dictionary mapping a character to a number corresponding to its position in the alphabet or in the
            scoring/substitution matrix.
        """
        alphabet_mapping = {char: i for i, char in enumerate(self.alphabet)}
        gap_map = {char: len(self.alphabet) for char in self.gap_characters}
        alphabet_mapping.update(gap_map)
        skip_map = {char: len(self.alphabet) + 1 for char in self.skip_letters}
        alphabet_mapping.update(skip_map)
        return alphabet_mapping

    def _update_scoring_matrix(self):
        """
        Update Scoring Matrix

        This function acts as a switch statement to generate the correct scoring/substitution matrix based on whether
        the calculator will be used for proteins or DNA and the model specified.

        Return:
             numpy.array: The substitution/scoring matrix.
        """
        if self.model == 'identity':
            substitution_matrix = self._build_identity_scoring_matrix()
        elif self.aln_type == 'dna' or self.aln_type == 'protein':
            substitution_matrix = self._rebuild_scoring_matrix()
        else:
            raise ValueError('Unexpected combination of variables found when building substitution matrix')
        return substitution_matrix

    def _build_identity_scoring_matrix(self):
        """
        Build Identity Scoring Matrix

        This function builds the scoring matrix if the specified model is identity.

        Return:
             numpy.array: The identity scoring matrix.
        """
        substitution_matrix = np.eye(len(self.alphabet) + 2)
        substitution_matrix[len(self.alphabet) + 1, len(self.alphabet) + 1] = 0
        return substitution_matrix

    def _rebuild_scoring_matrix(self):
        """
        Rebuild Scoring Matrix

        This function converts a Bio.Phylo.TreeConstruction.DistanceMatrix to a numpy array and then adds additional
        columns/rows for the gap and skip letter characters.

        Returns:
            numpy.array: The substitution matrix for the specified model.
        """
        scoring_matrix = np.array(self.scoring_matrix)
        substitution_matrix = np.insert(scoring_matrix, obj=scoring_matrix.shape[0], values=0, axis=0)
        substitution_matrix = np.insert(substitution_matrix, obj=scoring_matrix.shape[1], values=0, axis=1)
        substitution_matrix = np.insert(substitution_matrix, obj=substitution_matrix.shape[0], values=0, axis=0)
        substitution_matrix = np.insert(substitution_matrix, obj=substitution_matrix.shape[1], values=0, axis=1)
        return substitution_matrix

    def _convert_seq_to_numeric(self, seq):
        """
        Convert Seq To Numeric

        This function uses an alphabet mapping (see _build_mapping) to convert a sequence to a 1D array of integers.

        Args:
            seq (Bio.Seq|Bio.SeqRecord|str): A protein or DNA sequence.
        Return:
            numpy.array: A 1D array containing the numerical representation of the passed in sequence.
        """
        # Convert a SeqRecord sequence to a numerical representation (using indices in scoring_matrix)
        numeric = [self.mapping[char] for char in seq]
        return np.array(numeric)

    def _pairwise(self, seq1, seq2):
        """
        Pairwise

        This function scores two sequences by converting them via a mapping table, identifying positions to skip,
        retrieving the substitution scores, and calculating the final distance between the two sequences.

        Args:
            seq1 (Bio.Seq|Bio.SeqRecord|str): A protein or DNA sequence to score against seq2.
            seq2 (Bio.Seq|Bio.SeqRecord|str): A protein or DNA sequence to score against seq1.
        Returns:
            float: The distance between the two sequences.
        """
        num_seq1 = self._convert_seq_to_numeric(seq1)  # Convert seq1 to its indices in the scoring_matrix
        num_seq2 = self._convert_seq_to_numeric(seq2)  # Convert seq2 to its indices in the scoring_matrix
        non_gap_pos1 = num_seq1 < len(self.alphabet) + 1  # Find all positions which are not skip_letters in seq1
        non_gap_pos2 = num_seq2 < len(self.alphabet) + 1  # Find all positions which are not skip_letters in seq2
        combined_non_gap_pos = non_gap_pos1 & non_gap_pos2  # Determine positions that are not skip_letters in either
        # Retrieve scores from scoring_matrix for all positions in the two sequences
        ij_scores = self.scoring_matrix[num_seq1[combined_non_gap_pos], num_seq2[combined_non_gap_pos]]
        # Retrieve all identity (max) scores for seq1
        ii_scores = self.scoring_matrix[num_seq1[combined_non_gap_pos], num_seq1[combined_non_gap_pos]]
        # Retrieve all identity (max) scores for seq2
        jj_scores = self.scoring_matrix[num_seq2[combined_non_gap_pos], num_seq2[combined_non_gap_pos]]
        # Compute the sum of each of the scores
        score = np.sum(ij_scores)
        max_score1 = np.sum(ii_scores)
        max_score2 = np.sum(jj_scores)
        max_score = max(max_score1, max_score2)  # Determine if seq1 or seq2 has the higher possible score
        # Compute the final score
        if max_score == 0:
            final_score = 1
        else:
            final_score = 1 - (score * 1.0 / max_score)
        return final_score

    def get_identity_distance(self, msa):
        """
        Get Identity Distance

        Compute the identity distance between the sequences in the provided alignment.

        Args:
            msa (Bio.Align.MultipleSeqAlignment): The alignment for which to calculate identity distances.
        Returns:
            Bio.Phylo.TreeConstruction.DistanceMatrix: The identity distance matrix for the alignment.
        """
        names = [s.id for s in msa]
        dm = DistanceMatrix(names)
        numerical_alignment = np.vstack([self._convert_seq_to_numeric(seq) for seq in msa])
        msa_size = len(msa)
        for i in range(msa_size):
            check = numerical_alignment - numerical_alignment[i]
            identity_counts = np.sum(check == 0, axis=1)
            fraction_identity = identity_counts / float(numerical_alignment.shape[1])
            distances = 1 - fraction_identity
            for j in range(i + 1, msa_size):
                dm[names[i], names[j]] = distances[j]
        return dm

    def get_scoring_matrix_distance(self, msa):
        """
        Get Scoring Matrix Distance

        Compute the distance between the sequences in the provided alignment based on the specified distance model.

        Args:
            msa (Bio.Align.MultipleSeqAlignment): The alignment for which to calculate distances.
        Returns:
            Bio.Phylo.TreeConstruction.DistanceMatrix: The distance matrix for the alignment.
        """
        names = [s.id for s in msa]
        dm = DistanceMatrix(names)
        for seq1, seq2 in combinations(msa, 2):
            dm[seq1.id, seq2.id] = self._pairwise(seq1, seq2)
        return dm

    def get_distance(self, msa):
        """Return a DistanceMatrix for MSA object.

        :Parameters:
            msa : MultipleSeqAlignment
                DNA or Protein multiple sequence alignment.

        """
        if not isinstance(msa, MultipleSeqAlignment):
            raise TypeError("Must provide a MultipleSeqAlignment object.")
        if self.model == 'identity':
            dm = self.get_identity_distance(msa)
        else:
            dm = self.get_scoring_matrix_distance(msa)
        return dm

    def get_et_distance(self, msa):
        """
        Get ET Distance

        Calculates the sequence similarity using identity and substitution scoring metrics (this mirrors the previous
        implmentations used by ETC in the lab).

        Args:
            msa (Bio.Align.MultipleSeqAlignment): The alignment for which to calculate distances.
        Returns:
            Bio.Phylo.TreeConstruction.DistanceMatrix: The identity based seqeunce similarity distance matrix for the
            alignment.
            Bio.Phylo.TreeConstruction.DistanceMatrix: The substitution matrix based distance matrix for the alignment.
            pandas.DataFrame: A DataFrame with intermediate values for the distance calculation.
            float: The threshold used to determine the cutoff for similarity using the substitution matrix.
        """
        names = [s.id for s in msa]
        plain_identity = DistanceMatrix(names)
        psuedo_identity = DistanceMatrix(names)
        # Set cutoff for scoring matrix "identity"
        scoring_matrix_tril = np.tril(self.scoring_matrix, k=-1)
        positive_scores = scoring_matrix_tril[scoring_matrix_tril > 0]
        count = positive_scores.shape[0]
        if count > 0:
            sum = np.sum(positive_scores)
            average = float(sum) / count
            if average < 1.0:
                threshold = 1.0
            else:
                threshold = np.floor(average + 0.5)
        else:
            threshold = 1
        # Compute the non-gap length of the sequences
        data_dict = {'Seq1': [], 'Seq2': [], 'Min_Seq_Length': [], 'Id_Count': [], 'Threshold_Count': []}
        seq_conversion = {}
        for i in range(len(msa)):
            id1 = msa[i].id
            if id1 not in seq_conversion:
                # Convert seq i in the msa to a numerical representation (indices in scoring_matrix)
                num_repr1 = self._convert_seq_to_numeric(msa[i])
                # Find all positions which are not gaps or skip_letters in seq1 and the resulting sequence length
                non_gap_pos1 = num_repr1 < len(self.alphabet)
                non_gap_length1 = np.sum(non_gap_pos1)
                seq_conversion[id1] = {'non_gap_length': non_gap_length1, 'non_gap_pos': non_gap_pos1,
                                       'num_repr': num_repr1}
            for j in range(i + 1, len(msa)):
                id2 = msa[j].id
                if id2 not in seq_conversion:
                    # Convert seq j in the msa to a numerical representation (indices used in scoring_matrix)
                    num_repr2 = self._convert_seq_to_numeric(msa[j])
                    # Find all positions which are not skip_letters in seq2
                    non_gap_pos2 = num_repr2 < len(self.alphabet)
                    non_gap_length2 = np.sum(non_gap_pos2)
                    seq_conversion[id2] = {'non_gap_length': non_gap_length2, 'non_gap_pos': non_gap_pos2,
                                           'num_repr': num_repr2}
                # Determine positions that are not gaps or skip_letters in either sequence
                combined_non_gap_pos = seq_conversion[id1]['non_gap_pos'] & seq_conversion[id2]['non_gap_pos']
                # Subset the two sequences to only the positions which are not gaps or skip_letters
                final_seq1 = seq_conversion[id1]['num_repr'][combined_non_gap_pos]
                final_seq2 = seq_conversion[id2]['num_repr'][combined_non_gap_pos]
                # Count the number of positions which are identical between the two sequences
                identity_count = np.sum((final_seq1 - final_seq2) == 0)
                # Retrieve the scoring_matrix scores for the two sequences
                scores = self.scoring_matrix[final_seq1, final_seq2]
                # Find which scores pass the threshold (psuedo-identity) and count them
                passing_scores = scores >= threshold
                scoring_matrix_count = np.sum(passing_scores)
                # Determine which sequence was the shorter of the two
                seq_length = min(seq_conversion[id1]['non_gap_length'], seq_conversion[id2]['non_gap_length'])
                # Compute the plain or psuedo identity score using the minimum sequence length
                plain_identity[names[i], names[j]] = identity_count / float(seq_length)
                psuedo_identity[names[i], names[j]] = 1 - (scoring_matrix_count / float(seq_length))
                data_dict['Seq1'].append(id1)
                data_dict['Seq2'].append(id2)
                data_dict['Min_Seq_Length'].append(seq_length)
                data_dict['Id_Count'].append(identity_count)
                data_dict['Threshold_Count'].append(scoring_matrix_count)
        return plain_identity, psuedo_identity, pd.DataFrame(data_dict), threshold
