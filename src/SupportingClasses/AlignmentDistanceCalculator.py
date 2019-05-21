"""
Created on May 15, 2019

@author: Daniel Konecki
"""
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceMatrix
from Bio.Align import MultipleSeqAlignment
from itertools import combinations
import numpy as np


class AlignmentDistanceCalculator(DistanceCalculator):
    """
    This class is meant to perform the calculation of sequence distance between sequences in an alignment. This
    information can then be used for other uses such as constructing a phylogenetic tree.

    Attributes:
        aln_type (str):
        alphabet (list):
        model (str):
        mapping (dict):
        substitution_matrix (np.ndarray):
    """

    def __init__(self, protein=True, model='identity', skip_letters=None):
        """
        Args:
            protein (Boolean):
        Returns:
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
        alphabet_mapping = {char: i for i, char in enumerate(self.alphabet)}
        gap_map = {char: len(self.alphabet) for char in self.gap_characters}
        alphabet_mapping.update(gap_map)
        skip_map = {char: len(self.alphabet) + 1 for char in self.skip_letters}
        alphabet_mapping.update(skip_map)
        return alphabet_mapping

    def _update_scoring_matrix(self):
        if self.model == 'identity':
            substitution_matrix = self._build_identity_scoring_matrix()
        elif self.aln_type == 'dna' or self.aln_type == 'protein':
            substitution_matrix = self._rebuild_scoring_matrix()
        else:
            raise ValueError('Unexpected combination of variables found when building substitution matrix')
        return substitution_matrix

    def _build_identity_scoring_matrix(self):
        substitution_matrix = np.eye(len(self.alphabet) + 2)
        substitution_matrix[len(self.alphabet) + 1, len(self.alphabet) + 1] = 0
        return substitution_matrix

    def _rebuild_scoring_matrix(self):
        scoring_matrix = np.array(self.scoring_matrix)
        substitution_matrix = np.insert(scoring_matrix, obj=scoring_matrix.shape[0], values=0, axis=0)
        substitution_matrix = np.insert(substitution_matrix, obj=scoring_matrix.shape[1], values=0, axis=1)
        substitution_matrix = np.insert(substitution_matrix, obj=substitution_matrix.shape[0], values=0, axis=0)
        substitution_matrix = np.insert(substitution_matrix, obj=substitution_matrix.shape[1], values=0, axis=1)
        return substitution_matrix

    def _convert_seq_to_numeric(self, seq):
        # Convert a SeqRecord sequence to a numerical representation (using indices in scoring_matrix)
        numeric = [self.mapping[char] for char in seq]
        return np.array(numeric)

    def _pairwise(self, seq1, seq2):
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
        names = [s.id for s in msa]
        plain_identity = DistanceMatrix(names)
        psuedo_identity = DistanceMatrix(names)
        # Set cutoff for scoring matrix "identity"
        scoring_matrix_tril = np.tril(self.scoring_matrix)
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
        non_gap_lengths = []
        non_gap_pos = []
        num_repr = []
        for i in range(len(msa)):
            if i > len(num_repr):
                # Convert seq i in the msa to a numerical representation (indices in scoring_matrix)
                num_repr.append(self._convert_seq_to_numeric(msa[i]))
                # Find all positions which are not gaps or skip_letters in seq1 and the resulting sequence length
                non_gap_pos.append(num_repr[i] < len(self.alphabet))
                non_gap_lengths.append(np.sum(non_gap_pos[i]))
            for j in range(i + 1, len(msa)):
                if j > len(num_repr):
                    # Convert seq j in the msa to a numerical representation (indices used in scoring_matrix)
                    num_repr.append(self._convert_seq_to_numeric(msa[j]))
                    # Find all positions which are not skip_letters in seq2
                    non_gap_pos.append(num_repr[j] < len(self.alphabet))
                    non_gap_lengths.append(np.sum(non_gap_pos[j]))
                # Determine positions that are not gaps or skip_letters in either sequence
                combined_non_gap_pos = non_gap_pos[i] & non_gap_pos[j]
                # Subset the two sequences to only the positions which are not gaps or skip_letters
                final_seq1 = num_repr[i][combined_non_gap_pos]
                final_seq2 = num_repr[j][combined_non_gap_pos]
                # Count the number of positions which are identical between the two sequences
                identity_count = np.sum((final_seq1 - final_seq2) == 0)
                # Retrieve the scoring_matrix scores for the two sequences
                scores = self.scoring_matrix[final_seq1, final_seq2]
                # Find which scores pass the threshold (psuedo-identity) and count them
                passing_scores = scores >= threshold
                scoring_matrix_count = np.sum(passing_scores)
                # Determine which sequence was the shorter of the two
                seq_length = min(non_gap_lengths[i], non_gap_lengths[j])
                # Compute the plain or psuedo identity score using the minimum sequence length
                plain_identity[names[i], names[j]] = 1 - (identity_count / float(seq_length))
                psuedo_identity[names[i], names[j]] = 1 - (scoring_matrix_count / float(seq_length))
        return plain_identity, psuedo_identity
