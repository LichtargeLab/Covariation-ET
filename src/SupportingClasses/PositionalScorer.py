"""
Created on July 12, 2019

@author: Daniel Konecki
"""
import numpy as np
from scipy.stats import entropy

ambiguous_metrics = {'identity', 'plain_entropy'}

single_only_metrics = set()

pair_only_metrics = set()


class PositionalScorer(object):
    """
    This class is meant to serve as an interchangeable component of the Trace.trace function. It produces group and
    rank level scores using different metrics.

    Attributes:
        sequence_length (int): The length of the sequence being analysed.
        position_size (int): The size of the the positions being analyzed (expecting 1 single position scores or 2 pair
        position scores).
        dimensions (tuple):
        metric (str): Which metric to use when computing group and rank level scores. Currently available metrics:
            identity: Whether a position is fully conserved within a group and across all groups (resulting in a score
            of 0), or not (resulting in a score of 1).
    """

    def __init__(self, seq_length, pos_size, metric):
        """
        PositionalScorer initialization function

        The function to initialize an instance of the PositionalScorer

        Args:
            seq_length (int): The length of the sequence being analysed.
            pos_size (int): The size of the the positions being analyzed (expecting 1 single position scores or 2 pair
            position scores).
            metric (str): Which metric to use when computing group and rank level scores.
        """
        self.sequence_length = seq_length
        self.position_size = pos_size
        self.dimensions = tuple([self.sequence_length] * pos_size)
        if (self.position_size == 1) and (metric not in ambiguous_metrics | single_only_metrics):
            raise ValueError('Provided metric: {} not available for pos_size: {}, please select from:\n{}'.format(
                metric, self.position_size, ', '.join(list(ambiguous_metrics | single_only_metrics))))
        elif (self.position_size == 2) and (metric not in ambiguous_metrics | pair_only_metrics):
            raise ValueError('Provided metric: {} not available for pos_size: {}, please select from:\n{}'.format(
                metric, self.position_size, ', '.join(list(ambiguous_metrics | pair_only_metrics))))
        self.metric = metric

    def score_group(self, freq_table):
        """
        Score Group

        This function is intended to generate a score vector/matrix/tensor for a single group. It does so by scoring
        each position characterized in a FrequencyTable and using those scores to fill in a properly dimensioned
        vector/matrix/tensor.

        Args:
            freq_table (FrequencyTable): The table characterizing the character counts at each position in an alignment.
        Returns:
            np.array: A properly dimensioned vector/matrix/array containing the scores for each position in an alignment
            as determined by the specified metric.
        """
        scoring_functions = {'identity': group_identity_score, 'plain_entropy': group_plain_entropy_score}
        scores = np.zeros(self.dimensions)
        for pos in freq_table.get_positions():
            score = scoring_functions[self.metric](freq_table, pos)
            if self.position_size == 1:
                scores[pos] = score
            else:
                scores[pos[0], pos[1]] = score
        return scores

    def score_rank(self, score_tensor):
        """
        Score Rank

        This function is intended to generate a score vector/matrix/tensor for a single rank. It acts as a switch
        statement, calling the correct method to compute final scores for a set of group of scores.

        Args:
            score_tensor (np.array): A matrix/tensor with shape (n, s1, ..., sp) where n is the number of groups at the
            rank to be scored, s is the sequence length for the alignment being characterized, and p is the
            position_size specified at initialization. For example a score_tensor for rank 3 and an alignment with
            sequence length 20, scoring pairs of positions will have shape (3, 20, 20). This can be achieved by stacking
            all group scores at the desired rank along axis 0.
        Returns:
            np.array: A properly dimensioned vector/matrix/array containing the scores for each position in an alignment
            as determined by the specified metric.
        """
        scoring_functions = {'identity': rank_identity_score, 'plain_entropy': rank_plain_entropy_score}
        scores = scoring_functions[self.metric](score_tensor)
        return scores


def group_identity_score(freq_table, pos):
    """
    Group Identity Score

    This computes the identity score for a position in a characterized alignment. The position is considered conserved,
    resulting in a score of 0, if only one character from the valid alphabet is observed at that position. The position
    is considered variable, resulting in a score of 1, if multiple characters from the valid alphabet are observed at
    that position.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the identity score.
        pos (int/tuple): The position in the sequence/list of pairs for which to compute the identity score.
    Returns:
        int: 0 if the position is conserved (i.e. only a single nucleic/amino acid has been observed at that position),
        and 1 otherwise (i.e. if multiple nucleic/amino acids have been observed at that position).
    """
    if len(freq_table.get_chars(pos=pos)) == 1:
        return 0
    else:
        return 1


def rank_identity_score(score_matrix):
    """
    Rank Identity Score

    This computes the final rank specific identity score for all positions in a characterized alignment. A position is
    identical/conserved only if it was conserved, score 0, in all groups. In any other case the position is not
    considered conserved.

    Args:
        score_matrix (np.array):
    Returns:
        np.array: A score vector/matrix for all positions in the alignment with binary values to show whether a position
        is conserved in every group at the current rank (0) or if it is variable in at least one group (1).
    """
    cumulative_scores = np.sum(score_matrix, axis=0)
    rank_scores = 1 * (cumulative_scores != 0)
    return rank_scores


def group_plain_entropy_score(freq_table, pos):
    """
    Group Plain Entropy Score

    This function computes the plain entropy for a given group. In this case entropy is computed using the formula:
    S = -Sum(pk * log(pk), axis=0) as given by scipy.stats.entropy. Here k is a character in the valid alphabet while
    pk is the frequency of that character in the alignment at the current position.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the plain entropy
        score.
        pos (int/tuple): The position in the sequence/list of pairs for which to compute the plain entropy score.
    Returns:
        float: The plain entropy for the specified position in the FrequencyTable.
    """
    freq_table.compute_frequencies()
    positional_frequencies = freq_table.get_frequency_array(pos=pos)
    positional_entropy = entropy(positional_frequencies)
    return positional_entropy


def rank_plain_entropy_score(score_matrix):
    """
    Rank Identity Score

    This computes the final rank specific plain entropy score for all positions in a characterized alignment. The plain
    identity is normalized by rank.

    Args:
        score_matrix (np.array):
    Returns:
        np.array: A score vector/matrix for all positions in the alignment with float values to show whether a position
        is conserved in evert group at the current rank (0.0) or if it is variable in any of the groups (> 0.0).
    """
    rank = score_matrix.shape[0]
    weight = 1.0 / rank
    cumulative_scores = np.sum(score_matrix, axis=0)
    rank_scores = weight * cumulative_scores
    return rank_scores
