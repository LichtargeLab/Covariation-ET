"""
Created on July 12, 2019

@author: Daniel Konecki
"""
import numpy as np

ambiguous_metrics = {'identity'}

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
        if pos_size == 1:
            self.dimensions = self.sequence_length
        else:
            self.dimensions = (self.sequence_length, self.sequence_length)
        # self.dimensions = tuple([self.sequence_length] * pos_size)
        if (self.position_size == 1) and (metric not in ambiguous_metrics | single_only_metrics):
            raise ValueError('Provided metric: {} not available for pos_size: {}, please select from:\n{}'.format(
                metric, self.position_size, ', '.join(list(ambiguous_metrics | single_only_metrics))))
        elif (self.position_size == 2) and (metric not in ambiguous_metrics | pair_only_metrics):
            raise ValueError('Provided metric: {} not available for pos_size: {}, please select from:\n{}'.format(
                metric, self.position_size, ', '.join(list(ambiguous_metrics | pair_only_metrics))))
        self.metric = metric

    def score_group(self, freq_table):
        scoring_functions = {'identity': group_identity_score}
        scores = np.zeros(self.dimensions)
        for pos in freq_table.get_positions():
            score = scoring_functions[self.metric](freq_table, pos)
            if self.position_size == 1:
                scores[pos] = score
            else:
                scores[pos[0], pos[1]] = score
        return scores

    def score_rank(self, score_tensor):
        scoring_functions = {'identity': rank_identity_score}
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

    """
    cumulative_scores = np.sum(score_matrix, axis=0)
    rank_scores = 1 * (cumulative_scores != 0)
    return rank_scores
