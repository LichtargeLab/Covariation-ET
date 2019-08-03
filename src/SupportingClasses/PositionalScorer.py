"""
Created on July 12, 2019

@author: Daniel Konecki
"""
import numpy as np
from scipy.stats import entropy

integer_valued_metrics = {'identity'}

real_valued_metrics = {'plain_entropy', 'mutual_information', 'normalized_mutual_information',
                       'average_product_corrected_mutual_information'}

ambiguous_metrics = {'identity', 'plain_entropy'}

single_only_metrics = set()

pair_only_metrics = {'mutual_information', 'normalized_mutual_information',
                     'average_product_corrected_mutual_information'}


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
        metric_type (str): Whether the metric provided is an integer or real valued metric. This is used to determine
        rank scoring.
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
        if metric in integer_valued_metrics:
            self.metric_type = 'integer'
        elif metric in real_valued_metrics:
            self.metric_type = 'real'
        else:
            raise ValueError('Provided metric is neither integer valued nor real valued!')

    # def score_group(self, freq_table):
    #     """
    #     Score Group
    #
    #     This function is intended to generate a score vector/matrix/tensor for a single group. It does so by scoring
    #     each position characterized in a FrequencyTable and using those scores to fill in a properly dimensioned
    #     vector/matrix/tensor.
    #
    #     Args:
    #         freq_table (FrequencyTable): The table characterizing the character counts at each position in an alignment.
    #     Returns:
    #         np.array: A properly dimensioned vector/matrix/array containing the scores for each position in an alignment
    #         as determined by the specified metric.
    #     """
    #     from time import time
    #     scoring_functions = {'identity': group_identity_score, 'plain_entropy': group_plain_entropy_score,
    #                          'mutual_information': group_mutual_information_score,
    #                          'normalized_mutual_information': group_normalized_mutual_information_score,
    #                          'average_product_corrected_mutual_information': group_mutual_information_score}
    #
    #     scores = np.zeros(self.dimensions)
    #     for pos in freq_table.get_positions():
    #         # inner_start = time()
    #         score = scoring_functions[self.metric](freq_table, pos)
    #         # inner_end = time()
    #         # print('Scoring function took: {} min'.format((inner_end - inner_start) / 60.0))
    #         if self.position_size == 1:
    #             scores[pos] = score
    #         else:
    #             # Enforce that only the upper triangle of the matrix gets filled in.
    #             if pos[0] == pos[1]:
    #                 continue
    #             scores[pos[0], pos[1]] = score
    #     if self.metric == 'average_product_corrected_mutual_information':
    #         scores = average_product_correction(scores)
    #     return scores

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
        from time import time
        scoring_functions = {'identity': group_identity_score2, 'plain_entropy': group_plain_entropy_score2,
                             'mutual_information': group_mutual_information_score,
                             'normalized_mutual_information': group_normalized_mutual_information_score,
                             'average_product_corrected_mutual_information': group_mutual_information_score}
        scores = scoring_functions[self.metric](freq_table, self.dimensions)
        # scores = np.zeros(self.dimensions)
        # for pos in freq_table.get_positions():
        #     # inner_start = time()
        #     score = scoring_functions[self.metric](freq_table, pos)
        #     # inner_end = time()
        #     # print('Scoring function took: {} min'.format((inner_end - inner_start) / 60.0))
        #     if self.position_size == 1:
        #         scores[pos] = score
        #     else:
        #         # Enforce that only the upper triangle of the matrix gets filled in.
        #         if pos[0] == pos[1]:
        #             continue
        #         scores[pos[0], pos[1]] = score
        if self.metric == 'average_product_corrected_mutual_information':
            scores = average_product_correction(scores)
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
        scoring_functions = {'integer': rank_integer_value_score, 'real': rank_real_value_score}
        scores = scoring_functions[self.metric_type](score_tensor)
        return scores


def rank_integer_value_score(score_matrix):
    """
    Rank Integer Value Score

    This computes the final rank score for all positions in a characterized alignment if the group score was produced by
    an integer, not a real valued, scoring metric. A position is identical/conserved only if it was scored 0 in all
    groups. In any other case the position is not considered conserved (receives a score of 1).

    Args:
        score_matrix (np.array): A matrix/tensor of scores which each score vector/matrix for a given group stacked
        along axis 0. Scores should be binary integer values such as those produced by group_identity_score.
    Returns:
        np.array: A score vector/matrix for all positions in the alignment with binary values to show whether a position
        is conserved in every group at the current rank (0) or if it is variable in at least one group (1).
    """
    # cumulative_scores = np.sum(score_matrix, axis=0)
    # rank_scores = 1 * (cumulative_scores != 0)
    rank_scores = 1 * (score_matrix != 0)
    return rank_scores


def rank_real_value_score(score_matrix):
    """
    Rank Real Value Score

    This computes the final rank score for all positions in a characterized alignment if the group score was produced by
    a real valued, not an integer, scoring metric. The rank score is the sum of the real valued scores across all groups
    for a given position.

    Args:
        score_matrix (np.array): A matrix/tensor of scores which each score vector/matrix for a given group stacked
        along axis 0. Scores should be real valued, such as those produced by group_plain_entropy_score.
    Returns:
        np.array: A score vector/matrix for all positions in the alignment with float values to show whether a position
        is conserved in evert group at the current rank (0.0) or if it is variable in any of the groups (> 0.0).
    """
    rank = score_matrix.shape[0]
    weight = 1.0 / rank
    # cumulative_scores = np.sum(score_matrix, axis=0)
    # rank_scores = weight * cumulative_scores
    rank_scores = weight * score_matrix
    return rank_scores


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

def group_identity_score2(freq_table, dimensions):
    table = freq_table.get_count_matrix()
    positional_sums = np.sum(table > 0, axis=1)
    identical = (positional_sums > 1) * 1
    identical = identical.reshape(-1)
    if len(dimensions) == 1:
        final = identical
    elif len(dimensions) == 2:
        final = np.zeros(dimensions)
        final[np.triu_indices(n=dimensions[0])] = identical
    else:
        raise ValueError('group_identity_score2 is not implemented for dimensions describing axis other than 1 or 2.')
    return final


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
    positional_frequencies = freq_table.get_frequency_array(pos=pos)
    positional_entropy = entropy(positional_frequencies)
    return positional_entropy


def group_plain_entropy_score2(freq_table, dimensions):
    table = freq_table.get_frequency_matrix()
    entropies = entropy(table.T)
    if len(dimensions) == 1:
        final = entropies
    elif len(dimensions) == 2:
        final = np.zeros(dimensions)
        final[np.triu_indices(n=dimensions[0])] = entropies
    else:
        raise ValueError('group_plain_entropy_score2 is not implemented for dimensions describing axis other than 1 or 2.')
    return final



def mutual_information_computation(freq_table, pos):
    """
    Mutual Information Computation

    This function compute the mutual information score for a position. The formula used for mutual information in this
    case is: MI = Hi + Hj -Hij where Hi and Hj are the position specific entropies of the two positions and Hij is the
    joint entropy of the pair of positions. This is kept separate from the group scoring methods because several group
    scores are dependent on this as their base function, this way all relevant terms can be computed and returned to
    those functions from one common function.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the mutual information
        score.
        pos (int/tuple): The position in the sequence/list of pairs for which to compute the plain entropy score.
    Returns:
        int: The first position in the pair for which mutual information was computed
        int: The second position in the pair for which mutual information was computed
        float: The position specific entropy for the first position.
        float: The position specific entropy for the second position.
        float: The joint entropy for the pair of positions.
        float: The mutual information score for the specified position in the FrequencyTable.
    """
    i, j = pos
    entropy_i = group_plain_entropy_score(freq_table=freq_table, pos=(i, i))
    entropy_j = group_plain_entropy_score(freq_table=freq_table, pos=(j, j))
    joint_entropy_ij = group_plain_entropy_score(freq_table=freq_table, pos=pos)
    mutual_information = (entropy_i + entropy_j) - joint_entropy_ij
    return i, j, entropy_i, entropy_j, joint_entropy_ij, mutual_information

def mutual_information_computation2(freq_table, dimensions):
    joint_entropies_ij = group_plain_entropy_score2(freq_table=freq_table, dimensions=dimensions)
    diagonal_indices = (list(range(dimensions[0])), list(range(dimensions[1])))
    entropies_i = np.zeros(dimensions)
    entropies_i[list(range(dimensions[0])), :] = joint_entropies_ij[diagonal_indices]
    entropies_j = np.zeros(dimensions)
    entropies_j[:, list(range(dimensions[1]))] = joint_entropies_ij[diagonal_indices]
    mutual_information_matrix = (entropies_i + entropies_j) - joint_entropies_ij
    return entropies_i, entropies_j, joint_entropies_ij, mutual_information_matrix


def group_mutual_information_score(freq_table, pos):
    """
    Group Mutual Information Score

    This function compute the mutual information score for a given group. The formula used for mutual information in
    this case is: MI = Hi + Hj -Hij where Hi and Hj are the position specific entropies of the two positions and Hij is
    the joint entropy of the pair of positions.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the mutual information
        score.
        pos (int/tuple): The position in the sequence/list of pairs for which to compute the plain entropy score.
    Returns:
        float: The mutual information score for the specified position in the FrequencyTable.
    """
    _, _, _, _, _, mutual_information = mutual_information_computation(freq_table, pos)
    return mutual_information


def group_normalized_mutual_information_score(freq_table, pos):
    """
    Group Normalized Mutual Information Score

    This function compute the mutual information score for a given group. The formula used for mutual information in
    this case is: MI = Hi + Hj -Hij where Hi and Hj are the position specific entropies of the two positions and Hij is
    the joint entropy of the pair of positions.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the mutual information
        score.
        pos (int/tuple): The position in the sequence/list of pairs for which to compute the plain entropy score.
    Returns:
        float: The mutual information score for the specified position in the FrequencyTable.
    """
    _, _, entropy_i, entropy_j, _, mutual_information = mutual_information_computation( freq_table=freq_table, pos=pos)
    normalization = entropy_i + entropy_j
    if normalization == 0.0:
        normalized_mutual_information = 0.0
        if mutual_information != 0.0:
            raise ValueError('normalization == 0.0 but mutual information == {}'.format(mutual_information))
    else:
        normalized_mutual_information = mutual_information / normalization
    return normalized_mutual_information


def average_product_correction(mutual_information_matrix):
    """
    Average Product Correction

    This function uses a mutual information matrix to calculate average product corrected mutual information. Average
    product correction includes division by the average mutual information of the unique off diagonal terms in the
    matrix (e.g. the upper triangle). If this average is 0, a matrix of all zeros will be returned, but first a check
    will be performed to ensure that the mutual information matrix was also all zeros (this should be the case because
    mutual information scores should fall in the range between 0 and 1 and thus there should be no negatives which could
    cause a zero average while other positions are non-zero). If this check fails a ValueError will be raised. If the
    average is not zero then the rest of the average product correction is computed and applied to the mutual
    information matrix in order to generate final scores.

    Args:
        mutual_information_matrix (np.array): An upper triangle mutual information score matrix for which to compute the
        mutual information with average product correction.
    Returns:
        np.array: An upper triangle matrix with mutual information with average product correction scores. If the
        average over the
    """
    if mutual_information_matrix.shape[0] != mutual_information_matrix.shape[1]:
        raise ValueError('Mutual information matrix is expected to be square!')
    # Determine the size of the matrix (number of non-gap positions in the alignment reference sequence).
    dim = mutual_information_matrix.shape[0]
    # Compute the position specific mutual information averages (excludes the position itself)
    diagonal_values = mutual_information_matrix[range(dim), range(dim)]
    # Compute the average over the entire mutual information matrix (excludes the diagonal)
    diagonal_sum = np.sum(diagonal_values)
    matrix_sum = np.sum(mutual_information_matrix) - diagonal_sum
    if matrix_sum == 0.0:
        apc_corrected = np.zeros((dim, dim))
        if np.abs(mutual_information_matrix).any():
            raise ValueError('APC correction will experience divide by zero error, but mutual information matrix includes non-zero values.')
    else:
        matrix_average = matrix_sum / np.sum(range(dim))
        # Since only the upper triangle of the matrix has been filled in the sums along both the column and the row are
        # needed to get the cumulative sum for a given position.
        position_specific_sums = (np.sum(mutual_information_matrix, axis=0) + np.sum(mutual_information_matrix, axis=1)
                                  - diagonal_values)
        position_specific_averages = position_specific_sums / float(dim - 1)
        # Calculate the matrix of products for the position specific average mutual information
        apc_numerator = np.outer(position_specific_averages, position_specific_averages)
        apc_factor = apc_numerator / matrix_average
        # Ensure that the correction factor is applied only to the portion of the matrix which has values (upper
        # triangle).
        upper_triangle_mask = np.zeros((dim, dim))
        upper_triangle_mask[np.triu_indices(dim, k=1)] = 1
        apc_factor = apc_factor * upper_triangle_mask
        # Compute the final corrected values.
        apc_corrected = mutual_information_matrix - apc_factor
    return apc_corrected
