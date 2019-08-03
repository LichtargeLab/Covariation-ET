"""
Created on July 12, 2019

@author: Daniel Konecki
"""
import numpy as np
from time import time
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
        scoring_functions = {'identity': group_identity_score, 'plain_entropy': group_plain_entropy_score,
                             'mutual_information': group_mutual_information_score,
                             'normalized_mutual_information': group_normalized_mutual_information_score,
                             'average_product_corrected_mutual_information': group_mutual_information_score}
        scores = scoring_functions[self.metric](freq_table, self.dimensions)
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
    rank_scores = weight * score_matrix
    return rank_scores


def group_identity_score(freq_table, dimensions):
    """
    Group Identity Score

    This computes the identity score for all positions in a characterized alignment. A position is considered conserved,
    resulting in a score of 0, if only one character from the valid alphabet is observed at that position. The position
    is considered variable, resulting in a score of 1, if multiple characters from the valid alphabet are observed at
    that position.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the identity score.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then an array is returned, if two are given an array is returned.
    Returns:
        np.array: 0 for positions that are conserved (i.e. only a single nucleic/amino acid has been observed at that
        position), and 1 otherwise (i.e. if multiple nucleic/amino acids have been observed at that position).
    """
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


def group_plain_entropy_score(freq_table, dimensions):
    """
    Group Plain Entropy Score

    This function computes the plain entropies for a given group. In this case entropy is computed using the formula:
    S = -Sum(pk * log(pk), axis=0) as given by scipy.stats.entropy. Here k is a character in the valid alphabet while
    pk is the frequency of that character in the alignment at the current position.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the plain entropy
        score.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then an array is returned, if two are given an array is returned.
    Returns:
        np.array: The plain entropies for all positions in the FrequencyTable.
    """
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


def mutual_information_computation(freq_table, dimensions):
    """
    Mutual Information Computation

    This function computes the mutual information score for all positions. The formula used for mutual information in
    this case is: MI = Hi + Hj -Hij where Hi and Hj are the position specific entropies of the two positions and Hij is
    the joint entropy of the pair of positions. This is kept separate from the group scoring methods because several
    group scores are dependent on this as their base function, this way all relevant terms can be computed and returned
    to those functions from one common function.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the mutual information
        score.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then an array is returned, if two are given an array is returned.
    Returns:
        np.array: The position specific entropies by rows.
        np.array: The position specific entropies by columns.
        np.array: The joint entropies for pairs of positions.
        np.array: The mutual information scores for all pairs of positions in the FrequencyTable.
    """
    joint_entropies_ij = group_plain_entropy_score(freq_table=freq_table, dimensions=dimensions)
    diagonal_indices = (list(range(dimensions[0])), list(range(dimensions[1])))
    entropies_j = np.zeros(dimensions)
    entropies_j[list(range(dimensions[0])), :] = joint_entropies_ij[diagonal_indices]
    entropies_i = entropies_j.T
    mask = np.triu(np.ones(dimensions), k=1)
    entropies_j = entropies_j * mask
    entropies_i = entropies_i * mask
    # Clear the joint entropy diagonal
    joint_entropies_ij[list(range(dimensions[0])), list(range(dimensions[1]))] = 0.0
    mutual_information_matrix = (entropies_i + entropies_j) - joint_entropies_ij
    return entropies_i, entropies_j, joint_entropies_ij, mutual_information_matrix


def group_mutual_information_score(freq_table, dimensions):
    """
    Group Mutual Information Score

    This function compute the mutual information score for a given group. The formula used for mutual information in
    this case is: MI = Hi + Hj -Hij where Hi and Hj are the position specific entropies of the two positions and Hij is
    the joint entropy of the pair of positions.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the mutual information
        score.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then an array is returned, if two are given an array is returned.
    Returns:
        np.array: The mutual information scores for all pairs of positions in the FrequencyTable.
    """
    _, _, _, mutual_information = mutual_information_computation(freq_table, dimensions)
    return mutual_information


def group_normalized_mutual_information_score(freq_table, dimensions):
    """
    Group Normalized Mutual Information Score

    This function compute the mutual information score for a given group. The formula used for mutual information in
    this case is: MI = Hi + Hj -Hij where Hi and Hj are the position specific entropies of the two positions and Hij is
    the joint entropy of the pair of positions.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the mutual information
        score.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then an array is returned, if two are given an array is returned.
    Returns:
        np.array: The normalized mutual information score for all positions in the FrequencyTable.
    """
    entropies_i, entropies_j, _, mutual_information = mutual_information_computation(freq_table=freq_table,
                                                                                     dimensions=dimensions)
    final = np.zeros(dimensions)
    normalization = entropies_i + entropies_j
    indices = np.nonzero(normalization)
    final[indices] += mutual_information[indices]
    final[indices] /= normalization[indices]
    return final


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
    diagonal_values = mutual_information_matrix[list(range(dim)), list(range(dim))]
    # Compute the average over the entire mutual information matrix (excludes the diagonal)
    diagonal_sum = np.sum(diagonal_values)
    matrix_sum = np.sum(mutual_information_matrix) - diagonal_sum
    upper_with_diag = np.triu(mutual_information_matrix)
    upper_no_diag = np.triu(mutual_information_matrix, k=1)
    lower_with_diag = np.tril(mutual_information_matrix)
    lower_no_diag = np.tril(mutual_information_matrix, k=-1)
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
