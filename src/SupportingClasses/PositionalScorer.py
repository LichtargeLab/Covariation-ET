"""
Created on July 12, 2019

@author: Daniel Konecki
"""
import numpy as np
from scipy.sparse import csc_matrix

integer_valued_metrics = {'identity'}

real_valued_metrics = {'plain_entropy', 'mutual_information', 'normalized_mutual_information',
                       'average_product_corrected_mutual_information',
                       'filtered_average_product_corrected_mutual_information',
                       'match_count', 'mismatch_count', 'match_mismatch_count_ratio', 'match_mismatch_count_angle',
                       'match_entropy', 'mismatch_entropy', 'match_mismatch_entropy_ratio',
                       'match_mismatch_entropy_angle',
                       'match_diversity', 'mismatch_diversity', 'match_mismatch_diversity_ratio',
                       'match_mismatch_diversity_angle',
                       'match_diversity_mismatch_entropy_ratio', 'match_diversity_mismatch_entropy_angle'}

ambiguous_metrics = {'identity', 'plain_entropy'}

single_only_metrics = set()

pair_only_metrics = {'mutual_information', 'normalized_mutual_information',
                     'average_product_corrected_mutual_information',
                     'filtered_average_product_corrected_mutual_information', 'match_count', 'mismatch_count',
                     'match_mismatch_count_ratio', 'match_mismatch_count_angle', 'match_entropy', 'mismatch_entropy',
                     'match_mismatch_entropy_ratio', 'match_mismatch_entropy_angle', 'match_diversity',
                     'mismatch_diversity', 'match_mismatch_diversity_ratio', 'match_mismatch_diversity_angle',
                     'match_diversity_mismatch_entropy_ratio', 'match_diversity_mismatch_entropy_angle'}

min_metrics = {'identity', 'plain_entropy', 'mismatch_count', 'match_mismatch_count_ratio',
               'match_mismatch_count_angle', 'mismatch_entropy', 'match_mismatch_entropy_ratio',
               'match_mismatch_entropy_angle', 'mismatch_diversity', 'match_mismatch_diversity_ratio',
               'match_mismatch_diversity_angle', 'match_diversity_mismatch_entropy_ratio',
               'match_diversity_mismatch_entropy_angle'}

max_metrics = {'mutual_information', 'normalized_mutual_information', 'average_product_corrected_mutual_information',
               'filtered_average_product_corrected_mutual_information', 'match_count', 'match_entropy',
               'match_diversity'}


class PositionalScorer(object):
    """
    This class is meant to serve as an interchangeable component of the Trace.trace function. It produces group and
    rank level scores using different metrics.

    Attributes:
        sequence_length (int): The length of the sequence being analysed.
        position_size (int): The size of the the positions being analyzed (expecting 1 single position scores or 2 pair
        position scores).
        dimensions (tuple): The dimensions expected for the scoring arrays to be generated at the group and rank levels.
        metric (str): Which metric to use when computing group and rank level scores. Currently available metrics:
            identity: Whether a position is fully conserved within a group and across all groups (resulting in a score
            of 0), or not (resulting in a score of 1).
            plain_entropy: Entropy score for a position in the alignment (0.0 means invariant while the higher the score
            the more variable the position is).
            mutual_information: Mutual information score for pairs of positions in the alignment (0.0 means invariant or
            random while the higher the score the more covarying the position is).
            normalized_mutual_information: Normalized mutual information score for pairs of positions in the alignment
            (0.0 means invariant or random while the closer to 1.0 the score the more covarying the position is).
            average_product_corrected_mutual_information: Mutual information score corrected with the APC for pairs of
            positions in the alignment (the lower the value, it may become negative, the more invariant or random while
            the higher the score the more covarying the position is).
            filtered_average_product_corrected_mutual_information: Mutual information score corrected with the APC with
            very low scores squashed to 0.0 for pairs of positions in the alignment (a score of 0.0 means the position
            is invariant or random while the higher the score the more covarying the position is).



            match_mismatch_entropy_angle: A new metric which measures the angle between a vector describing match
            entropy (one axis) and mismatch entropy (a second axis). An angle of 0 corresponds to either fully invariant
            or fully covarying positions while an angle of 90 corresponds to a randomly varying position.
        metric_type (str): Whether the metric provided is an integer or real valued metric. This is used to determine
        rank scoring.
        rank_type (str): Whether the metric provided has its best score at its 'min' or 'max' (e.g. an identity of 0
        is good, it corresponds to a invariance while 1 is bad and corresponds with variance, while for mutual
        information a low score signifies lack of covariation while a high score corresponds to covariation.
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
        elif self.position_size > 2:
            raise ValueError(f'Position sizes > 2 not currently supported {self.position_size} provided.')
        else:
            pass
        self.metric = metric
        if metric in integer_valued_metrics:
            self.metric_type = 'integer'
        elif metric in real_valued_metrics:
            self.metric_type = 'real'
        else:
            raise ValueError('Provided metric is neither integer valued nor real valued!')
        if metric in min_metrics:
            self.rank_type = 'min'
        elif metric in max_metrics:
            self.rank_type = 'max'
        else:
            raise ValueError('Provided metric is neither min or max ranked!')

    def score_group(self, freq_table):
        """
        Score Group

        This function is intended to generate a score vector/matrix/tensor for a single group. It does so by scoring
        each position characterized in a FrequencyTable and using those scores to fill in a properly dimensioned
        vector/matrix/tensor. This function acts as a switch statement calling the proper scoring function on the input
        to determine the desired scores.

        Args:
            freq_table (FrequencyTable/dict): The table characterizing the character counts at each position in an
            alignment. If the match_mismatch_entropy_angle is being calculated then a dictionary should be provided
            which contains the keys 'match' and 'mismatch' mapped to the appropriate FrequencyTables.
        Returns:
            np.array: A properly dimensioned vector/matrix/array containing the scores for each position in an alignment
            as determined by the specified metric.
        """
        scoring_functions = {'identity': group_identity_score, 'plain_entropy': group_plain_entropy_score,
                             'mutual_information': group_mutual_information_score,
                             'normalized_mutual_information': group_normalized_mutual_information_score,
                             'average_product_corrected_mutual_information': group_mutual_information_score,
                             'filtered_average_product_corrected_mutual_information': group_mutual_information_score,
                             'match_count': group_match_count_score, 'mismatch_count': group_mismatch_count_score,
                             'match_mismatch_count_ratio': group_match_mismatch_count_ratio,
                             'match_mismatch_count_angle': group_match_mismatch_count_angle,
                             'match_entropy': group_match_entropy_score,
                             'mismatch_entropy': group_mismatch_entropy_score,
                             'match_mismatch_entropy_ratio': group_match_mismatch_entropy_ratio,
                             'match_mismatch_entropy_angle': group_match_mismatch_entropy_angle,
                             'match_diversity': group_match_diversity_score,
                             'mismatch_diversity': group_mismatch_diversity_score,
                             'match_mismatch_diversity_ratio': group_match_mismatch_diversity_ratio,
                             'match_mismatch_diversity_angle': group_match_mismatch_diversity_angle,
                             'match_diversity_mismatch_entropy_ratio': group_match_diversity_mismatch_entropy_ratio,
                             'match_diversity_mismatch_entropy_angle': group_match_diversity_mismatch_entropy_angle}
        scores = scoring_functions[self.metric](freq_table, self.dimensions)
        if self.metric == 'average_product_corrected_mutual_information':
            scores = average_product_correction(scores)
        elif self.metric == 'filtered_average_product_corrected_mutual_information':
            scores = filtered_average_product_correction(scores)
        return scores

    def score_rank(self, score_tensor, rank):
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
            rank (int): The rank for which the score is being computed, needed for normalizing the final scores for real
            valued metrics.
        Returns:
            np.array: A properly dimensioned vector/matrix/array containing the scores for each position in an alignment
            as determined by the specified metric.
        """
        if score_tensor.shape != self.dimensions:
            raise ValueError('The scoring matrix provided must mach the dimensions of PositionalScorer: '
                             + f'{score_tensor.shape} vs {self.dimensions}')
        scoring_functions = {'integer': rank_integer_value_score, 'real': rank_real_value_score}
        scores = scoring_functions[self.metric_type](score_tensor, rank)
        if self.position_size == 2:
            scores = np.triu(scores, k=1)
        return scores


def rank_integer_value_score(score_matrix, rank):
    """
    Rank Integer Value Score

    This computes the final rank score for all positions in a characterized alignment if the group score was produced by
    an integer, not a real valued, scoring metric. A position is identical/conserved only if it was scored 0 in all
    groups. In any other case the position is not considered conserved (receives a score of 1).

    Args:
        score_matrix (np.array): A 1 or 2 d matrix of scores summed over all groups at the specified rank. Scores should
        be binary integer values such as those produced by group_identity_score.
        rank (int): The rank for which the score is being computed.
    Returns:
        np.array: A score vector/matrix for all positions in the alignment with binary values to show whether a position
        is conserved in every group at the current rank (0) or if it is variable in at least one group (1).
    """
    if score_matrix is None:
        raise ValueError('score_matrix is expected to be a 1 or 2 dimensional np.array!')
    if rank is None:
        raise ValueError('For which rank is rank_integer_value_score being called!')
    rank_scores = 1 * (score_matrix != 0)
    return rank_scores


def rank_real_value_score(score_matrix, rank):
    """
    Rank Real Value Score

    This computes the final rank score for all positions in a characterized alignment if the group score was produced by
    a real valued, not an integer, scoring metric. The rank score is the weighted sum of the real valued scores across
    all groups for a given position.

    Args:
        score_matrix (np.array): A 1 or 2 dimensional matrix of scores summed over all groups at the specified rank.
        Scores should be real valued, such as those produced by group_plain_entropy_score.
        rank (int): The rank for which the score is being computed, needed to normalize the final score.
    Returns:
        np.array: A score vector/matrix for all positions in the alignment with float values to show whether a position
        is conserved in evert group at the current rank (0.0) or if it is variable in any of the groups (> 0.0).
    """
    if score_matrix is None:
        raise ValueError('score_matrix is expected to be a 1 or 2 dimensional np.array!')
    if rank is None:
        raise ValueError('For which rank is rank_integer_value_score being called!')
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
    if len(dimensions) != freq_table.position_size:
        raise ValueError('FrequencyTable position size and dimensions do not agree!')
    elif len(dimensions) == 1:
        final = identical
    elif len(dimensions) == 2:
        final = np.zeros(dimensions)
        final[np.triu_indices(n=dimensions[0])] = identical
    else:
        raise ValueError('group_identity_score is not implemented for dimensions describing axis other than 1 or 2.')
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
    counts = freq_table.get_table()
    depth = float(freq_table.get_depth())
    freq = counts.data / depth
    freq_log = np.log(freq)
    inter_prod = csc_matrix((freq * freq_log, counts.indices, counts.indptr), shape=counts.shape)
    del counts
    inter_sum = np.array(inter_prod.sum(axis=1)).reshape(-1)
    entropies = -1.0 * inter_sum
    if len(dimensions) != freq_table.position_size:
        raise ValueError('FrequencyTable position size and dimensions do not agree!')
    elif len(dimensions) == 1:
        final = entropies
    elif len(dimensions) == 2:
        final = np.zeros(dimensions)
        final[np.triu_indices(n=dimensions[0])] = entropies
    else:
        raise ValueError('group_plain_entropy_score is not implemented for dimensions describing axis other than '
                         '1 or 2.')
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
    if len(dimensions) != freq_table.position_size:
        raise ValueError('FrequencyTable position size and dimensions do not agree!')
    elif len(dimensions) != 2:
        raise ValueError('Mutual information computation must be performed on position of size 2!')
    else:
        pass
    joint_entropies_ij = group_plain_entropy_score(freq_table=freq_table, dimensions=dimensions)
    diagonal_indices = (list(range(dimensions[0])), list(range(dimensions[1])))
    entropies_j = np.zeros(dimensions)
    entropies_j[list(range(dimensions[0])), :] = joint_entropies_ij[diagonal_indices]
    entropies_i = entropies_j.T
    mask = np.triu(np.ones(dimensions), k=1)
    entropies_j = entropies_j * mask
    entropies_i = entropies_i * mask
    joint_entropies_ij[diagonal_indices] = 0.0
    mutual_information_matrix = (entropies_i + entropies_j) - joint_entropies_ij
    return entropies_i, entropies_j, joint_entropies_ij, mutual_information_matrix


def group_mutual_information_score(freq_table, dimensions):
    """
    Group Mutual Information Score

    This function computes the mutual information score for a given group. The formula used for mutual information in
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

    This function computes the normalized mutual information score for a given group. The formula used in this case
    employs the arithmetic mean to determine the normalization constant: NMI = (Hi + Hj -Hij) / mean(Hi, Hj) where Hi
    and Hj are the position specific entropies of the two positions and Hij is the joint entropy of the pair of
    positions. There are two special cases for this computation. First, if both Hi and Hj are 0.0 then the pair of
    positions is fully conserved and will received the highest possible score 1.0, mirroring the implementation in other
    popular packages (see sklearn.metrics.cluster.supervised.normalized_mutual_info). Second, if the normalization
    for a pair of positions is determined to be 0.0, then its value is set to 0.0.

    Args:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the normalized mutual
        information score.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then an array is returned, if two are given an array is returned.
    Returns:
        np.array: The normalized mutual information score for all positions in the FrequencyTable.
    """
    entropies_i, entropies_j, _, mutual_information = mutual_information_computation(freq_table=freq_table,
                                                                                     dimensions=dimensions)
    final = np.zeros(dimensions)
    # Define normalization constant (using the arithmetic mean here)
    normalization = np.mean([entropies_i, entropies_j], axis=0)
    # Identify positions which can be normalized without error
    indices = np.nonzero(normalization)
    # Normalize positions which have a non-zero normalization constant
    final[indices] += mutual_information[indices]
    final[indices] /= normalization[indices]
    # If the entropy is 0 for both positions then we have a special case where the normalized mutual information is 1.
    upper_triangle_mask = np.triu(np.ones(dimensions), k=1) == 1
    hi_0 = entropies_i == 0
    hj_0 = entropies_j == 0
    bool_indices = upper_triangle_mask & (hi_0 & hj_0)
    final[bool_indices] = 1.0
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
        np.array: An upper triangle matrix with mutual information with average product correction scores.
    """
    if mutual_information_matrix is None:
        raise ValueError('No mutual information matrix provided!')
    if mutual_information_matrix.shape[0] != mutual_information_matrix.shape[1]:
        raise ValueError('Mutual information matrix is expected to be square!')
    if np.tril(mutual_information_matrix).any():
        raise ValueError('Mutual information matrix is expected to be an upper triangle matrix (not including the '
                         'diagonal!')
    # Determine the size of the matrix (number of non-gap positions in the alignment reference sequence).
    dim = mutual_information_matrix.shape[0]
    # Compute the position specific mutual information averages (excludes the position itself)
    diagonal_values = mutual_information_matrix[list(range(dim)), list(range(dim))]
    # Compute the average over the entire mutual information matrix (excludes the diagonal)
    diagonal_sum = np.sum(diagonal_values)
    matrix_sum = np.sum(mutual_information_matrix) - diagonal_sum
    if matrix_sum == 0.0:
        apc_corrected = np.zeros((dim, dim))
        if np.abs(mutual_information_matrix).any():
            raise ValueError('APC correction will experience divide by zero error, but mutual information matrix '
                             'includes non-zero values.')
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


def filtered_average_product_correction(mutual_information_matrix, threshold=0.0001):
    """
    Filtered Average Product Correction

    This function uses the average_product_correction function to generate a matrix of apc corrected mi values. This
    function then checks for low mutual information values (<0.0001) and coerces all final scores for those positions to
    0.0 as opposed to the average product corrected values.

    Args:
        mutual_information_matrix (np.array): An upper triangle mutual information score matrix for which to compute the
        mutual information with average product correction.
        threshold (float): The mutual information value below which to set final scores to 0.0
    Returns:
        np.array: An upper triangle matrix with average product corrected mutual information scores. If the mutual
        information score at a given position is below the threshold the final score is overwritten with 0.0.
    """
    apc_corrected = average_product_correction(mutual_information_matrix=mutual_information_matrix)
    # Performing filtering that Angela performs
    positions = mutual_information_matrix <= threshold
    apc_corrected[positions] = 0.0
    return apc_corrected


def count_computation(freq_table, dimensions):
    """
    Count Computation

    This function accepts a frequency table and returns the corresponding final count values for all positions, where
    the count is the sum of the counts of each character observed at that position. This is intended for use in the
    group_match_count_score and group_mismatch_count_score functions.

    Arguments:
        freq_table (FrequencyTable): The characterization of matches or mismatches for pairs of positions in an
        alignment, to use when computing the counts.
        dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected and
        will return a 2-D array.
    Return:
        np.array: An array with the shape given by dimensions, containing the total character counts for all positions
        in the provided frequency table.
    """
    if len(dimensions) != freq_table.position_size:
        raise ValueError('FrequencyTable position size and dimensions do not agree!')
    elif len(dimensions) == 2:
        counts = freq_table.get_table()
        position_sums = counts.sum(axis=1).flatten()
        final = np.zeros(dimensions)
        final[np.triu_indices(n=dimensions[0])] = position_sums
        # Having scores for positions paired to themselves is not meaningful in the context of covariation prediction.
        final[list(range(dimensions[0])), list(range(dimensions[0]))] = 0.0
    else:
        raise ValueError('count_computation metrics are only intended for use with FrequencyTable objects of position '
                         'size 2!')
    return final


def diversity_computation(freq_table, dimensions):
    """
    Diversity Computation

    This function accepts a frequency table and returns the corresponding diversity values for all positions, where:
        diversity = e^(h) = e^(-1 * Sum(p*log(p)))

    Arguments:
        freq_table (FrequencyTable): The characterization of an alignment, to use when computing the diversity.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then a 1-D array is returned, if two are given a 2-D array is returned.
    Return:
        np.array: An array with the shape given by dimensions, containing the diversity values for all positions in the
        provided frequency table.
    """
    if len(dimensions) == 2:
        entropies = group_plain_entropy_score(freq_table=freq_table, dimensions=dimensions)
        diversities = np.exp(entropies)
        # Having scores for positions paired to themselves is not meaningful in the context of covariation prediction.
        diversities = np.triu(diversities, k=1)
    else:
        raise ValueError('diversity_computation metrics are only intended for use with FrequencyTable objects of '
                         'position size 2!')
    return diversities


def ratio_computation(match_table, mismatch_table):
    """
    Ratio Computation

    This function uses two match and mismatch statistics to calculate the ratio:
        ratio = mismatch metric / match metric
    Special cases causing divide by 0 errors are corrected for as follows. If the match statistic is 0 but the
    mismatch statistic is not, the lowest possible float which evaluates to 90 degrees by tan (as determined by
    np.tan(np.pi / 2.0)). If the mismatch statistic is also 0, then 0 is assigned at that position instead. One
    intention of this computation is that it can be applied to calculate the angle between fully conserved/covarying (0)
    and fully variable (pi/2).

    Arguments:
        match_table (np.array): An array of metrics for positions in an alignment which constitute a match (conservation
        or covariation).
        mismatch_table (np.array): An array of metrics for positions in an alignment which constitute a mismatch
        (variation).
    Return:
        np.array: An array with the same shape as the provided inputs which contains the ratio of mismatch / match
        metrics, including the corrections described above.
    """
    ratio = np.zeros(match_table.shape)
    # Avoid divide by zero warnings by detecting those positions before the divide.
    div_by_0 = match_table == 0.0
    # Perform the divide for only those positions where the denominator is not 0.
    ratio[~div_by_0] = mismatch_table[~div_by_0] / match_table[~div_by_0]
    # Set the positions where the denominator is 0 to the lowest possible float which evaluates to 90 degrees by tan.
    ratio[div_by_0] = np.tan(np.pi / 2.0)
    # Determine the positions where the numerator and denominator are 0 and re-assign those values to 0.
    both_0 = div_by_0 & (mismatch_table == 0.0)
    ratio[both_0] = 0.0
    return ratio


def angle_computation(ratios):
    """
    Angle Computation

    This function accepts a matrix of ratio of mismatch / match values for a given metric. It computes the angle between
    the vector described by the match/mismatch values (from the match y axis). This is achieved using the formula:

        theta = arctan(mismatch / match)

    Two special cases are handled by this computation. If the match values is 0.0 (resulting in a divide by 0 warning)
    for a given position, then the only signal must come from the mismatches meaning that the vector is a running
    parallel to the x-axis so pi/2.0 or 90 degrees is returned. If the mismatch value is 0.0 (or is also 0.0) the value
    returned is an angle of 0.0 because the position must not have mismatches, or be one of two edge cases where there
    are either 2 or 1 sequences.

    Arguments:
        ratios (np.array): An array of mismatch / match statistic ratios as would be produced by the ratio_computation
        method.
    Returns:
        np.array: The angle computed between the match and mismatch values.
    """
    angles = np.arctan(ratios)
    return angles


def group_match_count_score(freq_tables, dimensions):
    """
    Group Match Count Score

    This function accepts a frequency table, representing the match cases (invariant or covariant signal) for pairs of
    positions in an alignment, and returns the corresponding final count values for all positions using the
    count_computation function.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected and
        will return a 2-D array.
    Return:
        np.array: An array with the shape given by dimensions, containing the total character counts for all positions
        in the provided match frequency table.
    """
    return count_computation(freq_table=freq_tables['match'], dimensions=dimensions)


def group_mismatch_count_score(freq_tables, dimensions):
    """
    Group Mismatch Count Score

    This function accepts a frequency table, representing the mismatch cases (variation signal) for pairs of positions
    in an alignment, and returns the corresponding final count values for all positions using the count_computation
    function.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected and
        will return a 2-D array.
    Return:
        np.array: An array with the shape given by dimensions, containing the total character counts for all positions
        in the provided mismatch frequency table.
    """
    return count_computation(freq_table=freq_tables['mismatch'], dimensions=dimensions)


def group_match_mismatch_count_ratio(freq_tables, dimensions):
    """
    Group Match Mismatch Count Ratio

    This function computes the ratio between match (invariant or covariant signal) and mismatch (variation signal)
    counts. A ratio of 0 corresponds to invariance or covariation while a ratio of np.tan(np.pi / 2.0) corresponds to
    fully variable. The ratio is computed by first calculating the counts of matches and the counts of mismatches and
    passing them to the ratio_computation method.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected and
        will return a 2-D array.
    Returns:
        np.array: An array of ratios computed by ratio_computation (see documentation), providing the ratio between the
        match and mismatch count axes, with a ratio of 0 corresponding to invariance or covariation and
        np.tan(np.pi / 2.0) corresponding to full variation.
    """
    match_counts = group_match_count_score(freq_tables=freq_tables, dimensions=dimensions)
    mismatch_counts = group_mismatch_count_score(freq_tables=freq_tables, dimensions=dimensions)
    ratio = ratio_computation(match_table=match_counts, mismatch_table=mismatch_counts)
    return ratio


def group_match_mismatch_count_angle(freq_tables, dimensions):
    """
        Group Match Mismatch Count Angle

        This function computes the angle between match (invariant or covariant signal) and mismatch (variation signal)
        counts. A ratio of 0 corresponds to invariance or covariation while a ratio of np.tan(np.pi / 2.0) corresponds
        to fully variable. The angle is computed by first calculating the counts of matches and the counts of mismatches
        and passing them to the ratio_computation method. The resulting ratios are then passed to the angle_computation
        function for the final angles.

        Arguments:
            freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
            objects.
            dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected
            and will return a 2-D array.
        Returns:
            np.array: An array of angles computed by angle_computation (see documentation), providing the ratio between
            the match and mismatch count axes, with a ratio of 0 corresponding to invariance or covariation and
            np.pi / 2.0 corresponding to full variation.
        """
    ratios = group_match_mismatch_count_ratio(freq_tables=freq_tables, dimensions=dimensions)
    angles = angle_computation(ratios=ratios)
    return angles


def group_match_entropy_score(freq_tables, dimensions):
    """
    Group Match Entropy Score

    This function accepts a frequency table, representing the match cases (invariant or covariant signal) for pairs of
    positions in an alignment, and returns the corresponding entropy values for all positions using the
    count_computation function.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected and
        will return a 2-D array.
    Return:
        np.array: An array with the shape given by dimensions, containing the entropy of only matching characters for
        all positions in the provided match frequency table.
    """
    if len(dimensions) == 2:
        entropy = group_plain_entropy_score(freq_table=freq_tables['match'], dimensions=dimensions)
        # Having scores for positions paired to themselves is not meaningful in the context of covariation prediction.
        entropy[list(range(dimensions[0])), list(range(dimensions[0]))] = 0.0
    else:
        raise ValueError('group_match_entropy_score metrics are only intended for use with FrequencyTable objects of '
                         'position size 2!')
    return entropy


def group_mismatch_entropy_score(freq_tables, dimensions):
    """
    Group Mismatch Entropy Score

    This function accepts a frequency table, representing the mismatch cases (variation signal) for pairs of positions
    in an alignment, and returns the corresponding entropy values for all positions using the count_computation
    function.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected and
        will return a 2-D array.
    Return:
        np.array: An array with the shape given by dimensions, containing the entropy of only mismatching characters for
        all positions in the provided mismatch frequency table.
    """
    if len(dimensions) == 2:
        entropy = group_plain_entropy_score(freq_table=freq_tables['mismatch'], dimensions=dimensions)
        # Having scores for positions paired to themselves is not meaningful in the context of covariation prediction.
        entropy[list(range(dimensions[0])), list(range(dimensions[0]))] = 0.0
    else:
        raise ValueError('group_mismatch_entropy_score metrics are only intended for use with FrequencyTable objects '
                         'of position size 2!')
    return entropy


def group_match_mismatch_entropy_ratio(freq_tables, dimensions):
    """
    Group Match Mismatch Entropy Ratio

    This function computes the ratio between match (invariant or covariant signal) and mismatch (variation signal). A
    ratio of 0 corresponds to invariance or covariation while a ratio of np.finfo(float).max corresponds to fully
    variable. The ratio is computed by first calculating the entropy of matches and the entropy of mismatches and
    passing them to the ratio_computation method.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then a vector (1-d array) is returned, if two are given a matrix (2-d array) is returned.
    Returns:
        np.array: An array of ratios computed by ratio_computation (see documentation), providing the ratio between the
        match and mismatch entropy axes, with a ratio of 0 corresponding to invariance or covariation and
        np.finfo(float).max corresponding to full variation.
    """
    match_entropy = group_match_entropy_score(freq_tables=freq_tables, dimensions=dimensions)
    mismatch_entropy = group_mismatch_entropy_score(freq_tables=freq_tables, dimensions=dimensions)
    ratio = ratio_computation(match_table=match_entropy, mismatch_table=mismatch_entropy)
    return ratio


def group_match_mismatch_entropy_angle(freq_tables, dimensions):
    """
    Group Match Mismatch Entropy Angle

    This function computes the angle that each position in a group is from the optimal invariant or covariant signal. An
    angle of 0 corresponds to invariance or covariation while and angle of 90 (pi/2) corresponds to fully variable. The
    angle is computed by first calculating the entropy of matches and the entropy of mismatches, then computing their
    ratio (with corrections, see ratio_computation), and finally their angle (see angle_computation).

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then a vector (1-d array) is returned, if two are given a matrix (2-d array) is returned.
    Returns:
        np.array: An array of angles computed by angle_computation (see documentation), providing the angle between the
        match and mismatch entropy axes, with an angle of 0 corresponding to closeness to the match axis and 90 or pi/2
        corresponding to closeness to the mismatch axis.
    """
    ratios = group_match_mismatch_entropy_ratio(freq_tables=freq_tables, dimensions=dimensions)
    angles = angle_computation(ratios=ratios)
    return angles


def group_match_diversity_score(freq_tables, dimensions):
    """
    Group Match Diversity Score

    This function accepts a frequency table, representing the match cases (invariant or covariant signal) for pairs of
    positions in an alignment, and returns the corresponding diversity values for all positions using the
    diversity_computation function.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected and
        will return a 2-D array.
    Return:
        np.array: An array with the shape given by dimensions, containing the diversity only matching characters for all
        positions in the provided match frequency table.
    """
    return diversity_computation(freq_table=freq_tables['match'], dimensions=dimensions)


def group_mismatch_diversity_score(freq_tables, dimensions):
    """
    Group Mismatch Diversity Score

    This function accepts a frequency table, representing the mismatch cases (variation signal) for pairs of positions
    in an alignment, and returns the corresponding diversity values for all positions using the diversity_computation
    function.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected and
        will return a 2-D array.
    Return:
        np.array: An array with the shape given by dimensions, containing the diversity for only mismatch characters for
        all positions in the provided mismatch frequency table.
    """
    return diversity_computation(freq_table=freq_tables['mismatch'], dimensions=dimensions)


def group_match_mismatch_diversity_ratio(freq_tables, dimensions):
    """
    Group Match Mismatch Diversity Ratio

    This function computes the ratio between match (invariant or covariant signal) and mismatch (variation signal)
    counts. A ratio of 0 corresponds to invariance or covariation while a ratio of np.tan(np.pi / 2.0) corresponds to
    fully variable. The ratio is computed by first calculating the diversity of matches and the diversity of mismatches
    and passing them to the ratio_computation method.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected and
        will return a 2-D array.
    Returns:
        np.array: An array of ratios computed by ratio_computation (see documentation), providing the ratio between the
        match and mismatch diversity axes, with a ratio of 0 corresponding to invariance or covariation and
        np.tan(np.pi / 2.0) corresponding to full variation.
    """
    match_diversity = group_match_diversity_score(freq_tables=freq_tables, dimensions=dimensions)
    mismatch_diversity = group_mismatch_diversity_score(freq_tables=freq_tables, dimensions=dimensions)
    ratio = ratio_computation(match_table=match_diversity, mismatch_table=mismatch_diversity)
    return ratio


def group_match_mismatch_diversity_angle(freq_tables, dimensions):
    """
        Group Match Mismatch Count Angle

        This function computes the angle between match (invariant or covariant signal) and mismatch (variation signal)
        counts. A ratio of 0 corresponds to invariance or covariation while a ratio of np.tan(np.pi / 2.0) corresponds
        to fully variable. The angle is computed by first calculating the counts of matches and the counts of mismatches
        and passing them to the ratio_computation method. The resulting ratios are then passed to the angle_computation
        function for the final angles.

        Arguments:
            freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
            objects.
            dimensions (tuple): A tuple describing the dimensions of the expected return, two dimensions are expected
            and will return a 2-D array.
        Returns:
            np.array: An array of angles computed by angle_computation (see documentation), providing the ratio between
            the match and mismatch diversity axes, with a ratio of 0 corresponding to invariance or covariation and
            np.pi / 2.0 corresponding to full variation.
        """
    ratios = group_match_mismatch_diversity_ratio(freq_tables=freq_tables, dimensions=dimensions)
    angles = angle_computation(ratios=ratios)
    return angles


def group_match_diversity_mismatch_entropy_ratio(freq_tables, dimensions):
    """
    Group Match Diversity Mismatch Entropy Ratio

    This function computes the ratio between match (invariant or covariant signal) and mismatch (variation signal). A
    ratio of 0 corresponds to invariance or covariation while a ratio of np.finfo(float).max corresponds to fully
    variable. The ratio is computed by first calculating the diversity of matches and the entropy of mismatches and
    passing them to the ratio_computation method.

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then a vector (1-d array) is returned, if two are given a matrix (2-d array) is returned.
    Returns:
        np.array: An array of ratios computed by ratio_computation (see documentation), providing the ratio between the
        match diversity and mismatch entropy axes, with a ratio of 0 corresponding to invariance or covariation and
        np.finfo(float).max corresponding to full variation.
    """
    match_diversity = diversity_computation(freq_table=freq_tables['match'], dimensions=dimensions)
    mismatch_entropy = group_plain_entropy_score(freq_table=freq_tables['mismatch'], dimensions=dimensions)
    ratio = ratio_computation(match_table=match_diversity, mismatch_table=mismatch_entropy)
    return ratio


def group_match_diversity_mismatch_entropy_angle(freq_tables, dimensions):
    """
    Group Match Diversity Mismatch Entropy Angle

    This function computes the angle that each position in a group is from the optimal invariant or covariant signal. An
    angle of 0 corresponds to invariance or covariation while and angle of 90 (pi/2) corresponds to fully variable. The
    angle is computed by first calculating the diversity of matches and the entropy of mismatches, then computing their
    ratio (with corrections, see ratio_computation), and finally their angle (see angle_computation).

    Arguments:
        freq_tables (dict): A dictionary mapping the keys 'match' and 'mismatch' to corresponding FrequencyTable
        objects.
        dimensions (tuple): A tuple describing the dimensions of the expected return, if only one dimension is given
        then a vector (1-d array) is returned, if two are given a matrix (2-d array) is returned.
    Returns:
        np.array: An array of angles computed by angle_computation (see documentation), providing the angle between the
        match diversity and mismatch entropy axes, with an angle of 0 corresponding to closeness to the match axis and
        90 or pi/2 corresponding to closeness to the mismatch axis.
    """
    ratios = group_match_diversity_mismatch_entropy_ratio(freq_tables=freq_tables, dimensions=dimensions)
    angles = angle_computation(ratios=ratios)
    return angles
