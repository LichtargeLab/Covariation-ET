"""
Created on May 15, 2019

@author: Daniel Konecki
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from itertools import combinations
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceMatrix
from utils import build_mapping, convert_seq_to_numeric
from EvolutionaryTraceAlphabet import FullIUPACDNA, FullIUPACProtein


class AlignmentDistanceCalculator(DistanceCalculator):
    """
    This class is meant to perform the calculation of sequence distance between sequences in an alignment. This
    information can then be used for other uses such as constructing a phylogenetic tree.

    Attributes:
        aln_type (str): Whether this is a protein or DNA alignment.
        alphabet (list): The corresponding amino or nucleic acid alphabet.
        model (str): The distance/substitution model to use when calculating sequence distance.
        alphabet_size (int): The number of characters in the alphabet used by this calculator.
        gap_characters (set): The gap characters not included in the alphabet which are mapped to alphabet_size in
        mapping (below).
        mapping (dict): A mapping from the character alphabet to their indices (used when computing identity distance or
        referencing substitution scores).
        scoring_matrix (np.ndarray): The corresponding scoring/substitution matrix for the chosen model.
    """

    def __init__(self, protein=True, model='identity', skip_letters=None):
        """
        Args:
            protein (bool): Whether or not this calculator will be used on protein alignments or not.
            model (str): The distance/substitution model to use when calculating sequence distance.
            skip_letters (list): Which characters to skip when scoring sequences in the alignment.
        """
        assert isinstance(protein, bool), 'Protein type is bool'
        if protein:
            self.aln_type = 'protein'
            self.alphabet = FullIUPACProtein()
            possible_models = DistanceCalculator.protein_models
        else:
            self.aln_type = 'dna'
            self.alphabet = FullIUPACDNA()
            possible_models = DistanceCalculator.dna_models
        possible_models.append('identity')
        if model not in possible_models:
            raise ValueError("Model '{}' not in list of possible models:\n{}".format(model, ', '.join(possible_models)))
        else:
            self.model = model
        super(AlignmentDistanceCalculator, self).__init__(model=model, skip_letters=skip_letters)
        self.alphabet_size, self.gap_characters, self.mapping, _ = build_mapping(alphabet=self.alphabet,
                                                                                 skip_letters=skip_letters)
        self.scoring_matrix = self._update_scoring_matrix()

    def _build_identity_scoring_matrix(self):
        """
        Build Identity Scoring Matrix

        This function builds the scoring matrix if the specified model is identity.

        Return:
             numpy.array: The identity scoring matrix.
        """

        substitution_matrix = np.eye(self.alphabet_size + 2)
        substitution_matrix[self.alphabet_size + 1, self.alphabet_size + 1] = 0
        return substitution_matrix

    def _rebuild_scoring_matrix(self):
        """
        Rebuild Scoring Matrix

        This function converts a Bio.Phylo.TreeConstruction.DistanceMatrix to a numpy array and then adds additional
        columns/rows for the gap and skip letter characters.

        Returns:
            numpy.array: The substitution matrix for the specified model.
        """
        # Convert the current scoring matrix, initialized by Bio.Phylo.TreeConstruction.DistanceCalculator, and convert
        # it to a np.array.
        scoring_matrix = np.array(self.scoring_matrix)
        # Pad the array to account for gap and skip characters.
        substitution_matrix = np.pad(scoring_matrix, pad_width=((0, 2), (0, 2)), mode='constant', constant_values=0)
        return substitution_matrix

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
        init_pairwise(alpha_map=self.mapping, alpha_size=self.alphabet_size, mod=self.model,
                      scoring_mat=self.scoring_matrix)
        _, _, final_score = pairwise(seq1=seq1, seq2=seq2)
        return final_score

    def get_identity_distance(self, msa, processes=1):
        """
        Get Identity Distance

        Compute the identity distance between the sequences in the provided alignment.

        Args:
            msa (Bio.Align.MultipleSeqAlignment): The alignment for which to calculate identity distances.
            processes (int): The size of processing pool which may be used to compute these distances.
        Returns:
            Bio.Phylo.TreeConstruction.DistanceMatrix: The identity distance matrix for the alignment.
        """
        names = [s.id for s in msa]
        dm = DistanceMatrix(names)
        num_aln = np.vstack([convert_seq_to_numeric(seq, mapping=self.mapping) for seq in msa])
        msa_size = len(msa)
        dist_pbar = tqdm(total=msa_size, unit='distances')

        def update_dm(res):
            """
            This function receives the output of a single call to identity and stores that in the current distance
            matrix, and then updates the progress bar for the distance calculation process.

            Args:
                res (tuple): A tuple containing the position of the sequence for which distances were calculated as well
                as an array of distances to other sequences.
            """
            seq_i, identity_distances = res
            for j in range(seq_i + 1, msa_size):
                dm[names[seq_i], names[j]] = identity_distances[j]
            dist_pbar.update(1)
            dist_pbar.refresh()

        if processes == 1:
            for i in range(msa_size):
                init_identity(num_aln)
                res = identity(i)
                update_dm(res)
        else:
            pool = Pool(processes=processes, initializer=init_identity,
                        initargs=(num_aln, ))
            for i in range(msa_size):
                pool.apply_async(identity, (i, ), callback=update_dm)
            pool.close()
            pool.join()
        dist_pbar.close()
        return dm

    def get_scoring_matrix_distance(self, msa, processes=1):
        """
        Get Scoring Matrix Distance

        Compute the distance between the sequences in the provided alignment based on the specified distance model.

        Args:
            msa (Bio.Align.MultipleSeqAlignment): The alignment for which to calculate distances.
            processes (int): The size of processing pool which may be used to compute these distances.
        Returns:
            Bio.Phylo.TreeConstruction.DistanceMatrix: The distance matrix for the alignment.
        """
        names = [s.id for s in msa]
        dm = DistanceMatrix(names)
        seq_pairs = list(combinations(msa, 2))
        dist_pbar = tqdm(total=len(seq_pairs), unit='distances')

        def update_dm(result):
            """
            This function receives the output of a single call to pairwise and stores that in the current distance
            matrix, and then updates the progress bar for the distance calculation process.

            Args:
                result (tuple): The return from pairwise, consisting of the two identifiers for the sequences whose
                distance has been calculated, and the final distance.
            """
            seq1_id, seq2_id, final_score = result
            dm[seq1_id, seq2_id] = final_score
            dist_pbar.update(1)
            dist_pbar.refresh()

        if processes == 1:
            init_pairwise(self.mapping, self.alphabet_size, self.model, self.scoring_matrix)
            for seq1, seq2 in seq_pairs:
                res = pairwise(seq1, seq2)
                update_dm(res)
        else:
            pool = Pool(processes=processes, initializer=init_pairwise,
                        initargs=(self.mapping, self.alphabet_size, self.model, self.scoring_matrix))
            for seq1, seq2 in seq_pairs:
                pool.apply_async(pairwise, (seq1, seq2), callback=update_dm)
            pool.close()
            pool.join()
        dist_pbar.close()
        return dm

    def get_distance(self, msa, processes=1):
        """Return a DistanceMatrix for MSA object.

        :Parameters:
            msa : MultipleSeqAlignment
                DNA or Protein multiple sequence alignment.
            processes : int
                The size of processing pool which may be used to compute these distances.
        """
        if not isinstance(msa, MultipleSeqAlignment):
            raise TypeError("Must provide a MultipleSeqAlignment object.")
        if self.model == 'identity':
            dm = self.get_identity_distance(msa, processes=processes)
        else:
            dm = self.get_scoring_matrix_distance(msa, processes=processes)
        return dm

    def get_et_distance(self, msa, processes=1):
        """
        Get ET Distance

        Calculates the sequence similarity using identity and substitution scoring metrics (this mirrors the previous
        implementations used by ETC in the lab).

        Args:
            msa (Bio.Align.MultipleSeqAlignment): The alignment for which to calculate distances.
            processes (int): The size of processing pool which may be used to compute these distances.
        Returns:
            Bio.Phylo.TreeConstruction.DistanceMatrix: The identity based sequence similarity distance matrix for the
            alignment.
            Bio.Phylo.TreeConstruction.DistanceMatrix: The substitution matrix based distance matrix for the alignment.
            pandas.DataFrame: A DataFrame with intermediate values for the distance calculation.
            float: The threshold used to determine the cutoff for similarity using the substitution matrix.
        """
        if not isinstance(msa, MultipleSeqAlignment):
            raise TypeError("Must provide a MultipleSeqAlignment object.")
        # Set cutoff for scoring matrix "identity"
        scoring_matrix_tril = np.tril(self.scoring_matrix, k=-1)
        positive_scores = scoring_matrix_tril[scoring_matrix_tril > 0]
        count = positive_scores.shape[0]
        if count > 0:
            positive_sum = np.sum(positive_scores)
            average = float(positive_sum) / count
            if average < 1.0:
                threshold = 1.0
            else:
                threshold = np.floor(average + 0.5)
        else:
            threshold = 1
        # Compute the non-gap length of the sequences
        seq_conversion = {}
        char_bar = tqdm(total=len(msa), unit='sequences')

        def update_seq_conversion(res):
            """
            This function receives the output of a single call to characterize_sequence and stores it in the
            seq_conversion dictionary, and then updates the progress bar for the characterization process.

            Args:
                res (tuple): The return from characterize_sequence, consisting of a sequence identifier, the non-gap
                length of the sequence, the non-gap positions in the sequence, and the numerical representation of the
                sequence.
            """
            seq_id, non_gap_length, non_gap_pos, num_repr = res
            seq_conversion[seq_id] = {'non_gap_length': non_gap_length, 'non_gap_pos': non_gap_pos,
                                      'num_repr': num_repr}
            char_bar.update(1)
            char_bar.refresh()

        if processes == 1:
            init_characterize_sequence(msa, self.mapping, self.alphabet_size)
            for i in range(len(msa)):
                res = characterize_sequence(i)
                update_seq_conversion(res)
        else:
            pool = Pool(processes=processes, initializer=init_characterize_sequence, initargs=(msa, self.mapping,
                                                                                               self.alphabet_size))
            for i in range(len(msa)):
                pool.apply_async(characterize_sequence, (i,), callback=update_seq_conversion)
            pool.close()
            pool.join()
        char_bar.close()
        # Compute similarity between sequences using the sequences characterizations just completed.
        names = [s.id for s in msa]
        plain_identity = DistanceMatrix(names)
        psuedo_identity = DistanceMatrix(names)
        data_dict = {'Seq1': [], 'Seq2': [], 'Min_Seq_Length': [], 'Id_Count': [], 'Threshold_Count': []}
        seq_pairs = list(combinations(names, 2))
        dist_pbar = tqdm(total=len(seq_pairs), unit='distances')

        def update_distances(result):
            """
            This function receives the output of a single call to pairwise and stores that in the current distance
            matrix, and then updates the progress bar for the distance calculation process.

            Args:
                result (tuple): The return from similarity, consisting of the two identifiers for the sequences whose
                distance have been calculated, their identity similarity distance, their scoring matrix similarity
                distance, and the sequence length, identity count, and scoring matrix count determined during their
                scoring.
            """
            id1, id2, id_score, similarity_score, seq_length, identity_count, scoring_matrix_count = result
            plain_identity[id1, id2] = id_score
            psuedo_identity[id1, id2] = similarity_score
            data_dict['Seq1'].append(id1)
            data_dict['Seq2'].append(id2)
            data_dict['Min_Seq_Length'].append(seq_length)
            data_dict['Id_Count'].append(identity_count)
            data_dict['Threshold_Count'].append(scoring_matrix_count)
            dist_pbar.update(1)
            dist_pbar.refresh()

        if processes == 1:
            init_similarity(seq_conversion, threshold, self.scoring_matrix)
            for seq_id1, seq_id2 in seq_pairs:
                res = similarity(seq_id1, seq_id2)
                update_distances(res)
        else:
            pool2 = Pool(processes=processes, initializer=init_similarity,
                         initargs=(seq_conversion, threshold, self.scoring_matrix))
            for seq_id1, seq_id2 in seq_pairs:
                pool2.apply_async(similarity, (seq_id1, seq_id2), callback=update_distances)
            pool2.close()
            pool2.join()
        dist_pbar.close()
        return plain_identity, psuedo_identity, pd.DataFrame(data_dict), threshold


def init_identity(num_aln):
    """
    Initialize Identity

    This function initializes global variables required by identity for scoring the distance between two sequences.

    Args:
        num_aln (np.array): The numerical representation of a sequence alignment.
    """
    global numerical_alignment
    numerical_alignment = num_aln


def identity(i):
    """
    Identity

    This function calculates the distances between one sequences in an alignment and all other sequences in the
    alignment according to the identity distance metric.

    Args:
        i (int): The position of the sequence in the alignment for which to calculate identity distances.
    Return:
        int: The position for which identity distances were computed.
        np.array: The distances to all sequences in the alignment.
    """
    check = numerical_alignment - numerical_alignment[i]
    identity_counts = np.sum(check == 0, axis=1)
    fraction_identity = identity_counts / float(numerical_alignment.shape[1])
    distances = 1 - fraction_identity
    return i, distances


def init_pairwise(alpha_map, alpha_size, mod, scoring_mat):
    """
    Initialize Pairwise

    This function initializes global variables required by pairwise for scoring the distance between two sequences.

    Args:
        alpha_map (dict): The mapping from character to position in the scoring matrix.
        alpha_size (int): The size of the alphabet which the sequences being scored are defined by.
        mod (str): Which distance model to use when computing the distance.
        scoring_mat (np.array): The substitution/log odds table defining sequence distance scoring.
    """
    global pairwise_map, pairwise_alpha_size, pairwise_model, pairwise_scoring_mat
    pairwise_map = alpha_map
    pairwise_alpha_size = alpha_size
    pairwise_model = mod
    pairwise_scoring_mat = scoring_mat


def pairwise(seq1, seq2):
    """
    Pairwise

    This function scores two sequences by converting them via a mapping table, identifying positions to skip,
    retrieving the substitution scores, and calculating the final distance between the two sequences.

    Args:
        seq1 (Bio.Seq|Bio.SeqRecord|str): A protein or DNA sequence to score against seq2.
        seq2 (Bio.Seq|Bio.SeqRecord|str): A protein or DNA sequence to score against seq1.
    Returns:
        str: The identifier for seq1.
        str: The identifier for seq2.
        float: The distance between the two sequences.
    """
    # Convert seq1 to its indices in the scoring_matrix
    num_seq1 = convert_seq_to_numeric(seq1, mapping=pairwise_map)
    # Convert seq2 to its indices in the scoring_matrix
    num_seq2 = convert_seq_to_numeric(seq2, mapping=pairwise_map)
    threshold = pairwise_alpha_size
    if pairwise_model == 'identity':
        threshold += 1
    non_gap_pos1 = num_seq1 < threshold  # Find all positions which are not skip_letters in seq1
    non_gap_pos2 = num_seq2 < threshold  # Find all positions which are not skip_letters in seq2
    combined_non_gap_pos = non_gap_pos1 & non_gap_pos2  # Determine positions that are not skip_letters in either
    # Retrieve scores from scoring_matrix for all positions in the two sequences
    ij_scores = pairwise_scoring_mat[num_seq1[combined_non_gap_pos], num_seq2[combined_non_gap_pos]]
    # Retrieve all identity (max) scores for seq1
    ii_scores = pairwise_scoring_mat[num_seq1[combined_non_gap_pos], num_seq1[combined_non_gap_pos]]
    # Retrieve all identity (max) scores for seq2
    jj_scores = pairwise_scoring_mat[num_seq2[combined_non_gap_pos], num_seq2[combined_non_gap_pos]]
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
    return seq1.id, seq2.id, final_score


def init_characterize_sequence(aln, alpha_map, alpha_size):
    """
    Initialize Characterize Sequence

    Initializes global variables required to perform characterize_sequence calls.

    Args:
        aln (Bio.Align.MultipleSeqAlignment): The alignment for which distances are being calculated.
        alpha_map (dict): The dictionary mapping characters from the alignment/sequence alphabet to their positions in
        the scoring matrices.
        alpha_size (int): The size of the alphabet being for the alignment/sequence.
    """
    global alignment, mapping, alphabet_size
    alignment = aln
    mapping = alpha_map
    alphabet_size = alpha_size


def characterize_sequence(i):
    """
    Characterize Sequence

    This function converts one of the sequences in the alignment to its numerical representation and then determines
    which positions are not gaps and the number of non-gap positions in the sequence.

    Args:
        i (int): The sequence in the alignment to characterize.
    Return:
        str: The sequence identifier for the characterized sequence.
        int: The number of non-gap characters in the sequence.
        np.array: A boolean array identifying non-gap characters in the sequence.
        np.array: The numerical representation of the sequence.
    """
    seq_id = alignment[i].id
    # Convert seq i in the msa to a numerical representation (indices in scoring_matrix)
    num_repr = convert_seq_to_numeric(alignment[i], mapping)
    # Find all positions which are not gaps or skip_letters in seq1 and the resulting sequence length
    non_gap_pos = num_repr < alphabet_size
    non_gap_length = np.sum(non_gap_pos)
    return seq_id, non_gap_length, non_gap_pos, num_repr


def init_similarity(seq_con, cutoff, score_mat):
    """
    Initialize Similarity

    Initializes global variables requires to perform similarity calls.

    Args:
        seq_con (dict): The dictionary of sequence characterizations for the alignment for which distances are being
        computed.
        cutoff (float): The value at which two amino acids are or are not considered similar.
        score_mat (np.array): The scoring/substitution/logs odds matrix for distance calculation.
    """
    global seq_conversion, sim_threshold, scoring_matrix
    seq_conversion = seq_con
    sim_threshold = cutoff
    scoring_matrix = score_mat


def similarity(id1, id2):
    """
    Similarity

    Method to calculate the similarity between two sequences.

    Args:
        id1 (str): The sequence identifier for the first sequence in the alignment.
        id2 (str): The sequence identifier for the second sequence in the alignment.
    Returns:
        str: The sequence identifier for the first sequence in the alignment.
        str: The sequence identifier for the second sequence in the alignment.
        float: The similarity score computed based on the identity metric.
        float: The similarity score computed based on the scoring matrix.
        int: The non-gap sequence length used for the calculation (min of the two non-gap sequence lengths)
        int: Count of similar amino acids between the two sequences by the identity metric.
        int: Count of similar amino acids between the two sequences by the scoring matrix and cutoff.
    """
    # Determine positions that are not gaps or skip_letters in either sequence
    combined_non_gap_pos = seq_conversion[id1]['non_gap_pos'] & seq_conversion[id2]['non_gap_pos']
    # Subset the two sequences to only the positions which are not gaps or skip_letters
    final_seq1 = seq_conversion[id1]['num_repr'][combined_non_gap_pos]
    final_seq2 = seq_conversion[id2]['num_repr'][combined_non_gap_pos]
    # Count the number of positions which are identical between the two sequences
    identity_count = np.sum((final_seq1 - final_seq2) == 0)
    # Retrieve the scoring_matrix scores for the two sequences
    scores = scoring_matrix[final_seq1, final_seq2]
    # Find which scores pass the threshold (psuedo-identity) and count them
    passing_scores = scores >= sim_threshold
    scoring_matrix_count = np.sum(passing_scores)
    # Determine which sequence was the shorter of the two
    seq_length = min(seq_conversion[id1]['non_gap_length'], seq_conversion[id2]['non_gap_length'])
    # Compute the plain or psuedo identity score using the minimum sequence length
    identity_score = identity_count / float(seq_length)
    similarity_score = 1 - (scoring_matrix_count / float(seq_length))
    return id1, id2, identity_score, similarity_score, seq_length, identity_count, scoring_matrix_count


def convert_array_to_distance_matrix(array, names):
    """
    Convert Array To Distance Matrix

    This function converts a numpy.array of distances to a Bio.Phylo.TreeConstruction.DistanceMatrix object. This means
    turning the lower triangle of the array into a list of lists and initializing the DistanceMatrix object with that
    data and the identifiers of the sequences.

    Args:
        array (numpy.array): An array of distances, the lower triangle of which will be used to create a DistanceMatrix
        object.
        names (list): A list of sequence ids.
    Return:
        Bio.Phylo.TreeConstruction.DistanceMatrix: A DistanceMatrix object with the distances from the array.
    """
    indices = np.tril_indices(array.shape[0], 0, array.shape[1])
    list_of_lists = []
    for i in range(array.shape[0]):
        column_indices = indices[1][indices[0] == i]
        list_of_lists.append(list(array[i, column_indices]))
    dist_mat = DistanceMatrix(names=names, matrix=list_of_lists)
    return dist_mat
