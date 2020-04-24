"""
Created on June 16, 2019

@author: Daniel Konecki
"""
import numpy as np
from scipy.stats import rankdata
from Bio.Alphabet import Alphabet, Gapped

# Common gap characters
gap_characters = {'-', '.', '*'}


def build_mapping(alphabet, skip_letters=None):
    """
    Build Mapping

    Constructs a dictionary mapping characters in the given alphabet to their position in the alphabet (which
    may correspond to their index in substitution matrices). It also maps gap characters and skip letters to positions
    outside of that range.

    Args:
        alphabet (Bio.Alphabet.Alphabet,list,str): An alphabet object with the letters that should be mapped, or a list
        or string containing all of the letters which should be mapped.
        skip_letters (list): Which characters to skip when scoring sequences in the alignment.
    Returns:
        int: The size of the alphabet (not including gaps or skip letters) represented by this map.
        set: The gap characters in this map (if there are gap characters in the provided alphabet then these will not be
        present in the returned set).
        dict: Dictionary mapping a character to a number corresponding to its position in the alphabet and/or in the
        scoring/substitution matrix.
        np.array: Array mapping a number to character such that the character can be decoded from a position in an
        array or table built based on the alphabet.
    """
    if isinstance(alphabet, Alphabet) or isinstance(alphabet, Gapped):
        letters = alphabet.letters
        character_size = alphabet.size
    elif type(alphabet) == list:
        letters = alphabet
        character_size = len(alphabet[0])
    elif type(alphabet) == str:
        letters = list(alphabet)
        character_size = 1
    else:
        raise ValueError("'alphabet' expects values of type Bio.Alphabet, list, or str.")
    if skip_letters:
        letters = [letter for letter in letters if letter not in skip_letters]
    alphabet_size = len(letters)
    alpha_map = {char: i for i, char in enumerate(letters)}
    curr_gaps = {g * character_size for g in gap_characters}
    if skip_letters:
        for sl in skip_letters:
            if len(sl) != character_size:
                raise ValueError(f'skip_letters contained a character {sl} which did not match the alphabet character '
                                 f'size: {character_size}')
        skip_map = {char: alphabet_size + 1 for char in skip_letters}
        alpha_map.update(skip_map)
        curr_gaps = curr_gaps - set(skip_letters)
    curr_gaps = curr_gaps - set(letters)
    gap_map = {char: alphabet_size for char in curr_gaps}
    alpha_map.update(gap_map)
    reverse_map = np.array(list(letters))
    return alphabet_size, curr_gaps, alpha_map, reverse_map


def convert_seq_to_numeric(seq, mapping):
    """
    Convert Seq To Numeric

    This function uses an alphabet mapping (see build_mapping) to convert a sequence to a 1D array of integers.

    Args:
        seq (Bio.Seq|Bio.SeqRecord|str): A protein or DNA sequence.
        mapping (dict): A dictionary mapping a character to its position in the alphabet (can be produced using
        build_mapping).
    Return:
        numpy.array: A 1D array containing the numerical representation of the passed in sequence.
    """
    numeric = [mapping[char] for char in seq]
    return np.array(numeric)


def compute_rank_and_coverage(seq_length, scores, pos_size, rank_type):
    """
    Compute Rank and Coverage

    This function generates rank and coverage values for a set of scores.

    Args:
        seq_length (int): The length of the sequences used to generate the scores for which rank and coverage are being
        computed.
        scores (np.array): A set of scores to rank and compute coverage for.
        pos_size (int): The dimensionality of the array (whether single, 1, positions or pair, 2, positions are
        being characterized).
        rank_type (str): Whether the optimal value of a set of scores is its 'max' or its 'min'.
    Returns:
        np.array: An array of ranks for the set of scores.
        np.array: An array of coverage scores (what percentile of values are at or below the given score).
    """
    if rank_type == 'max':
        weight = -1.0
    elif rank_type == 'min':
        weight = 1.0
    else:
        raise ValueError('No support for rank types other than max or min, {} provided'.format(rank_type))
    if pos_size == 1:
        indices = range(seq_length)
        normalization = float(seq_length)
        to_rank = scores * weight
        ranks = np.zeros(seq_length)
        coverages = np.zeros(seq_length)
    elif pos_size == 2:
        indices = np.triu_indices(seq_length, k=1)
        normalization = float(len(indices[0]))
        to_rank = scores[indices] * weight
        ranks = np.zeros((seq_length, seq_length))
        coverages = np.zeros((seq_length, seq_length))
    else:
        raise ValueError('Ranking not supported for position sizes other than 1 or 2, {} provided'.format(pos_size))
    ranks[indices] = rankdata(to_rank, method='dense')
    coverages[indices] = rankdata(to_rank, method='max')
    coverages /= normalization
    return ranks, coverages
