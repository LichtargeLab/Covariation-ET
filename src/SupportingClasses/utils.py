"""
Created on June 16, 2019

@author: Daniel Konecki
"""
import numpy as np
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
        set: The gap characters in this map.
        dict: Dictionary mapping a character to a number corresponding to its position in the alphabet and/or in the
        scoring/substitution matrix.
    """
    if isinstance(alphabet, Alphabet) or isinstance(alphabet, Gapped):
        letters = alphabet.letters
    elif type(alphabet) == list:
        letters = ''.join(alphabet)
    elif type(alphabet) == str:
        letters = alphabet
    else:
        raise ValueError("'alphabet' expects values of type Bio.Alphabet or list.")
    alphabet_size = len(letters)
    alpha_map = {char: i for i, char in enumerate(letters)}
    if skip_letters:
        skip_map = {char: alphabet_size + 1 for char in skip_letters}
        alpha_map.update(skip_map)
        curr_gaps = gap_characters - set(skip_letters)
    else:
        curr_gaps = gap_characters
    gap_map = {char: alphabet_size for char in curr_gaps}
    alpha_map.update(gap_map)
    return alphabet_size, curr_gaps, alpha_map


def convert_seq_to_numeric(seq, mapping):
    """
    Convert Seq To Numeric

    This function uses an alphabet mapping (see build_mapping) to convert a sequence to a 1D array of integers.

    Args:
        seq (Bio.Seq|Bio.SeqRecord|str): A protein or DNA sequence.
    Return:
        numpy.array: A 1D array containing the numerical representation of the passed in sequence.
    """
    numeric = [mapping[char] for char in seq]
    return np.array(numeric)
