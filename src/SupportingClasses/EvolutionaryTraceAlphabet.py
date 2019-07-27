"""
Created on June 16, 2019

@author: Daniel Konecki
"""
from itertools import product
from Bio import Alphabet
from Bio.Phylo.TreeConstruction import DistanceCalculator


class FullIUPACProtein(Alphabet.ProteinAlphabet):
    """
    This class represents the full set of characters represented in the protein substitution matrices. This is more than
    the IUPAC Protein alphabet but less than the ExtendedIUPACProtein alphabet.
    """
    letters = ''.join(DistanceCalculator.protein_alphabet)
    size = 1


class FullIUPACDNA(Alphabet.DNAAlphabet):
    """
    This class represents the full set of characters represented in the DNA substitution matrices.
    """
    letters = ''.join(DistanceCalculator.dna_alphabet)
    size = 1


class MultiPositionAlphabet(Alphabet.Alphabet):
    """
    This class represents the full set of character combinations represented by an alphabet which spans multiple
    positions. For example if you wanted to look at pairs of amino acids this would have the size n!/(n-k)! where n is
    the size of the amino acid alphabet and k is 2 (since you are looking at pairs).
    """

    def __init__(self, alphabet, size):
        """
        The initialization for a multiple position alphabet starting from some initial alphabet.

        Args:
            alphabet (Bio.Alphabet.Alphabet/Bio.Alphabet.Gapped): The alphabet which should be expanded to consider
            multiple positions.
            size (int): The number of positions an alphabet should cover.
        """
        self.size = size
        self.letters = [''.join(list(x)) for x in product(alphabet.letters, repeat=size)]
