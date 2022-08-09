"""
Created on June 16, 2019

@author: Daniel Konecki
"""
from itertools import product
from Bio import Alphabet
from Bio.Phylo.TreeConstruction import DistanceCalculator

class PPI_FullIUPACProtein(Alphabet.Alphabet):
    """
    This class represents the full set of characters represented in the protein substitution matrices. This is more than
    the IUPAC Protein alphabet but less than the ExtendedIUPACProtein alphabet.
    """
    letters = ["-", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
               "X", "B", "Z", ":"]
    size = 1

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
            alphabet (str/list/ Bio.Alphabet.Alphabet/Bio.Alphabet.Gapped): The alphabet which should be expanded to
            consider multiple positions.
            size (int): The number of positions by which to increase the existing alphabet. E.g. if an alphabet with
            size 1 is provided and size is set to 2 then the size of the new alphabet object will be 2, but if the
            initial alphabet has size 2 and size two is specified, then the new alphabet will have size 4.
        """
        try:
            # Assuming an Alphabet object was provided retrieve the size and characters
            prev_size = alphabet.size
            characters = alphabet.letters
        except AttributeError:
            try:
                # If size could not be retrieved attempt to retrieve just letters
                prev_size = len(alphabet.letters[0])
                characters = alphabet.letters
            except AttributeError:
                # If letters could not be retrieved assume a string or list has been provided and use the alphabet
                # itself as the set of characters and the size of the first element (1 for strings and variable for
                # lists) as the size of the input alphabet.
                prev_size = len(alphabet[0])
                characters = alphabet
        self.size = size * prev_size
        self.letters = [''.join(list(x)) for x in product(characters, repeat=size)]
