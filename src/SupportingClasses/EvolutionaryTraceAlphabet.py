"""
Created on June 16, 2019

@author: Daniel Konecki
"""
from Bio import Alphabet
from Bio.Phylo.TreeConstruction import DistanceCalculator


class FullIUPACProtein(Alphabet.ProteinAlphabet):
    """
    This class represents the full set of characters represented in the protein substitution matrices. This is more than
    the IUPAC Protein alphabet but less than the ExtendedIUPACProtein alphabet.
    """
    letters = ''.join(DistanceCalculator.protein_alphabet)
    size = len(letters)


class FullIUPACDNA(Alphabet.DNAAlphabet):
    """
    This class represents the full set of characters represented in the DNA substitution matrices. This is more than the
    IUPAC Protein alphabet but less than the ExtendedIUPACProtein alphabet.
    """
    letters = ''.join(DistanceCalculator.dna_alphabet)
    size = len(letters)
