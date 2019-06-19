"""
Created on June 19, 2019

@author: Daniel Konecki
"""
from unittest import TestCase
from Bio.Phylo.TreeConstruction import DistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACDNA, FullIUPACProtein


class TestEvolutionaryTraceAlphabet(TestCase):

    def test_DNA_alphabet(self):
        dna_alphabet = FullIUPACDNA()
        self.assertEqual(dna_alphabet.size, 4)
        for char in DistanceCalculator.dna_alphabet:
            self.assertTrue(char in dna_alphabet.letters)

    def test_protein_alphabet(self):
        protein_alphabet = FullIUPACProtein()
        self.assertEqual(protein_alphabet.size, 23)
        for char in DistanceCalculator.protein_alphabet:
            self.assertTrue(char in protein_alphabet.letters)