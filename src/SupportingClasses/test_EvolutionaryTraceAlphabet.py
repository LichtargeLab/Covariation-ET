"""
Created on June 19, 2019

@author: Daniel Konecki
"""
import unittest
from unittest import TestCase
from Bio.Phylo.TreeConstruction import DistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACDNA, FullIUPACProtein, MultiPositionAlphabet


class TestEvolutionaryTraceAlphabet(TestCase):

    def test1_DNA_alphabet(self):
        dna_alphabet = FullIUPACDNA()
        self.assertEqual(dna_alphabet.size, 1)
        self.assertEqual(len(dna_alphabet.letters), 4)
        for char in DistanceCalculator.dna_alphabet:
            self.assertTrue(char in dna_alphabet.letters)

    def test2_protein_alphabet(self):
        protein_alphabet = FullIUPACProtein()
        self.assertEqual(protein_alphabet.size, 1)
        self.assertEqual(len(protein_alphabet.letters), 23)
        for char in DistanceCalculator.protein_alphabet:
            self.assertTrue(char in protein_alphabet.letters)

    def test3a_multi_position_alphabet(self):
        dna_alphabet = FullIUPACDNA()
        multi_position_alphabet = MultiPositionAlphabet(alphabet=FullIUPACDNA(), size=2)
        self.assertEqual(multi_position_alphabet.size, 2)
        self.assertEqual(len(multi_position_alphabet.letters), pow(4, 2))
        for i in range(4):
            for j in range(4):
                char = '{}{}'.format(dna_alphabet.letters[i], dna_alphabet.letters[j])
                self.assertTrue(char in multi_position_alphabet.letters)

    def test3b_multi_position_alphabet(self):
        protein_alphabet = FullIUPACProtein()
        multi_position_alphabet = MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2)
        self.assertEqual(multi_position_alphabet.size, 2)
        self.assertEqual(len(multi_position_alphabet.letters), pow(23, 2))
        for i in range(23):
            for j in range(23):
                char = '{}{}'.format(protein_alphabet.letters[i], protein_alphabet.letters[j])
                self.assertTrue(char in multi_position_alphabet.letters)

    def test3c_multi_position_alphabet(self):
        dna_alphabet = FullIUPACDNA()
        multi_position_alphabet = MultiPositionAlphabet(alphabet=FullIUPACDNA(), size=3)
        self.assertEqual(multi_position_alphabet.size, 3)
        self.assertEqual(len(multi_position_alphabet.letters), pow(4, 3))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    char = '{}{}{}'.format(dna_alphabet.letters[i], dna_alphabet.letters[j], dna_alphabet.letters[k])
                    self.assertTrue(char in multi_position_alphabet.letters)


if __name__ == '__main__':
    unittest.main()
