"""
Created on June 19, 2019

@author: Daniel Konecki
"""
import unittest
from unittest import TestCase
from Bio.Phylo.TreeConstruction import DistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACDNA, FullIUPACProtein, MultiPositionAlphabet


class TestEvolutionaryTraceAlphabet(TestCase):

    def evaluate_alphabet(self, alphabet, char_size, alpha_size, expected_characters):
        self.assertEqual(alphabet.size, char_size)
        self.assertEqual(len(alphabet.letters), alpha_size)
        for char in expected_characters:
            self.assertTrue(char in alphabet.letters)

    def test1_DNA_alphabet(self):
        self.evaluate_alphabet(alphabet=FullIUPACDNA(), char_size=1, alpha_size=4,
                               expected_characters=DistanceCalculator.dna_alphabet)

    def test2_protein_alphabet(self):
        self.evaluate_alphabet(alphabet=FullIUPACProtein(), char_size=1, alpha_size=23,
                               expected_characters=DistanceCalculator.protein_alphabet)

    def test3a_multi_position_alphabet(self):
        dna_alphabet = FullIUPACDNA()
        multi_position_alphabet = MultiPositionAlphabet(alphabet=dna_alphabet, size=2)
        expected_chars = []
        for i in range(len(dna_alphabet.letters)):
            for j in range(len(dna_alphabet.letters)):
                expected_chars.append('{}{}'.format(dna_alphabet.letters[i], dna_alphabet.letters[j]))
        self.evaluate_alphabet(alphabet=multi_position_alphabet, char_size=2, alpha_size=pow(4, 2),
                               expected_characters=expected_chars)

    def test3b_multi_position_alphabet(self):
        protein_alphabet = FullIUPACProtein()
        multi_position_alphabet = MultiPositionAlphabet(alphabet=protein_alphabet, size=2)
        expected_chars = []
        for i in range(len(protein_alphabet.letters)):
            for j in range(len(protein_alphabet.letters)):
                expected_chars.append('{}{}'.format(protein_alphabet.letters[i], protein_alphabet.letters[j]))
        self.evaluate_alphabet(alphabet=multi_position_alphabet, char_size=2, alpha_size=pow(23, 2),
                               expected_characters=expected_chars)

    def test3c_multi_position_alphabet(self):
        dna_alphabet = FullIUPACDNA()
        multi_position_alphabet = MultiPositionAlphabet(alphabet=dna_alphabet, size=3)
        expected_chars = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    expected_chars.append('{}{}{}'.format(dna_alphabet.letters[i], dna_alphabet.letters[j],
                                                          dna_alphabet.letters[k]))
        self.evaluate_alphabet(alphabet=multi_position_alphabet, char_size=3, alpha_size=pow(4, 3),
                               expected_characters=expected_chars)


if __name__ == '__main__':
    unittest.main()
