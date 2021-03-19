"""
Created on June 19, 2019

@author: Daniel Konecki
"""
import os
import sys
import unittest
from unittest import TestCase

#
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
source_code_path = os.path.join(os.environ.get('PROJECT_PATH'), 'src')
# Add the project path to the python path so the required clases can be imported
if source_code_path not in sys.path:
    sys.path.append(os.path.join(os.environ.get('PROJECT_PATH'), 'src'))
#

from SupportingClasses.EvolutionaryTraceAlphabet import FullIUPACDNA, FullIUPACProtein, MultiPositionAlphabet

single_letter_dna = {'A', 'C', 'G', 'T'}
single_letter_protein = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                         'Y', 'B', 'X', 'Z'}
pair_dna ={'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT'}
quad_dna = {'AAAA', 'AAAC', 'AAAG', 'AAAT', 'AACA', 'AACC', 'AACG', 'AACT', 'AAGA', 'AAGC', 'AAGG', 'AAGT', 'AATA',
            'AATC', 'AATG', 'AATT', 'ACAA', 'ACAC', 'ACAG', 'ACAT', 'ACCA', 'ACCC', 'ACCG', 'ACCT', 'ACGA', 'ACGC',
            'ACGG', 'ACGT', 'ACTA', 'ACTC', 'ACTG', 'ACTT', 'AGAA', 'AGAC', 'AGAG', 'AGAT', 'AGCA', 'AGCC', 'AGCG',
            'AGCT', 'AGGA', 'AGGC', 'AGGG', 'AGGT', 'AGTA', 'AGTC', 'AGTG', 'AGTT', 'ATAA', 'ATAC', 'ATAG', 'ATAT',
            'ATCA', 'ATCC', 'ATCG', 'ATCT', 'ATGA', 'ATGC', 'ATGG', 'ATGT', 'ATTA', 'ATTC', 'ATTG', 'ATTT', 'CAAA',
            'CAAC', 'CAAG', 'CAAT', 'CACA', 'CACC', 'CACG', 'CACT', 'CAGA', 'CAGC', 'CAGG', 'CAGT', 'CATA', 'CATC',
            'CATG', 'CATT', 'CCAA', 'CCAC', 'CCAG', 'CCAT', 'CCCA', 'CCCC', 'CCCG', 'CCCT', 'CCGA', 'CCGC', 'CCGG',
            'CCGT', 'CCTA', 'CCTC', 'CCTG', 'CCTT', 'CGAA', 'CGAC', 'CGAG', 'CGAT', 'CGCA', 'CGCC', 'CGCG', 'CGCT',
            'CGGA', 'CGGC', 'CGGG', 'CGGT', 'CGTA', 'CGTC', 'CGTG', 'CGTT', 'CTAA', 'CTAC', 'CTAG', 'CTAT', 'CTCA',
            'CTCC', 'CTCG', 'CTCT', 'CTGA', 'CTGC', 'CTGG', 'CTGT', 'CTTA', 'CTTC', 'CTTG', 'CTTT', 'GAAA', 'GAAC',
            'GAAG', 'GAAT', 'GACA', 'GACC', 'GACG', 'GACT', 'GAGA', 'GAGC', 'GAGG', 'GAGT', 'GATA', 'GATC', 'GATG',
            'GATT', 'GCAA', 'GCAC', 'GCAG', 'GCAT', 'GCCA', 'GCCC', 'GCCG', 'GCCT', 'GCGA', 'GCGC', 'GCGG', 'GCGT',
            'GCTA', 'GCTC', 'GCTG', 'GCTT', 'GGAA', 'GGAC', 'GGAG', 'GGAT', 'GGCA', 'GGCC', 'GGCG', 'GGCT', 'GGGA',
            'GGGC', 'GGGG', 'GGGT', 'GGTA', 'GGTC', 'GGTG', 'GGTT', 'GTAA', 'GTAC', 'GTAG', 'GTAT', 'GTCA', 'GTCC',
            'GTCG', 'GTCT', 'GTGA', 'GTGC', 'GTGG', 'GTGT', 'GTTA', 'GTTC', 'GTTG', 'GTTT', 'TAAA', 'TAAC', 'TAAG',
            'TAAT', 'TACA', 'TACC', 'TACG', 'TACT', 'TAGA', 'TAGC', 'TAGG', 'TAGT', 'TATA', 'TATC', 'TATG', 'TATT',
            'TCAA', 'TCAC', 'TCAG', 'TCAT', 'TCCA', 'TCCC', 'TCCG', 'TCCT', 'TCGA', 'TCGC', 'TCGG', 'TCGT', 'TCTA',
            'TCTC', 'TCTG', 'TCTT', 'TGAA', 'TGAC', 'TGAG', 'TGAT', 'TGCA', 'TGCC', 'TGCG', 'TGCT', 'TGGA', 'TGGC',
            'TGGG', 'TGGT', 'TGTA', 'TGTC', 'TGTG', 'TGTT', 'TTAA', 'TTAC', 'TTAG', 'TTAT', 'TTCA', 'TTCC', 'TTCG',
            'TTCT', 'TTGA', 'TTGC', 'TTGG', 'TTGT', 'TTTA', 'TTTC', 'TTTG', 'TTTT'}


def check_alphabet(test, alpha, expected_size, expected_count, expected_chars):
    test.assertEqual(alpha.size, expected_size)
    for char in expected_chars:
        test.assertTrue(char in alpha.letters)
    test.assertEqual(len(alpha.letters), expected_count)


class TestSingleLetterAlphabets(TestCase):

    def test_DNA_alphabet(self):
        dna_alpha = FullIUPACDNA()
        check_alphabet(test=self, alpha=dna_alpha, expected_size=1, expected_count=4,
                       expected_chars=single_letter_dna)

    def test_protein_alphabet(self):
        protein_alpha = FullIUPACProtein()
        check_alphabet(test=self, alpha=protein_alpha, expected_size=1, expected_count=23,
                       expected_chars=single_letter_protein)


class TestMultiLetterAlphabets(TestCase):

    def test_multi_letter_alphabet_size_1(self):
        dna_alpha = FullIUPACDNA()
        multi_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=1)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=1, expected_count=4,
                       expected_chars=single_letter_dna)

    def test_multi_letter_alphabet_size_2(self):
        dna_alpha = FullIUPACDNA()
        multi_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=2)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=2, expected_count=16,
                       expected_chars=pair_dna)

    def test_multi_letter_alphabet_size_4(self):
        dna_alpha = FullIUPACDNA()
        multi_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=4)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=4, expected_count=256,
                       expected_chars=quad_dna)

    def test_multi_letter_alphabet_size_4_from_size_2(self):
        dna_alpha = FullIUPACDNA()
        pair_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=2)
        multi_alpha = MultiPositionAlphabet(alphabet=pair_alpha, size=2)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=4, expected_count=256,
                       expected_chars=quad_dna)

    def test_multi_letter_alphabet_size_1_from_string(self):
        dna_alpha = ''.join(FullIUPACDNA().letters)
        multi_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=1)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=1, expected_count=4,
                       expected_chars=single_letter_dna)

    def test_multi_letter_alphabet_size_2_from_string(self):
        dna_alpha = ''.join(FullIUPACDNA().letters)
        multi_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=2)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=2, expected_count=16,
                       expected_chars=pair_dna)

    def test_multi_letter_alphabet_size_4_from_string(self):
        dna_alpha = ''.join(FullIUPACDNA().letters)
        multi_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=4)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=4, expected_count=256,
                       expected_chars=quad_dna)

    def test_multi_letter_alphabet_size_1_from_list(self):
        dna_alpha = list(FullIUPACDNA().letters)
        multi_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=1)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=1, expected_count=4,
                       expected_chars=single_letter_dna)

    def test_multi_letter_alphabet_size_2_from_list(self):
        dna_alpha = list(FullIUPACDNA().letters)
        multi_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=2)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=2, expected_count=16,
                       expected_chars=pair_dna)

    def test_multi_letter_alphabet_size_4_from_list(self):
        dna_alpha = list(FullIUPACDNA().letters)
        multi_alpha = MultiPositionAlphabet(alphabet=dna_alpha, size=4)
        check_alphabet(test=self, alpha=multi_alpha, expected_size=4, expected_count=256,
                       expected_chars=quad_dna)


if __name__ == '__main__':
    unittest.main()
