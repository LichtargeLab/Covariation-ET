import os
import sys
import math
import numpy as np
from pymol import cmd
from shutil import rmtree
import unittest
from unittest import TestCase
from itertools import combinations
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

#
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
source_code_path = os.path.join(os.environ.get('PROJECT_PATH'), 'src')
# Add the project path to the python path so the required classes can be imported
if source_code_path not in sys.path:
    sys.path.append(os.path.join(os.environ.get('PROJECT_PATH'), 'src'))
#

from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.utils import compute_rank_and_coverage
from SupportingClasses.EvolutionaryTraceAlphabet import FullIUPACProtein
from Evaluation.Scorer import Scorer, init_scw_z_score_selection, scw_z_score_selection
from Evaluation.SequencePDBMap import SequencePDBMap
from Evaluation.SelectionClusterWeighting import SelectionClusterWeighting
from Testing.test_Base import (protein_seq1, protein_seq2, protein_seq3, protein_aln, write_out_temp_fn)
from Testing.test_PDBReference import chain_a_pdb_partial2, chain_a_pdb_partial, chain_b_pdb, chain_b_pdb_partial

pro_str = f'>{protein_seq1.id}\n{protein_seq1.seq}\n>{protein_seq2.id}\n{protein_seq2.seq}\n>{protein_seq3.id}\n{protein_seq3.seq}'
pro_str_long = f'>{protein_seq1.id}\n{protein_seq1.seq*10}\n>{protein_seq2.id}\n{protein_seq2.seq*10}\n>{protein_seq3.id}\n{protein_seq3.seq*10}'
aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
protein_aln1 = protein_aln.remove_gaps()
protein_aln1.file_name = aln_fn
protein_aln2 = SeqAlignment(query_id='seq2', file_name=aln_fn)
protein_aln2.import_alignment()
protein_aln2 = protein_aln2.remove_gaps()
protein_aln3 = SeqAlignment(query_id='seq3', file_name=aln_fn)
protein_aln3.import_alignment()
protein_aln3 = protein_aln3.remove_gaps()
os.remove(aln_fn)
pro_pdb1 = chain_a_pdb_partial2 + chain_a_pdb_partial
pro_pdb1_scramble = chain_a_pdb_partial + chain_a_pdb_partial2

chain_a_pdb_partial3 = 'ATOM      1  N  AMET A   1     152.897  26.590  66.235  1.00 57.82           N  \n'\
                       'ATOM      1  N  BMET A   1     150.123  25.456  70.789  1.00 57.82           N  \n'\
                       'ATOM      2  CA AMET A   1     153.488  26.592  67.584  1.00 57.79           C  \n'\
                       'ATOM      2  CA BMET A   1     155.123  30.456  65.789  1.00 57.79           C  \n'\
                       'ATOM      3  C  AMET A   1     153.657  28.043  68.066  1.00 57.26           C  \n'\
                       'ATOM      3  C  BMET A   1     155.123  25.456  70.789  1.00 57.26           C  \n'\
                       'ATOM      4  O  AMET A   1     153.977  28.924  67.266  1.00 57.37           O  \n'\
                       'ATOM      4  O  BMET A   1     150.123  30.456  70.789  1.00 57.37           O  \n'\
                       'ATOM      5  CB AMET A   1     154.843  25.881  67.544  1.00 58.16           C  \n'\
                       'ATOM      5  CB BMET A   1     155.123  25.456  65.789  1.00 58.16           C  \n'\
                       'ATOM      6  CG AMET A   1     155.689  25.983  68.820  1.00 59.67           C  \n'\
                       'ATOM      6  CG BMET A   1     160.123  20.456  75.789  1.00 59.67           C  \n'\
                       'ATOM      7  SD AMET A   1     157.418  25.517  68.551  1.00 62.17           S  \n'\
                       'ATOM      7  SD BMET A   1     165.123  35.456  55.789  1.00 62.17           S  \n'\
                       'ATOM      8  CE AMET A   1     158.062  26.956  67.686  1.00 61.68           C  \n'\
                       'ATOM      8  CE BMET A   1     175.123  50.456  75.789  1.00 61.68           C  \n'

pro_pdb_1_alt_locs = chain_a_pdb_partial3 + chain_a_pdb_partial
pro_pdb2 = chain_b_pdb
pro_pdb2_scramble = chain_b_pdb + chain_b_pdb_partial
pro_pdb_full = pro_pdb1 + pro_pdb2
pro_pdb_full_scramble = pro_pdb1_scramble + pro_pdb2_scramble

CONTACT_DISTANCE2 = 16


def et_calcDist(atoms1, atoms2):
    """return smallest distance (squared) between two groups of atoms"""
    # (not distant by more than ~100 A)
    # mind2=CONTACT_DISTANCE2+100
    c1 = atoms1[0]  # atoms must not be empty
    c2 = atoms2[0]
    mind2 = (c1[0] - c2[0]) * (c1[0] - c2[0]) + \
            (c1[1] - c2[1]) * (c1[1] - c2[1]) + \
            (c1[2] - c2[2]) * (c1[2] - c2[2])
    for c1 in atoms1:
        for c2 in atoms2:
            d2 = (c1[0] - c2[0]) * (c1[0] - c2[0]) + \
                 (c1[1] - c2[1]) * (c1[1] - c2[1]) + \
                 (c1[2] - c2[2]) * (c1[2] - c2[2])
            if d2 < mind2:
                mind2 = d2
    return mind2  # Square of distance between most proximate atoms


def et_computeAdjacency(chain, mapping):
    """Compute the pairs of contacting residues
    A(i,j) implemented as a hash of hash of residue numbers"""
    three2one = {
        "ALA": 'A',
        "ARG": 'R',
        "ASN": 'N',
        "ASP": 'D',
        "CYS": 'C',
        "GLN": 'Q',
        "GLU": 'E',
        "GLY": 'G',
        "HIS": 'H',
        "ILE": 'I',
        "LEU": 'L',
        "LYS": 'K',
        "MET": 'M',
        "PHE": 'F',
        "PRO": 'P',
        "SER": 'S',
        "THR": 'T',
        "TRP": 'W',
        "TYR": 'Y',
        "VAL": 'V',
        "A": "A",
        "G": "G",
        "T": "T",
        "U": "U",
        "C": "C", }

    ResAtoms = {}
    for residue in chain:
        try:
            aa = three2one[residue.get_resname()]
        except KeyError:
            continue
        # resi = residue.get_id()[1]
        resi = mapping[residue.get_id()[1]]
        for atom in residue:
            try:
                # ResAtoms[resi - 1].append(atom.coord)
                ResAtoms[resi].append(atom.coord)
            except KeyError:
                # ResAtoms[resi - 1] = [atom.coord]
                ResAtoms[resi] = [atom.coord]
    A = {}
    for resi in ResAtoms.keys():
        for resj in ResAtoms.keys():
            if resi < resj:
                curr_dist = et_calcDist(ResAtoms[resi], ResAtoms[resj])
                if curr_dist < CONTACT_DISTANCE2:
                    try:
                        A[resi][resj] = 1
                    except KeyError:
                        A[resi] = {resj: 1}
    return A, ResAtoms


def et_calcZScore(reslist, L, A, bias=1):
    """Calculate z-score (z_S) for residue selection reslist=[1,2,...]
    z_S = (w-<w>_S)/sigma_S
    The steps are:
    1. Calculate Selection Clustering Weight (SCW) 'w'
    2. Calculate mean SCW (<w>_S) in the ensemble of random
    selections of len(reslist) residues
    3. Calculate mean square SCW (<w^2>_S) and standard deviation (sigma_S)
    Reference: Mihalek, Res, Yao, Lichtarge (2003)

    reslist - a list of int's of protein residue numbers, e.g. ET residues
    L - length of protein
    A - the adjacency matrix implemented as a dictionary. The first key is related to the second key by resi<resj.
    bias - option to calculate with bias or nobias (j-i factor)"""
    w = 0
    if bias == 1:
        for resi in reslist:
            for resj in reslist:
                if resi < resj:
                    try:
                        Aij = A[resi][resj]  # A(i,j)==1
                        w += (resj - resi)
                    except KeyError:
                        pass
    elif bias == 0:
        for resi in reslist:
            for resj in reslist:
                if resi < resj:
                    try:
                        Aij = A[resi][resj]  # A(i,j)==1
                        w += 1
                    except KeyError:
                        pass
    M = len(reslist)
    pi1 = M * (M - 1.0) / (L * (L - 1.0))
    pi2 = pi1 * (M - 2.0) / (L - 2.0)
    pi3 = pi2 * (M - 3.0) / (L - 3.0)
    w_ave = 0
    w2_ave = 0
    cases = {'Case1': 0, 'Case2': 0, 'Case3': 0}
    if bias == 1:
        for resi, neighborsj in A.items():
            for resj in neighborsj:
                w_ave += (resj - resi)
                for resk, neighborsl in A.items():
                    for resl in neighborsl:
                        if (resi == resk and resj == resl) or \
                                (resi == resl and resj == resk):
                            w2_ave += pi1 * (resj - resi) * (resl - resk)
                            cases['Case1'] += (resj - resi) * (resl - resk)
                        elif (resi == resk) or (resj == resl) or \
                                (resi == resl) or (resj == resk):
                            w2_ave += pi2 * (resj - resi) * (resl - resk)
                            cases['Case2'] += (resj - resi) * (resl - resk)
                        else:
                            w2_ave += pi3 * (resj - resi) * (resl - resk)
                            cases['Case3'] += (resj - resi) * (resl - resk)
    elif bias == 0:
        for resi, neighborsj in A.items():
            w_ave += len(neighborsj)
            for resj in neighborsj:
                for resk, neighborsl in A.items():
                    for resl in neighborsl:
                        if (resi == resk and resj == resl) or \
                                (resi == resl and resj == resk):
                            w2_ave += pi1
                            cases['Case1'] += 1
                        elif (resi == resk) or (resj == resl) or \
                                (resi == resl) or (resj == resk):
                            w2_ave += pi2
                            cases['Case2'] += 1
                        else:
                            w2_ave += pi3
                            cases['Case3'] += 1
    w_ave = w_ave * pi1
    # print('EXPECTED M: ', M)
    # print('EXPECTED L: ', L)
    # print('EXPECTED W: ', w)
    # print('EXPECTED RES LIST: ', sorted(reslist))
    # print('EXPECTED W_AVE: ', w_ave)
    # print('EXPECTED W_AVE^2: ', (w_ave * w_ave))
    # print('EXPECTED W^2_AVE: ', w2_ave)
    # print('EXPECTED DIFF: ', w2_ave - w_ave * w_ave)
    # print('EXPECTED DIFF2: ', w2_ave - (w_ave * w_ave))
    sigma = math.sqrt(w2_ave - w_ave * w_ave)
    if sigma == 0:
        return M, L, pi1, pi2, pi3, 'NA', w, w_ave, w2_ave, sigma, cases
    return M, L, pi1, pi2, pi3, (w - w_ave) / sigma, w, w_ave, w2_ave, sigma, cases


class TestScorerInit(TestCase):

    def evaluate_init(self, expected_query, expected_aln, expected_structure, expected_cutoff, expected_chain):
        scorer = Scorer(query=expected_query, seq_alignment=expected_aln, pdb_reference=expected_structure,
                        chain=expected_chain)
        self.assertIsInstance(scorer.query_pdb_mapper, SequencePDBMap)
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNone(scorer.data)

    def test_init_aln_file_pdb_file_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        self.evaluate_init(expected_query='seq1', expected_aln=aln_fn, expected_structure=pdb_fn, expected_cutoff=8.0,
                           expected_chain='A')
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_aln_obj_pdb_file_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        self.evaluate_init(expected_query='seq1', expected_aln=aln_obj, expected_structure=pdb_fn, expected_cutoff=8.0,
                           expected_chain='A')
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_aln_file_pdb_obj_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        self.evaluate_init(expected_query='seq1', expected_aln=aln_fn, expected_structure=pdb_obj, expected_cutoff=8.0,
                           expected_chain='A')
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_aln_file_pdb_file_no_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        self.evaluate_init(expected_query='seq1', expected_aln=aln_fn, expected_structure=pdb_fn, expected_cutoff=8.0,
                           expected_chain=None)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_failure_empty(self):
        with self.assertRaises(TypeError):
            Scorer()


class TestScorerFit(TestCase):

    def setUp(self):
        self.expected_mapping_A = {0: 0, 1: 1, 2: 2}
        self.expected_mapping_B = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        self.expected_mapping_mismatch = {0: 0, 1: 3, 2: 4}
        self.expected_pro_seq_2 = SeqRecord(id='seq2', seq=Seq('MTREE', alphabet=FullIUPACProtein()))
        self.expected_pro_seq_3 = SeqRecord(id='seq3', seq=Seq('MFREE', alphabet=FullIUPACProtein()))

    def evaluate_fit(self, scorer):
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        scorer.fit()
        self.assertTrue(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNone(scorer.data)
        scorer.query_pdb_mapper.best_chain = None
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        scorer.fit()
        self.assertTrue(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNone(scorer.data)
        scorer.query_pdb_mapper.query_pdb_mapping = None
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        scorer.fit()
        self.assertTrue(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNone(scorer.data)
        scorer.query_pdb_mapper.pdb_query_mapping = None
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        scorer.fit()
        self.assertTrue(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNone(scorer.data)

    def test_fit_aln_file_pdb_file_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        scorer = Scorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb_fn, chain='A')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain='A')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb, chain='A')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, chain='A')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, chain=None)
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain='A')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain=None)
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = Scorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, chain=None)
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain=None)
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = Scorer(query='seq2', seq_alignment=aln_fn, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq2', seq_alignment=aln_fn, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb, chain=None)
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb_fn, chain=None)
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        scorer = Scorer(query='seq3', seq_alignment=aln_fn, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq3', seq_alignment=aln_fn, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb, chain=None)
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full_scramble)
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full_scramble)
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb_fn, chain=None)
        self.evaluate_fit(scorer=scorer)
        os.remove(aln_fn)
        os.remove(pdb_fn)


class TestScorerGetCoords(TestCase):  # get_all, get_c_alpha, get_c_beta, get_coords

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        protein_aln1.write_out_alignment(aln_fn)
        cls.expected_chain_a_coords = {1: {'all': [np.array([152.897, 26.590, 66.235], dtype=np.float32),
                                                   np.array([153.488, 26.592, 67.584], dtype=np.float32),
                                                   np.array([153.657, 28.043, 68.066], dtype=np.float32),
                                                   np.array([153.977, 28.924, 67.266], dtype=np.float32),
                                                   np.array([154.843, 25.881, 67.544], dtype=np.float32),
                                                   np.array([155.689, 25.983, 68.820], dtype=np.float32),
                                                   np.array([157.418, 25.517, 68.551], dtype=np.float32),
                                                   np.array([158.062, 26.956, 67.686], dtype=np.float32)],
                                           'alpha': [np.array([153.488, 26.592, 67.584], dtype=np.float32)],
                                           'beta': [np.array([154.843, 25.881, 67.544], dtype=np.float32)]},
                                       2: {'all': [np.array([153.913, 21.571, 52.586], dtype=np.float32),
                                                   np.array([153.617, 20.553, 53.599], dtype=np.float32),
                                                   np.array([153.850, 21.067, 55.018], dtype=np.float32),
                                                   np.array([153.177, 20.639, 55.960], dtype=np.float32),
                                                   np.array([154.431, 19.277, 53.350], dtype=np.float32),
                                                   np.array([153.767, 18.286, 52.392], dtype=np.float32),
                                                   np.array([153.675, 18.807, 50.965], dtype=np.float32),
                                                   np.array([152.542, 19.056, 50.495], dtype=np.float32),
                                                   np.array([154.735, 18.978, 50.321], dtype=np.float32)],
                                           'alpha': [np.array([153.617, 20.553, 53.599], dtype=np.float32)],
                                           'beta': [np.array([154.431, 19.277, 53.350], dtype=np.float32)]},
                                       3: {'all': [np.array([24.499, 13.739, 37.648], dtype=np.float32),
                                                   np.array([24.278, 13.068, 38.914], dtype=np.float32),
                                                   np.array([22.973, 13.448, 39.580], dtype=np.float32),
                                                   np.array([22.188, 12.566, 39.933], dtype=np.float32),
                                                   np.array([25.405, 13.382, 39.929], dtype=np.float32),
                                                   np.array([25.336, 12.578, 41.212], dtype=np.float32),
                                                   np.array([26.603, 13.084, 39.227], dtype=np.float32)],
                                           'alpha': [np.array([24.278, 13.068, 38.914], dtype=np.float32)],
                                           'beta': [np.array([25.405, 13.382, 39.929], dtype=np.float32)]}}
        cls.expected_chain_a_alt_coords = {1: {'all': [np.array([152.897, 26.590, 66.235], dtype=np.float32),
                                                       np.array([150.123, 25.456, 70.789], dtype=np.float32),
                                                       np.array([153.488, 26.592, 67.584], dtype=np.float32),
                                                       np.array([155.123, 30.456, 65.789], dtype=np.float32),
                                                       np.array([153.657, 28.043, 68.066], dtype=np.float32),
                                                       np.array([155.123, 25.456, 70.789], dtype=np.float32),
                                                       np.array([153.977, 28.924, 67.266], dtype=np.float32),
                                                       np.array([150.123, 30.456, 70.789], dtype=np.float32),
                                                       np.array([154.843, 25.881, 67.544], dtype=np.float32),
                                                       np.array([155.123, 25.456, 65.789], dtype=np.float32),
                                                       np.array([155.689, 25.983, 68.820], dtype=np.float32),
                                                       np.array([160.123, 20.456, 75.789], dtype=np.float32),
                                                       np.array([157.418, 25.517, 68.551], dtype=np.float32),
                                                       np.array([165.123, 35.456, 55.789], dtype=np.float32),
                                                       np.array([158.062, 26.956, 67.686], dtype=np.float32),
                                                       np.array([175.123, 50.456, 75.789], dtype=np.float32)],
                                               'alpha': [np.array([153.488, 26.592, 67.584], dtype=np.float32),
                                                         np.array([155.123, 30.456, 65.789], dtype=np.float32)],
                                               'beta': [np.array([154.843, 25.881, 67.544], dtype=np.float32),
                                                        np.array([155.123, 25.456, 65.789], dtype=np.float32)]},
                                           2: {'all': [np.array([153.913, 21.571, 52.586], dtype=np.float32),
                                                       np.array([153.617, 20.553, 53.599], dtype=np.float32),
                                                       np.array([153.850, 21.067, 55.018], dtype=np.float32),
                                                       np.array([153.177, 20.639, 55.960], dtype=np.float32),
                                                       np.array([154.431, 19.277, 53.350], dtype=np.float32),
                                                       np.array([153.767, 18.286, 52.392], dtype=np.float32),
                                                       np.array([153.675, 18.807, 50.965], dtype=np.float32),
                                                       np.array([152.542, 19.056, 50.495], dtype=np.float32),
                                                       np.array([154.735, 18.978, 50.321], dtype=np.float32)],
                                               'alpha': [np.array([153.617, 20.553, 53.599], dtype=np.float32)],
                                               'beta': [np.array([154.431, 19.277, 53.350], dtype=np.float32)]},
                                           3: {'all': [np.array([24.499, 13.739, 37.648], dtype=np.float32),
                                                       np.array([24.278, 13.068, 38.914], dtype=np.float32),
                                                       np.array([22.973, 13.448, 39.580], dtype=np.float32),
                                                       np.array([22.188, 12.566, 39.933], dtype=np.float32),
                                                       np.array([25.405, 13.382, 39.929], dtype=np.float32),
                                                       np.array([25.336, 12.578, 41.212], dtype=np.float32),
                                                       np.array([26.603, 13.084, 39.227], dtype=np.float32)],
                                               'alpha': [np.array([24.278, 13.068, 38.914], dtype=np.float32)],
                                               'beta': [np.array([25.405, 13.382, 39.929], dtype=np.float32)]}}
        cls.expected_chain_b_coords = {1: {'all': [np.array([152.897, 26.590, 66.235], dtype=np.float32),
                                                   np.array([153.488, 26.592, 67.584], dtype=np.float32),
                                                   np.array([153.657, 28.043, 68.066], dtype=np.float32),
                                                   np.array([153.977, 28.924, 67.266], dtype=np.float32),
                                                   np.array([154.843, 25.881, 67.544], dtype=np.float32),
                                                   np.array([155.689, 25.983, 68.820], dtype=np.float32),
                                                   np.array([157.418, 25.517, 68.551], dtype=np.float32),
                                                   np.array([158.062, 26.956, 67.686], dtype=np.float32)],
                                           'alpha': [np.array([153.488, 26.592, 67.584], dtype=np.float32)],
                                           'beta': [np.array([154.843, 25.881, 67.544], dtype=np.float32)]},
                                       2: {'all': [np.array([24.499, 13.739, 37.648], dtype=np.float32),
                                                   np.array([24.278, 13.068, 38.914], dtype=np.float32),
                                                   np.array([22.973, 13.448, 39.580], dtype=np.float32),
                                                   np.array([22.188, 12.566, 39.933], dtype=np.float32),
                                                   np.array([25.405, 13.382, 39.929], dtype=np.float32),
                                                   np.array([25.336, 12.578, 41.212], dtype=np.float32),
                                                   np.array([26.603, 13.084, 39.227], dtype=np.float32)],
                                           'alpha': [np.array([24.278, 13.068, 38.914], dtype=np.float32)],
                                           'beta': [np.array([25.405, 13.382, 39.929], dtype=np.float32)]},
                                       3: {'all': [np.array([24.805, 9.537, 22.454], dtype=np.float32),
                                                   np.array([24.052, 8.386, 22.974], dtype=np.float32),
                                                   np.array([24.897, 7.502, 23.849], dtype=np.float32),
                                                   np.array([24.504, 7.220, 24.972], dtype=np.float32),
                                                   np.array([23.506, 7.549, 21.793], dtype=np.float32),
                                                   np.array([22.741, 6.293, 22.182], dtype=np.float32),
                                                   np.array([22.242, 5.559, 20.931], dtype=np.float32),
                                                   np.array([23.319, 5.176, 20.037], dtype=np.float32),
                                                   np.array([23.931, 3.984, 20.083], dtype=np.float32),
                                                   np.array([23.622, 3.034, 20.961], dtype=np.float32),
                                                   np.array([24.895, 3.751, 19.199], dtype=np.float32)],
                                           'alpha': [np.array([24.052, 8.386, 22.974], dtype=np.float32)],
                                           'beta': [np.array([23.506, 7.549, 21.793], dtype=np.float32)]},
                                       4: {'all': [np.array([163.913, 21.571, 52.586], dtype=np.float32),
                                                   np.array([163.617, 20.553, 53.599], dtype=np.float32),
                                                   np.array([163.850, 21.067, 55.018], dtype=np.float32),
                                                   np.array([163.177, 20.639, 55.960], dtype=np.float32),
                                                   np.array([164.431, 19.277, 53.350], dtype=np.float32),
                                                   np.array([163.767, 18.286, 52.392], dtype=np.float32),
                                                   np.array([163.675, 18.807, 50.965], dtype=np.float32),
                                                   np.array([162.542, 19.056, 50.495], dtype=np.float32),
                                                   np.array([164.735, 18.978, 50.321], dtype=np.float32)],
                                           'alpha': [np.array([163.617, 20.553, 53.599], dtype=np.float32)],
                                           'beta': [np.array([164.431, 19.277, 53.350], dtype=np.float32)]},
                                       5: {'all': [np.array([153.913, 31.571, 52.586], dtype=np.float32),
                                                   np.array([153.617, 30.553, 53.599], dtype=np.float32),
                                                   np.array([153.850, 31.067, 55.018], dtype=np.float32),
                                                   np.array([153.177, 30.639, 55.960], dtype=np.float32),
                                                   np.array([154.431, 29.277, 53.350], dtype=np.float32),
                                                   np.array([153.767, 28.286, 52.392], dtype=np.float32),
                                                   np.array([153.675, 28.807, 50.965], dtype=np.float32),
                                                   np.array([152.542, 29.056, 50.495], dtype=np.float32),
                                                   np.array([154.735, 28.978, 50.321], dtype=np.float32)],
                                           'alpha': [np.array([153.617, 30.553, 53.599], dtype=np.float32)],
                                           'beta': [np.array([154.431, 29.277, 53.350], dtype=np.float32)]}}

    def evaluate_get_coords(self, query, pdb, chain, method_name, method, expected_coords, opt_param=None):
        cmd.load(pdb.file_name, query)
        cmd.select('best_chain', f'{query} and chain {chain}')
        for res_num in pdb.pdb_residue_list[chain]:
            residue = pdb.structure[0][chain][res_num]
            if opt_param is None:
                returned_coords = method(residue)
            else:
                returned_coords = method(residue, method=opt_param)
            self.assertEqual(len(returned_coords), len(expected_coords[res_num][method_name]))
            for i in range(len(returned_coords)):
                self.assertFalse((returned_coords[i] - expected_coords[res_num][method_name][i]).any())
        cmd.delete(query)

    def test_get_all_coords_chain_A(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='all',
                                 method=scorer._get_all_coords, expected_coords=self.expected_chain_a_coords)

    def test_get_alpha_coords_chain_A(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='alpha',
                                 method=scorer._get_c_alpha_coords, expected_coords=self.expected_chain_a_coords)

    def test_get_beta_coords_chain_A(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='beta',
                                 method=scorer._get_c_beta_coords, expected_coords=self.expected_chain_a_coords)

    def test_get_coords_all_chain_A(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='all',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_coords,
                                 opt_param='Any')

    def test_get_coords_alpha_chain_A(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='alpha',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_coords,
                                 opt_param='CA')

    def test_get_coords_beta_chain_A(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='beta',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_coords,
                                 opt_param='CB')

    def test_get_all_coords_chain_A_altLoc(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='all',
                                 method=scorer._get_all_coords, expected_coords=self.expected_chain_a_alt_coords)

    def test_get_alpha_coords_chain_A_altLoc(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='alpha',
                                 method=scorer._get_c_alpha_coords, expected_coords=self.expected_chain_a_alt_coords)

    def test_get_beta_coords_chain_A_altLoc(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='beta',
                                 method=scorer._get_c_beta_coords, expected_coords=self.expected_chain_a_alt_coords)

    def test_get_coords_all_chain_A_altLoc(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='all',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_alt_coords,
                                 opt_param='Any')

    def test_get_coords_alpha_chain_A_altLoc(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='alpha',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_alt_coords,
                                 opt_param='CA')

    def test_get_coords_beta_chain_A_altLoc(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='beta',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_alt_coords,
                                 opt_param='CB')

    def test_get_all_coords_chain_B(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='all',
                                 method=scorer._get_all_coords, expected_coords=self.expected_chain_b_coords)

    def test_get_alpha_coords_chain_B(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='alpha',
                                 method=scorer._get_c_alpha_coords, expected_coords=self.expected_chain_b_coords)

    def test_get_beta_coords_chain_B(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='beta',
                                 method=scorer._get_c_beta_coords, expected_coords=self.expected_chain_b_coords)

    def test_get_coords_all_chain_B(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='all',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_b_coords,
                                 opt_param='Any')

    def test_get_coords_alpha_chain_B(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='alpha',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_b_coords,
                                 opt_param='CA')

    def test_get_coords_beta_chain_B(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='beta',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_b_coords,
                                 opt_param='CB')


class TestScorerMeasureDistance(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        protein_aln1.write_out_alignment(aln_fn)

    def evaluate_measure_distance(self, scorer, method):
        scorer.fit()
        scorer.measure_distance(method=method)
        self.assertEqual(scorer.dist_type, method)
        residue_coords = {}
        size1 = len(scorer.query_pdb_mapper.pdb_ref.seq[scorer.query_pdb_mapper.best_chain])
        dists = np.zeros((size1, size1))
        dists2 = np.zeros((size1, size1))
        counter = 0
        counter_map = {}
        cmd.load(scorer.query_pdb_mapper.pdb_ref.file_name, scorer.query_pdb_mapper.query)
        cmd.select('best_chain', f'{scorer.query_pdb_mapper.query} and chain {scorer.query_pdb_mapper.best_chain}')
        for res_num in scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain]:
            counter_map[counter] = res_num
            residue = scorer.query_pdb_mapper.pdb_ref.structure[0][scorer.query_pdb_mapper.best_chain][res_num]
            coords = scorer._get_coords(residue, method=method)
            residue_coords[counter] = coords
            for residue2 in residue_coords:
                if residue2 == counter:
                    continue
                else:
                    if method == 'Any':
                        dist = et_calcDist(coords, residue_coords[residue2])
                    else:
                        dist = np.inf
                        for i in range(len(coords)):
                            for j in range(len(residue_coords[residue2])):
                                curr_dist = np.abs(np.power(coords[i][0] - residue_coords[residue2][j][0], 2) +
                                                   np.power(coords[i][1] - residue_coords[residue2][j][1], 2) +
                                                   np.power(coords[i][2] - residue_coords[residue2][j][2], 2))
                                if curr_dist < dist:
                                    dist = curr_dist
                    dist2 = np.sqrt(dist)
                    dists[counter, residue2] = dist
                    dists[residue2, counter] = dist
                    dists2[counter, residue2] = dist2
                    dists2[residue2, counter] = dist2
            counter += 1
        cmd.delete(scorer.query_pdb_mapper.query)
        distance_diff = np.square(scorer.distances) - dists
        self.assertLess(np.max(distance_diff), 1E-2)
        adj_diff = ((np.square(scorer.distances)[np.nonzero(distance_diff)] < CONTACT_DISTANCE2) ^
                    (dists[np.nonzero(distance_diff)] < CONTACT_DISTANCE2))
        self.assertEqual(np.sum(adj_diff), 0)
        self.assertEqual(len(np.nonzero(adj_diff)[0]), 0)
        distance_diff2 = scorer.distances - dists2
        self.assertLess(np.max(distance_diff2), 1E-4)

    def test_measure_distance_method_Any_chain_A(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='Any')

    def test_measure_distance_method_CA_chain_A(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='CA')

    def test_measure_distance_method_CB_chain_A(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='CB')

    def test_measure_distance_method_Any_chain_A_altLoc(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='Any')

    def test_measure_distance_method_CA_chain_A_altLoc(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='CA')

    def test_measure_distance_method_CB_chain_A_altLoc(self):
        scorer = Scorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='CB')

    def test_measure_distance_method_Any_chain_B(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        self.evaluate_measure_distance(scorer=scorer, method='Any')

    def test_measure_distance_method_CA_chain_B(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        self.evaluate_measure_distance(scorer=scorer, method='CA')

    def test_measure_distance_method_CB_chain_B(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        self.evaluate_measure_distance(scorer=scorer, method='CB')


class TestScorerPlotAUC(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_plot_auc(self, scorer, file_name, expected_file_name, output_dir):
        scorer.fit()
        scorer.measure_distance(method='CB')
        # These are fake values just to generate a plot in each case.
        tpr = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        fpr = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        auroc = 0.7
        scorer.plot_auc(auc_data=(tpr, fpr, auroc), title=None, file_name=file_name, output_dir=output_dir)
        self.assertTrue(os.path.isfile(expected_file_name))

    def test_no_fn_no_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_auroc.png'
        self.evaluate_plot_auc(scorer=scorer, file_name=None, expected_file_name=expected_file_name, output_dir=None)
        os.remove(expected_file_name)

    def test_fn_no_png_no_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_roc.png'
        self.evaluate_plot_auc(scorer=scorer, file_name='seq3_Cutoff20.0A_roc', expected_file_name=expected_file_name,
                               output_dir=None)
        os.remove(expected_file_name)

    def test_fn_no_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_auroc.png'
        self.evaluate_plot_auc(scorer=scorer, file_name=expected_file_name, expected_file_name=expected_file_name,
                               output_dir=None)
        os.remove(expected_file_name)

    def test_no_fn_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_auroc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auc(scorer=scorer, file_name=None, output_dir=new_dir,
                               expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)

    def test_fn_no_png_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_roc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auc(scorer=scorer, file_name='seq3_Cutoff20.0A_roc', output_dir=new_dir,
                               expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)

    def test_fn_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_auroc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auc(scorer=scorer, file_name=expected_file_name, output_dir=new_dir,
                               expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)


class TestScorerPlotAUPRC(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_plot_auprc(self, scorer, file_name, expected_file_name, output_dir):
        scorer.fit()
        scorer.measure_distance(method='CB')
        # scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        # scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        # scores += scores.T
        # ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        # scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        # precision, recall, auprc = scorer.score_precision_recall(category='Any')
        precision = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
        recall = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        auprc = 0.25
        scorer.plot_auprc(auprc_data=(precision, recall, auprc), title=None, file_name=file_name, output_dir=output_dir)
        self.assertTrue(os.path.isfile(expected_file_name))

    def test_no_fn_no_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_auprc.png'
        self.evaluate_plot_auprc(scorer=scorer, file_name=None, expected_file_name=expected_file_name, output_dir=None)
        os.remove(expected_file_name)

    def test_fn_no_png_no_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_auprc.png'
        self.evaluate_plot_auprc(scorer=scorer, file_name='seq3_Cutoff20.0A_auprc',
                                 expected_file_name=expected_file_name, output_dir=None)
        os.remove(expected_file_name)

    def test_fn_no_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_auprc.png'
        self.evaluate_plot_auprc(scorer=scorer, file_name=expected_file_name, expected_file_name=expected_file_name,
                                 output_dir=None)
        os.remove(expected_file_name)

    def test_no_fn_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_auprc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auprc(scorer=scorer, file_name=None, output_dir=new_dir,
                                 expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)

    def test_fn_no_png_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_auprc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auprc(scorer=scorer, file_name='seq3_Cutoff20.0A_auprc', output_dir=new_dir,
                                 expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)

    def test_fn_dir(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        expected_file_name = 'seq3_auprc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auprc(scorer=scorer, file_name=expected_file_name, output_dir=new_dir,
                                 expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)


class TestSCWZScoreSelection(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        protein_aln1.write_out_alignment(aln_fn)

    def evaluate_scw_z_score_selection(self, scw_scorer):
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        chain_length = scw_scorer.query_pdb_mapper.pdb_ref.size[scw_scorer.query_pdb_mapper.best_chain]
        for i in range(chain_length):
            for comb in combinations(list(range(scw_scorer.query_pdb_mapper.seq_aln.seq_length)), i):
                expected_scw_z_score = scw_scorer.clustering_z_score(res_list=list(comb))
                init_scw_z_score_selection(scw_scorer)
                scw_z_score = scw_z_score_selection(res_list=list(comb))
                mappable = all([x in scw_scorer.query_pdb_mapper.query_pdb_mapping for x in comb])
                if mappable:
                    self.assertEqual(scw_z_score[1], expected_scw_z_score[1])
                    self.assertEqual(scw_z_score[2], expected_scw_z_score[2])
                    self.assertEqual(scw_z_score[3], expected_scw_z_score[3])
                    self.assertEqual(scw_z_score[4], expected_scw_z_score[4])
                    self.assertEqual(scw_z_score[5], expected_scw_z_score[5])
                    self.assertEqual(scw_z_score[6], expected_scw_z_score[6])
                    self.assertEqual(scw_z_score[7], expected_scw_z_score[7])
                    self.assertEqual(scw_z_score[8], expected_scw_z_score[8])
                    self.assertEqual(scw_z_score[9], expected_scw_z_score[9])
                    self.assertEqual(scw_z_score[10], expected_scw_z_score[10])
                else:
                    self.assertIsNone(scw_z_score[1])
                    self.assertIsNone(scw_z_score[2])
                    self.assertIsNone(scw_z_score[3])
                    self.assertIsNone(scw_z_score[4])
                    self.assertIsNone(scw_z_score[5])
                    self.assertEqual(scw_z_score[6], '-')
                    self.assertIsNone(scw_z_score[7])
                    self.assertIsNone(scw_z_score[8])
                    self.assertIsNone(scw_z_score[9])
                    self.assertIsNone(scw_z_score[10])
                    self.assertEqual(scw_z_score[11], len(comb))

    def test_seq2_no_bias(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_scw_z_score_selection(scw_scorer=scw_scorer)

    def test_seq2_bias(self):
        scorer = Scorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_scw_z_score_selection(scw_scorer=scw_scorer)

    def test_seq3_no_bias(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_scw_z_score_selection(scw_scorer=scw_scorer)

    def test_seq3_bias(self):
        scorer = Scorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_scw_z_score_selection(scw_scorer=scw_scorer)


if __name__ == "__main__":
    unittest.main()