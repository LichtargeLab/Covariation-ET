"""

"""
import os
import sys
import math
import unittest
import numpy as np
from unittest import TestCase
from itertools import combinations
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
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.PDBReference import PDBReference
from Evaluation.SequencePDBMap import SequencePDBMap
from Evaluation.ContactScorer import ContactScorer
from Evaluation.SelectionClusterWeighting import (SelectionClusterWeighting, init_compute_w_and_w2_ave_sub,
                                                  compute_w_and_w2_ave_sub)
from Testing.test_Base import (protein_seq1, protein_seq2, protein_seq3, write_out_temp_fn, protein_aln)
from Testing.test_contactScorer import (et_calcDist, CONTACT_DISTANCE2, pro_str, pro_pdb1, pro_pdb_1_alt_locs,
                                        pro_pdb1_scramble, pro_pdb2, pro_pdb2_scramble, pro_pdb_full,
                                        pro_pdb_full_scramble, aln_fn, protein_aln1, protein_aln2, protein_aln3)


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


def et_calc_w2_sub_problems(A, bias=1):
    """Calculate w2_ave components for calculation z-score (z_S) for residue selection reslist=[1,2,...]
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
    part1 = 0.0
    part2 = 0.0
    part3 = 0.0
    if bias == 1:
        for resi, neighborsj in A.items():
            for resj in neighborsj:
                for resk, neighborsl in A.items():
                    for resl in neighborsl:
                        if (resi == resk and resj == resl) or \
                                (resi == resl and resj == resk):
                            part1 += (resj - resi) * (resl - resk)
                        elif (resi == resk) or (resj == resl) or \
                                (resi == resl) or (resj == resk):
                            part2 += (resj - resi) * (resl - resk)
                        else:
                            part3 += (resj - resi) * (resl - resk)
    elif bias == 0:
        for resi, neighborsj in A.items():
            for resj in neighborsj:
                for resk, neighborsl in A.items():
                    for resl in neighborsl:
                        if (resi == resk and resj == resl) or \
                                (resi == resl and resj == resk):
                            part1 += 1
                        elif (resi == resk) or (resj == resl) or \
                                (resi == resl) or (resj == resk):
                            part2 += 1
                        else:
                            part3 += 1
    return part1, part2, part3


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


class TestSCWInit(TestCase):

    def test_init_biased(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        aln_obj2 = aln_obj.remove_gaps()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        scw_obj = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                            biased=True)
        self.assertEqual(scw_obj.query_pdb_mapper.seq_aln.seq_length, aln_obj2.seq_length)
        self.assertEqual(scw_obj.query_pdb_mapper.pdb_ref, pdb_obj)
        self.assertEqual(scw_obj.query_pdb_mapper.best_chain, 'A')
        self.assertTrue(scw_obj.query_pdb_mapper.is_aligned())
        self.assertEqual(np.sum(np.abs(scw_obj.distances - scorer.distances)), 0)
        self.assertTrue(scw_obj.biased)
        self.assertIsNone(scw_obj.w_and_w2_ave_sub)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_unbiased(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        aln_obj2 = aln_obj.remove_gaps()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        scw_obj = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                            biased=False)
        self.assertEqual(scw_obj.query_pdb_mapper.seq_aln.seq_length, aln_obj2.seq_length)
        self.assertEqual(scw_obj.query_pdb_mapper.pdb_ref, pdb_obj)
        self.assertEqual(scw_obj.query_pdb_mapper.best_chain, 'A')
        self.assertTrue(scw_obj.query_pdb_mapper.is_aligned())
        self.assertEqual(np.sum(np.abs(scw_obj.distances - scorer.distances)), 0)
        self.assertFalse(scw_obj.biased)
        self.assertIsNone(scw_obj.w_and_w2_ave_sub)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_no_sequence_pdb_map(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=None, pdb_dists=scorer.distances, biased=True)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_bad_sequence_pdb_map(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        seq_pdb_map = SequencePDBMap(query='seq1', query_alignment=aln_obj, query_structure=pdb_obj, chain='A')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=seq_pdb_map, pdb_dists=scorer.distances, biased=True)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_fail_no_pdb_dist(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=None, biased=True)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_fail_no_bias(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances, biased=None)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_fail_bad_bias(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances, biased='biased')
        os.remove(aln_fn)
        os.remove(pdb_fn)


class TestComputeWAndW2Ave(TestCase):

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

    def evaluate_compute_w_and_w2_ave(self, distances, pdb_residues, biased):
        for i in range(len(pdb_residues)):
            curr_w_ave_pre = 0
            curr_case1 = 0
            curr_case2 = 0
            curr_case3 = 0
            for j in range(i + 1, len(pdb_residues)):
                distance = distances[i, j]
                if distance >= 4.0:
                    continue
                if biased:
                    bias_term = abs(pdb_residues[i] - pdb_residues[j])
                else:
                    bias_term = 1
                curr_w_ave_pre += bias_term
                for k in range(len(pdb_residues)):
                    for l in range(k + 1, len(pdb_residues)):
                        if biased:
                            bias_term2 = abs(pdb_residues[k] - pdb_residues[l])
                        else:
                            bias_term2 = 1
                        final_bias = bias_term * bias_term2
                        pair_overlap = len({i, j}.intersection({k, l}))
                        if pair_overlap == 2:
                            curr_case1 += final_bias
                        elif pair_overlap == 1:
                            curr_case2 += final_bias
                        elif pair_overlap == 0:
                            curr_case3 += final_bias
                        else:
                            raise ValueError('Impossible pair')
            init_compute_w_and_w2_ave_sub(dists=distances, structure_res_num=pdb_residues, bias_bool=biased)
            cases = compute_w_and_w2_ave_sub(res_i=i)
            self.assertEqual(cases['w_ave_pre'], curr_w_ave_pre)
            self.assertEqual(cases['Case1'], curr_case1)
            self.assertEqual(cases['Case2'], curr_case2)
            self.assertEqual(cases['Case3'], curr_case3)

    def test_seq2_no_bias(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_compute_w_and_w2_ave(distances=scorer.distances, biased=False,
                                           pdb_residues=scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain])

    def test_seq2_bias(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_compute_w_and_w2_ave(distances=scorer.distances, biased=True,
                                           pdb_residues=scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain])

    def test_seq3_no_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_compute_w_and_w2_ave(distances=scorer.distances, biased=False,
                                           pdb_residues=scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain])

    def test_seq3_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_compute_w_and_w2_ave(distances=scorer.distances, biased=True,
                                           pdb_residues=scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain])


class TestComputeBackgroundWAndW2Ave(TestCase):

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

    def evaluate_compute_background_w_and_w2_ave(self, scw_scorer, processes):
        scw_scorer.compute_background_w_and_w2_ave(processes=processes)
        best_chain = None
        for model in scw_scorer.query_pdb_mapper.pdb_ref.structure:
            for chain in model:
                if chain.id == scw_scorer.query_pdb_mapper.best_chain:
                    best_chain = chain
                    break
        if best_chain is None:
            raise ValueError('Best Chain Never Initialized')
        adj, res_atoms = et_computeAdjacency(chain=best_chain,
                                             mapping={res: i for i, res in enumerate(scw_scorer.query_pdb_mapper.pdb_ref.pdb_residue_list[scw_scorer.query_pdb_mapper.best_chain])})
        case1, case2, case3 = et_calc_w2_sub_problems(adj, bias=1 if scw_scorer.biased else 0)
        self.assertEqual(scw_scorer.w_and_w2_ave_sub['Case1'], case1)
        self.assertEqual(scw_scorer.w_and_w2_ave_sub['Case2'], case2)
        self.assertEqual(scw_scorer.w_and_w2_ave_sub['Case3'], case3)

    def test_seq2_no_bias_single_process(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=1)

    def test_seq2_no_bias_multi_process(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=2)

    def test_seq2_bias_single_process(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=1)

    def test_seq2_bias_multi_process(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=2)

    def test_seq3_no_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=2)

    def test_seq3_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=2)


class TestClusteringZScore(TestCase):

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

    def evaluate_clustering_z_score(self, scw_scorer):
        best_chain = None
        for model in scw_scorer.query_pdb_mapper.pdb_ref.structure:
            for chain in model:
                if chain.id == scw_scorer.query_pdb_mapper.best_chain:
                    best_chain = chain
                    break
        if best_chain is None:
            raise ValueError('Best Chain Never Initialized')
        adj, res_atoms = et_computeAdjacency(chain=best_chain,
                                             mapping={res: i for i, res in enumerate(
                                                 scw_scorer.query_pdb_mapper.pdb_ref.pdb_residue_list[
                                                     scw_scorer.query_pdb_mapper.best_chain])})
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        chain_length = scw_scorer.query_pdb_mapper.pdb_ref.size[scw_scorer.query_pdb_mapper.best_chain]
        for i in range(chain_length):
            for comb in combinations(list(range(scw_scorer.query_pdb_mapper.seq_aln.seq_length)), i):
                scw_z_score = scw_scorer.clustering_z_score(res_list=list(comb))
                print(scw_z_score)
                # a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, m
                res_list = [scw_scorer.query_pdb_mapper.pdb_ref.pdb_residue_list[scw_scorer.query_pdb_mapper.best_chain][x] for x in comb]
                expected_scw_z_score = et_calcZScore(reslist=res_list, L=chain_length, A=adj,
                                                     bias=1 if scw_scorer.biased else 0)
                # M, L, pi1, pi2, pi3, (w - w_ave) / sigma, w, w_ave, w2_ave, sigma, cases
                mappable = all([x in scw_scorer.query_pdb_mapper.query_pdb_mapping for x in comb])
                if mappable:
                    self.assertEqual(scw_z_score[1], expected_scw_z_score[0])
                    self.assertEqual(scw_z_score[2], expected_scw_z_score[1])
                    self.assertEqual(scw_z_score[3], expected_scw_z_score[2])
                    self.assertEqual(scw_z_score[4], expected_scw_z_score[3])
                    self.assertEqual(scw_z_score[5], expected_scw_z_score[4])
                    self.assertEqual(scw_z_score[6], expected_scw_z_score[5])
                    self.assertEqual(scw_z_score[7], expected_scw_z_score[6])
                    self.assertEqual(scw_z_score[8], expected_scw_z_score[7])
                    self.assertEqual(scw_z_score[9], expected_scw_z_score[8])
                    self.assertEqual(scw_z_score[10], expected_scw_z_score[9])
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
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_clustering_z_score(scw_scorer=scw_scorer)

    def test_seq2_bias(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_clustering_z_score(scw_scorer=scw_scorer)

    def test_seq3_no_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_clustering_z_score(scw_scorer=scw_scorer)

    def test_seq3_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_clustering_z_score(scw_scorer=scw_scorer)


if __name__ == '__main__':
    unittest.main()