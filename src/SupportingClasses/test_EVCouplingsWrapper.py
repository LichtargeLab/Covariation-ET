"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
from time import time
from shutil import rmtree
from multiprocessing import cpu_count
from dotenv import find_dotenv, load_dotenv
from evcouplings.utils import read_config_file
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
from test_Base import TestBase
from EVCouplingsWrapper import EVCouplingsWrapper
from utils import compute_rank_and_coverage


class TestEVCouplingsWrapper(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestEVCouplingsWrapper, cls).setUpClass()
        cls.small_fa_fn = cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln']
        cls.large_fa_fn = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
        cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
        rmtree(cls.out_small_dir, ignore_errors=True)
        cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
        rmtree(cls.out_large_dir, ignore_errors=True)

    def evaluate_init(self, query, aln_file, out_dir, expected_length, expected_sequence):
        self.assertFalse(os.path.isdir(os.path.abspath(out_dir)))
        evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol='standard')
        self.assertEqual(evc.out_dir, os.path.abspath(out_dir))
        self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
        self.assertEqual(evc.query, query)
        self.assertIsNotNone(evc.original_aln)
        self.assertGreaterEqual(evc.original_aln.seq_length, expected_length)
        self.assertEqual(str(evc.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(evc.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(evc.original_aln_fn))
        self.assertIsNotNone(evc.non_gapped_aln)
        self.assertEqual(evc.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(evc.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(evc.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(evc.non_gapped_aln_fn))
        self.assertEqual(evc.method, 'EVCouplings')
        self.assertEqual(evc.protocol, 'standard')
        self.assertIsNone(evc.scores)
        self.assertIsNone(evc.probability)
        self.assertIsNone(evc.coverages)
        self.assertIsNone(evc.rankings)
        self.assertIsNone(evc.time)

    def test_1a_init(self):
        self.evaluate_init(query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))

    def test_1b_init(self):
        self.evaluate_init(query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
                           expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
                           expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))

    def evaluate_configure_run(self, query, aln_file, out_dir, expected_length):
        if os.path.isdir(out_dir):
            rmtree(out_dir)
        evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol='standard')
        expected_cpus = cpu_count()
        evc.configure_run(cores=expected_cpus)
        expected_config_fn = os.path.join(out_dir, "{}_config.txt".format(query))
        self.assertTrue(os.path.isfile(expected_config_fn))
        config = read_config_file(expected_config_fn)
        self.assertEqual(config["global"]["prefix"], out_dir + ('/' if not out_dir.endswith('/') else ''))
        self.assertEqual(config["global"]["sequence_id"], query)
        self.assertEqual(config["global"]["sequence_file"], os.path.join(out_dir, '{}_seq.fasta'.format(query)))
        self.assertIsNone(config["global"]["region"])
        self.assertEqual(config["global"]["theta"], 0.8)
        self.assertEqual(config["global"]["cpu"], expected_cpus)
        self.assertEqual(config["pipeline"], 'protein_monomer')
        self.assertIsNone(config["batch"])
        self.assertIsNone(config["management"]["database_uri"])
        self.assertIsNone(config["management"]["job_name"])
        self.assertIsNone(config["management"]["archive"])
        self.assertEqual(config["environment"]["engine"], 'local')
        self.assertIsNone(config["environment"]["queue"])
        self.assertEqual(config["environment"]["cores"], expected_cpus)
        self.assertIsNone(config["environment"]["memory"])
        self.assertIsNone(config["environment"]["time"])
        self.assertIsNone(config["environment"]["configuration"])
        self.assertEqual(config["tools"]["plmc"], os.environ.get('PLMC'))
        self.assertEqual(config["tools"]["jackhmmer"], os.environ.get('JACKHMMER'))
        self.assertEqual(config["tools"]["hmmbuild"], os.environ.get('HMMBUILD'))
        self.assertEqual(config["tools"]["hmmsearch"], os.environ.get('HMMSEARCH'))
        self.assertIsNone(config["tools"]["hhfilter"])
        self.assertIsNone(config["tools"]["psipred"])
        self.assertIsNone(config["tools"]["cns"])
        self.assertIsNone(config["tools"]["maxcluster"])
        self.assertEqual(config["databases"]["uniprot"], os.environ.get('UNIPROT'))
        self.assertEqual(config["databases"]["uniref100"], os.environ.get('UNIREF100'))
        self.assertEqual(config["databases"]["uniref90"], os.environ.get('UNIREF90'))
        self.assertIsNone(config["databases"]["sequence_download_url"])
        self.assertIsNone(config["databases"]["pdb_mmtf_dir"])
        self.assertEqual(config["databases"]["sifts_mapping_table"], os.environ.get('SIFTS_MAPPING_TABLE'))
        self.assertEqual(config["databases"]["sifts_sequence_db"], os.environ.get('SIFTS_SEQUENCE_DB'))
        self.assertEqual(config["stages"], ['align', 'couplings'])
        self.assertEqual(config["align"]["protocol"], 'existing')
        self.assertEqual(config["align"]["input_alignment"], evc.non_gapped_aln_fn)
        self.assertEqual(config["align"]["first_index"], 1)
        self.assertFalse(config["align"]["compute_num_effective_seqs"])
        self.assertIsNone(config["align"]["seqid_filter"])
        self.assertEqual(config["align"]["minimum_sequence_coverage"], 0)
        self.assertEqual(config["align"]["minimum_column_coverage"], 0)
        self.assertFalse(config["align"]["extract_annotation"])
        self.assertEqual(config["couplings"]["protocol"], 'standard')
        self.assertEqual(config["couplings"]["iterations"], 100)
        self.assertEqual(config["couplings"]["lambda_J"], 0.01)
        self.assertTrue(config["couplings"]["lambda_J_times_Lq"])
        self.assertEqual(config["couplings"]["lambda_h"], 0.01)
        self.assertIsNone(config["couplings"]["lambda_group"])
        self.assertIsNone(config["couplings"]["scale_clusters"])
        if isinstance(evc.non_gapped_aln.alphabet.letters, str):
            alphabet = evc.non_gapped_aln.alphabet.letters
        elif isinstance(evc.non_gapped_aln.alphabet.letters, list):
            alphabet = ''.join(evc.non_gapped_aln.alphabet.letters)
        else:
            raise TypeError('Alphabet cannot be properly processed when letters are type: {}'.format(
                evc.non_gapped_aln.alphabet.letters))
        expected_alphabet = '-' + alphabet.replace('-', '')
        self.assertEqual(config["couplings"]["alphabet"], expected_alphabet)
        self.assertFalse(config["couplings"]["ignore_gaps"])
        self.assertTrue(config["couplings"]["reuse_ecs"])
        self.assertEqual(config["couplings"]["min_sequence_distance"], 0)

    def test_2a_configure_run(self):
        self.evaluate_configure_run(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
            expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])

    def test_2b_configure_run(self):
        self.evaluate_configure_run(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
            expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])

    def evaluate_import_scores(self, query, aln_file, out_dir, expected_length):
        evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol='standard')
        scores = np.random.RandomState(1234567890).rand(expected_length, expected_length)
        scores[np.tril_indices(expected_length, 1)] = 0
        scores += scores.T
        _, coverages = compute_rank_and_coverage(expected_length, scores, 2, 'max')
        probabilities = 1 - coverages
        indices = np.triu_indices(expected_length, 1)
        sorted_scores, sorted_x, sorted_y, sorted_probability = zip(*sorted(zip(scores[indices], indices[0], indices[1],
                                                                                probabilities[indices])))
        expected_dir = os.path.join(out_dir, 'couplings')
        os.makedirs(expected_dir, exist_ok=True)
        expected_path = os.path.join(expected_dir, '_CouplingScores.csv')
        with open(expected_path, 'w') as handle:
            handle.write('i,A_i,j,A_j,fn,cn,segment_i,segment_j,probability\n')
            for i in range(len(sorted_scores)):
                handle.write('{},X,{},X,0,{},X,X,{}\n'.format(sorted_x[i] + 1, sorted_y[i] + 1, sorted_scores[i],
                                                              sorted_probability[i]))
        evc.import_covariance_scores(out_path=expected_path)

        diff_expected_scores = scores - scores.T
        not_passing_expected_scores = diff_expected_scores > 1E-15
        self.assertFalse(not_passing_expected_scores.any())
        diff_computed_scores = evc.scores - evc.scores.T
        not_passing_computed_scores = diff_computed_scores > 1E-15
        self.assertFalse(not_passing_computed_scores.any())

        diff_scores = evc.scores - scores
        not_passing_scores = diff_scores > 1E-15
        self.assertFalse(not_passing_scores.any())
        diff_probabilities = evc.probability - probabilities
        not_passing_protbabilities = diff_probabilities > 1E-15
        self.assertFalse(not_passing_protbabilities.any())

    def test_3a_import_scores(self):
        self.evaluate_import_scores(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
            expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])

    def test_3b_import_scores(self):
        self.evaluate_import_scores(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
            expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])

    def evaluate_calculator_scores(self, query, aln_file, out_dir, expected_length, expected_sequence):
        evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol='standard')
        start = time()
        evc.calculate_scores(delete_files=False, cores=self.max_threads)
        end = time()
        expected_time = end - start
        self.assertEqual(evc.out_dir, os.path.abspath(out_dir))
        self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
        self.assertEqual(evc.query, query)
        self.assertIsNotNone(evc.original_aln)
        self.assertGreaterEqual(evc.original_aln.seq_length, expected_length)
        self.assertEqual(str(evc.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(evc.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(evc.original_aln_fn))
        self.assertIsNotNone(evc.non_gapped_aln)
        self.assertEqual(evc.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(evc.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(evc.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(evc.non_gapped_aln_fn))
        self.assertEqual(evc.method, 'EVCouplings')
        self.assertEqual(evc.protocol, 'standard')
        self.assertIsNotNone(evc.scores)
        self.assertIsNotNone(evc.probability)
        expected_ranks, expected_coverages = compute_rank_and_coverage(expected_length, evc.scores, 2, 'max')
        ranks_diff = evc.rankings - expected_ranks
        ranks_not_passing = ranks_diff > 0.0
        self.assertFalse(ranks_not_passing.any())
        coverages_diff = evc.coverages - expected_coverages
        coverages_not_passing = coverages_diff > 0.0
        self.assertFalse(coverages_not_passing.any())
        self.assertLessEqual(evc.time, expected_time)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, 'EVCouplings.npz')))

    def test_4a_calculate_scores(self):
        self.evaluate_calculator_scores(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
            expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
            expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))

    def test_4b_calculate_scores(self):
        self.evaluate_calculator_scores(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
            expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
            expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))


if __name__ == '__main__':
    unittest.main()