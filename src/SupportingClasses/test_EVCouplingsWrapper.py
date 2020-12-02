"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
import unittest
from unittest import TestCase
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
from EVCouplingsWrapper import EVCouplingsWrapper
from utils import compute_rank_and_coverage
from test_Base import protein_seq1, protein_seq2, protein_seq3, dna_seq1, dna_seq2, dna_seq3, write_out_temp_fn
from test_Base import processes as max_processes


pro_str = f'>{protein_seq1.id}\n{protein_seq1.seq}\n>{protein_seq2.id}\n{protein_seq2.seq}\n>{protein_seq3.id}\n{protein_seq3.seq}'
dna_str = f'>{dna_seq1.id}\n{dna_seq1.seq}\n>{dna_seq2.id}\n{dna_seq2.seq}\n>{dna_seq3.id}\n{dna_seq3.seq}'
test_dir = os.path.join(os.getcwd(), 'TestCase')


class TestEVCouplingsWrapperInit(TestCase):

    def evaluate_init(self, query, aln_file, out_dir, protocol, expected_length, expected_sequence, polymer_type='Protein'):
        self.assertFalse(os.path.isdir(os.path.abspath(out_dir)))
        evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol=protocol,
                                 polymer_type=polymer_type)
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
        self.assertEqual(evc.protocol, protocol)
        self.assertIsNone(evc.scores)
        self.assertIsNone(evc.probability)
        self.assertIsNone(evc.coverages)
        self.assertIsNone(evc.rankings)
        self.assertIsNone(evc.time)

    def test_evcouplingswrapper_init_standard_protein_aln_1(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, protocol='standard',
                           expected_length=3, expected_sequence='MET')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_standard_protein_aln_2(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq2', aln_file=protein_aln_fn, out_dir=test_dir, protocol='standard',
                           expected_length=5, expected_sequence='MTREE')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_standard_protein_aln_3(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq3', aln_file=protein_aln_fn, out_dir=test_dir, protocol='standard',
                           expected_length=5, expected_sequence='MFREE')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_mean_field_protein_aln_1(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, protocol='mean_field',
                           expected_length=3, expected_sequence='MET')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_mean_field_protein_aln_2(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq2', aln_file=protein_aln_fn, out_dir=test_dir, protocol='mean_field',
                           expected_length=5, expected_sequence='MTREE')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_mean_field_protein_aln_3(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq3', aln_file=protein_aln_fn, out_dir=test_dir, protocol='mean_field',
                           expected_length=5, expected_sequence='MFREE')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_standard_dna_aln_1(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, protocol='standard', expected_length=9,
                           expected_sequence='ATGGAGACT', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_standard_dna_aln_2(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq2', aln_file=dna_aln_fn, out_dir=test_dir, protocol='standard', expected_length=15,
                           expected_sequence='ATGACTAGAGAGGAG', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_standard_dna_aln_3(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq3', aln_file=dna_aln_fn, out_dir=test_dir, protocol='standard', expected_length=15,
                           expected_sequence='ATGTTTAGAGAGGAG', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_mean_field_dna_aln_1(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, protocol='mean_field',
                           expected_length=9, expected_sequence='ATGGAGACT', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_mean_field_dna_aln_2(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq2', aln_file=dna_aln_fn, out_dir=test_dir, protocol='mean_field',
                           expected_length=15, expected_sequence='ATGACTAGAGAGGAG', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_init_mean_field_dna_aln_3(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq3', aln_file=dna_aln_fn, out_dir=test_dir, protocol='mean_field',
                           expected_length=15, expected_sequence='ATGTTTAGAGAGGAG', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)


class TestEVCouplingsWrapperConfigureRun(TestCase):

    def evaluate_configure_run(self, query, aln_file, out_dir, protocol, polymer_type='Protein'):
        if os.path.isdir(out_dir):
            rmtree(out_dir)
        evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol=protocol,
                                 polymer_type=polymer_type)
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
        self.assertEqual(config["couplings"]["protocol"], protocol)
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

    def test_evcouplingswrapper_configure_run_standard_protein_aln(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_configure_run(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, protocol='standard')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_configure_run_mean_field_protein_aln(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_configure_run(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, protocol='mean_field')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_configure_run_standard_dna_aln(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_configure_run(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, protocol='standard',
                                    polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_configure_run_mean_field_dna_aln(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_configure_run(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, protocol='mean_field',
                                    polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)


class TestEVCouplingsWrapperImportCovarianceScores(TestCase):

    def evaluate_import_scores(self, query, aln_file, out_dir, expected_length, protocol, polymer_type='Protein'):
        evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol=protocol,
                                 polymer_type=polymer_type)
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

    def test_evcouplingswrapper_import_scores_standard_protein_aln(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_import_scores(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, protocol='standard',
                                    expected_length=3)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_import_scores_mean_field_protein_aln(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_import_scores(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, protocol='mean_field',
                                    expected_length=3)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_import_scores_standard_dna_aln(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_import_scores(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, protocol='standard',
                                    expected_length=9, polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_import_scores_mean_field_dna_aln(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_import_scores(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, protocol='mean_field',
                                    expected_length=9, polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)


class TestEVCouplingsWrapperCalculateScores(TestCase):

    def evaluate_calculate_scores(self, query, aln_file, out_dir, expected_length, protocol, polymer_type='Protein'):
        evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol=protocol,
                                 polymer_type=polymer_type)
        start = time()
        evc.calculate_scores(delete_files=False, cores=max_processes)
        end = time()
        expected_time = end - start
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

    def test_evcouplingswrapper_calculate_scores_standard_protein_aln_1(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculate_scores(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, protocol='standard',
                                       expected_length=3)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_standard_protein_aln_2(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculate_scores(query='seq2', aln_file=protein_aln_fn, out_dir=test_dir, protocol='standard',
                                       expected_length=5)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_standard_protein_aln_3(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculate_scores(query='seq3', aln_file=protein_aln_fn, out_dir=test_dir, protocol='standard',
                                       expected_length=5)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_mean_field_protein_aln_1(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculate_scores(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, protocol='mean_field',
                                       expected_length=3)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_mean_field_protein_aln_2(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculate_scores(query='seq2', aln_file=protein_aln_fn, out_dir=test_dir, protocol='mean_field',
                                       expected_length=5)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_mean_field_protein_aln_3(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculate_scores(query='seq3', aln_file=protein_aln_fn, out_dir=test_dir, protocol='mean_field',
                                       expected_length=5)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_standard_dna_aln_1(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculate_scores(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, protocol='standard',
                                       expected_length=9, polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_standard_dna_aln_2(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculate_scores(query='seq2', aln_file=dna_aln_fn, out_dir=test_dir, protocol='standard',
                                       expected_length=15, polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_standard_dna_aln_3(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculate_scores(query='seq3', aln_file=dna_aln_fn, out_dir=test_dir, protocol='standard',
                                       expected_length=15, polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_mean_field_dna_aln_1(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculate_scores(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, protocol='mean_field',
                                       expected_length=9, polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_mean_field_dna_aln_2(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculate_scores(query='seq2', aln_file=dna_aln_fn, out_dir=test_dir, protocol='mean_field',
                                       expected_length=15, polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_evcouplingswrapper_calculate_scores_mean_field_dna_aln_3(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculate_scores(query='seq3', aln_file=dna_aln_fn, out_dir=test_dir, protocol='mean_field',
                                       expected_length=15, polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()