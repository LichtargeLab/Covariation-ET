import os
import numpy as np
from unittest import TestCase
from SeqAlignment import SeqAlignment
from PDBReference import PDBReference
from ContactScorer import ContactScorer


class TestContactScorer(TestCase):

    def setUp(self):
        self.query1 = '1c17A'
        self.aln_file1 = '../Test/1c17A.fa'
        # self.aln_obj1 = SeqAlignment(file_name=self.aln_file1, query_id=self.query1)
        self.pdb_file1 = '../Test/query_1c17A.pdb'
        # self.pdb_obj1 = PDBReference(pdb_file=self.pdb_file1)
        self.scorer1 = ContactScorer(query= self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1,
                                     cutoff=8.0)
        self.query2 = '1h1vA'
        self.aln_file2 = '../Test/1h1vA.fa'
        # self.aln_obj2 = SeqAlignment(file_name=self.aln_file2, query_id=self.query2)
        self.pdb_file2 = '../Test/query_1h1vA.pdb'
        # self.pdb_obj2 = PDBReference(pdb_file=self.pdb_file2)
        self.scorer2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2,
                                     cutoff=8.0)

    def tearDown(self):
        del self.query1
        del self.aln_file1
        # del self.aln_obj1
        del self.pdb_file1
        # del self.pdb_obj1
        del self.scorer1
        del self.query2
        del self.aln_file2
        # del self.aln_obj2
        del self.pdb_file2
        # del self.pdb_obj2
        del self.scorer2

    def test__init__(self):
        with self.assertRaises(TypeError):
            ContactScorer()
        self.assertEqual(self.scorer1.query_alignment, os.path.abspath(self.aln_file1))
        self.assertEqual(self.scorer1.query_structure, os.path.abspath(self.pdb_file1))
        self.assertEqual(self.scorer1.cutoff, 8.0)
        self.assertIsNone(self.scorer1.best_chain)
        self.assertIsNone(self.scorer1.query_pdb_mapping)
        self.assertIsNone(self.scorer1._specific_mapping)
        self.assertIsNone(self.scorer1.distances)
        self.assertIsNone(self.scorer1.dist_type)
        self.assertEqual(self.scorer2.query_alignment, os.path.abspath(self.aln_file2))
        self.assertEqual(self.scorer2.query_structure, os.path.abspath(self.pdb_file2))
        self.assertEqual(self.scorer2.cutoff, 8.0)
        self.assertIsNone(self.scorer2.best_chain)
        self.assertIsNone(self.scorer2.query_pdb_mapping)
        self.assertIsNone(self.scorer2._specific_mapping)
        self.assertIsNone(self.scorer2.distances)
        self.assertIsNone(self.scorer2.dist_type)

    def test_fit(self):
        self.assertEqual(self.scorer1.query_alignment, os.path.abspath(self.aln_file1))
        self.assertEqual(self.scorer1.query_structure, os.path.abspath(self.pdb_file1))
        self.scorer1.fit()
        self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
        self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
        self.assertEqual(self.scorer1.best_chain, 'A')
        self.assertEqual(self.scorer1.query_pdb_mapping, {i: i for i in range(79)})
        self.scorer1.best_chain = None
        self.scorer1.fit()
        self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
        self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
        self.assertEqual(self.scorer1.best_chain, 'A')
        self.assertEqual(self.scorer1.query_pdb_mapping, {i: i for i in range(79)})
        self.scorer1.query_pdb_mapping = None
        self.scorer1.fit()
        self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
        self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
        self.assertEqual(self.scorer1.best_chain, 'A')
        self.assertEqual(self.scorer1.query_pdb_mapping, {i: i for i in range(79)})
        self.assertEqual(self.scorer2.query_alignment, os.path.abspath(self.aln_file2))
        self.assertEqual(self.scorer2.query_structure, os.path.abspath(self.pdb_file2))
        self.scorer2.fit()
        self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
        self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
        self.assertEqual(self.scorer2.best_chain, 'A')

        self.assertEqual(self.scorer2.query_pdb_mapping, {i: i for i in range(368)})
        self.scorer2.best_chain = None
        self.scorer2.fit()
        self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
        self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
        self.assertEqual(self.scorer2.best_chain, 'A')

        self.assertEqual(self.scorer2.query_pdb_mapping, {i: i for i in range(368)})
        self.scorer2.query_pdb_mapping = None
        self.scorer2.fit()
        self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
        self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
        self.assertEqual(self.scorer2.best_chain, 'A')

        self.assertEqual(self.scorer2.query_pdb_mapping, {i: i for i in range(368)})

    def test__str__(self):
        with self.assertRaises(ValueError):
            str(self.scorer1)
        self.scorer1.fit()
        self.assertEqual(str(self.scorer1),
                         'Query Sequence of Length: 79\nPDB with 1 Chains\nBest Sequence Match to Chain: A')
        with self.assertRaises(ValueError):
            str(self.scorer2)
        self.scorer2.fit()
        self.assertEqual(str(self.scorer2),
                         'Query Sequence of Length: 368\nPDB with 1 Chains\nBest Sequence Match to Chain: A')

    def test__get_all_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1 = np.vstack([[32.009, -3.867, -7.132], [31.313, -5.089, -7.616], [30.848, -5.958, -6.453],
                               [29.671, -5.946, -6.094], [32.271, -5.871, -8.520], [32.697, -5.107, -9.764],
                               [33.817, -6.055, -10.812], [35.183, -6.331, -9.687], [32.971, -4.141, -6.850],
                               [31.470, -3.497, -6.322], [32.029, -3.183, -7.914], [30.450, -4.786, -8.190],
                               [33.159, -6.118, -7.958], [31.788, -6.784, -8.833], [31.815, -4.858, -10.336],
                               [33.194, -4.198, -9.459], [36.108, -6.051, -10.169], [35.220, -7.375, -9.415],
                               [35.044, -5.732, -8.799]])
        measured1 = np.vstack(ContactScorer._get_all_coords(residue1))
        self.assertLess(np.sum(measured1 - expected1), 1e-5)
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        expected2 = np.vstack([[34.379, 19.087, 39.460], [33.929, 20.460, 39.036], [32.641, 20.827, 39.757],
                              [31.587, 20.266, 39.470], [33.683, 20.497, 37.525], [34.857, 20.070, 36.827],
                              [33.502, 21.908, 37.040]])
        measured2 = np.vstack(ContactScorer._get_all_coords(residue2))
        self.assertLess(np.sum(measured2 - expected2), 1e-5)

    def test__get_c_alpha_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1 = np.vstack([[31.313, -5.089, -7.616]])
        measured1 = np.vstack(ContactScorer._get_c_alpha_coords(residue1))
        self.assertLess(np.sum(measured1 - expected1), 1e-5)
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        expected2 = np.vstack([[33.929, 20.460, 39.036]])
        measured2 = np.vstack(ContactScorer._get_c_alpha_coords(residue2))
        self.assertLess(np.sum(measured2 - expected2), 1e-5)

    def test__get_c_beta_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1 = np.vstack([[32.271, -5.871, -8.520]])
        measured1 = np.vstack(ContactScorer._get_c_beta_coords(residue1))
        self.assertLess(np.sum(measured1 - expected1), 1e-5)
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        expected2 = np.vstack([[33.683, 20.497, 37.525]])
        measured2 = np.vstack(ContactScorer._get_c_beta_coords(residue2))
        self.assertLess(np.sum(measured2 - expected2), 1e-5)

    def test__get_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1a = np.vstack([[32.009, -3.867, -7.132], [31.313, -5.089, -7.616], [30.848, -5.958, -6.453],
                                [29.671, -5.946, -6.094], [32.271, -5.871, -8.520], [32.697, -5.107, -9.764],
                                [33.817, -6.055, -10.812], [35.183, -6.331, -9.687], [32.971, -4.141, -6.850],
                                [31.470, -3.497, -6.322], [32.029, -3.183, -7.914], [30.450, -4.786, -8.190],
                                [33.159, -6.118, -7.958], [31.788, -6.784, -8.833], [31.815, -4.858, -10.336],
                                [33.194, -4.198, -9.459], [36.108, -6.051, -10.169], [35.220, -7.375, -9.415],
                                [35.044, -5.732, -8.799]])
        measured1a = np.vstack(ContactScorer._get_coords(residue1, method='Any'))
        self.assertLess(np.sum(measured1a - expected1a), 1e-5)
        expected1b = np.vstack([[31.313, -5.089, -7.616]])
        measured1b = np.vstack(ContactScorer._get_coords(residue1, method='CA'))
        self.assertLess(np.sum(measured1b - expected1b), 1e-5)
        expected1c = np.vstack([[32.271, -5.871, -8.520]])
        measured1c = np.vstack(ContactScorer._get_coords(residue1, method='CB'))
        self.assertLess(np.sum(measured1c - expected1c), 1e-5)
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        expected2a = np.vstack([[34.379, 19.087, 39.460], [33.929, 20.460, 39.036], [32.641, 20.827, 39.757],
                                [31.587, 20.266, 39.470], [33.683, 20.497, 37.525], [34.857, 20.070, 36.827],
                                [33.502, 21.908, 37.040]])
        measured2a = np.vstack(ContactScorer._get_coords(residue2, method='Any'))
        self.assertLess(np.sum(measured2a - expected2a), 1e-5)
        expected2b = np.vstack([[33.929, 20.460, 39.036]])
        measured2b = np.vstack(ContactScorer._get_c_alpha_coords(residue2))
        self.assertLess(np.sum(measured2b - expected2b), 1e-5)
        expected2c = np.vstack([[33.683, 20.497, 37.525]])
        measured2c = np.vstack(ContactScorer._get_c_beta_coords(residue2))
        self.assertLess(np.sum(measured2c - expected2c), 1e-5)

    def test_measure_distance(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='Any')
        self.assertEqual(self.scorer1.dist_type, 'Any')
        residue1a = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        residue1b = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][2]
        pos1a = ContactScorer._get_all_coords(residue1a)
        pos1b = ContactScorer._get_all_coords(residue1b)
        expected1a = None
        for i in range(len(pos1a)):
            for j in range(len(pos1b)):
                curr_dist = np.sqrt(np.power(pos1a[i][0] - pos1b[j][0], 2) + np.power(pos1a[i][1] - pos1b[j][1], 2) +
                                    np.power(pos1a[i][2] - pos1b[j][2], 2))
                if (expected1a is None) or (curr_dist < expected1a):
                    expected1a = curr_dist
        self.assertLess(expected1a - self.scorer1.distances[0, 1], 1e-5)
        self.scorer1.measure_distance(method='CA')
        self.assertEqual(self.scorer1.dist_type, 'CA')
        expected1b = np.sqrt(np.power(31.313 - 31.462, 2) + np.power(-5.089 - -7.593, 2) + np.power(-7.616 - -4.746, 2))
        self.assertLess(expected1b - self.scorer1.distances[0, 1], 1e-5)
        self.scorer1.measure_distance(method='CB')
        self.assertEqual(self.scorer1.dist_type, 'CB')
        expected1c = np.sqrt(np.power(32.271 - 32.746, 2) + np.power(-5.871 - -8.085, 2) + np.power(-8.520 - -4.071, 2))
        self.assertLess(expected1c - self.scorer1.distances[0, 1], 1e-5)

        self.scorer2.fit()
        self.scorer2.measure_distance(method='Any')
        self.assertEqual(self.scorer2.dist_type, 'Any')
        residue2a = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        residue2b = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][6]
        pos2a = ContactScorer._get_all_coords(residue2a)
        pos2b = ContactScorer._get_all_coords(residue2b)
        expected2a = None
        for i in range(len(pos2a)):
            for j in range(len(pos2b)):
                curr_dist = np.sqrt(np.power(pos2a[i][0] - pos2b[j][0], 2) + np.power(pos2a[i][1] - pos2b[j][1], 2) +
                                    np.power(pos2a[i][2] - pos2b[j][2], 2))
                if (expected2a is None) or (curr_dist < expected2a):
                    expected2a = curr_dist
        print(expected2a)
        print(self.scorer2.distances[0, 1])
        self.assertLess(expected2a - self.scorer2.distances[0, 1], 1e-5)
        self.scorer2.measure_distance(method='CA')
        self.assertEqual(self.scorer2.dist_type, 'CA')
        expected2b = np.sqrt(np.power(33.929 - 31.582, 2) + np.power(20.460 - 22.092, 2) + np.power(39.036 - 41.510, 2))
        self.assertLess(expected2b - self.scorer2.distances[0, 1], 1e-5)
        self.scorer2.measure_distance(method='CB')
        self.assertEqual(self.scorer2.dist_type, 'CB')
        expected2c = np.sqrt(np.power(33.683 - 32.070, 2) + np.power(20.497 - 22.322, 2) + np.power(37.525 - 42.924, 2))
        self.assertLess(expected2c - self.scorer2.distances[0, 1], 1e-5)

    def test_find_pairs_by_separation(self):
        self.scorer1.fit()
        with self.assertRaises(ValueError):
            self.scorer1.find_pairs_by_separation(category='Wide')
        expected1 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
        for i in range(79):
            for j in range(i + 1, 79):
                pair = (i, j)
                separation = j - i
                if (separation >= 1) and (separation < 6):
                    expected1['Neighbors'].append(pair)
                if (separation >= 6) and (separation < 13):
                    expected1['Short'].append(pair)
                if (separation >= 13) and (separation < 24):
                    expected1['Medium'].append(pair)
                if separation >= 24:
                    expected1['Long'].append(pair)
                expected1['Any'].append(pair)
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Any'), expected1['Any'])
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Neighbors'), expected1['Neighbors'])
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Short'), expected1['Short'])
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Medium'), expected1['Medium'])
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Long'), expected1['Long'])
        self.scorer2.fit()
        with self.assertRaises(ValueError):
            self.scorer2.find_pairs_by_separation(category='Small')
        expected2 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
        for i in range(368):
            for j in range(i + 1, 368):
                pair = (i, j)
                separation = j - i
                if (separation >= 1) and (separation < 6):
                    expected2['Neighbors'].append(pair)
                if (separation >= 6) and (separation < 13):
                    expected2['Short'].append(pair)
                if (separation >= 13) and (separation < 24):
                    expected2['Medium'].append(pair)
                if separation >= 24:
                    expected2['Long'].append(pair)
                expected2['Any'].append(pair)
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Any'), expected2['Any'])
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Neighbors'), expected2['Neighbors'])
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Short'), expected2['Short'])
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Medium'), expected2['Medium'])
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Long'), expected2['Long'])


    def test__map_predictions_to_pdb(self):
        self.fail()

    def test_score_auc(self):
        self.fail()

    def test_plot_auc(self):
        self.fail()

    def test_score_precision(self):
        self.fail()

    def test_score_clustering_of_contact_predictions(self):
        self.fail()

    def test__clustering_z_score(self):
        self.fail()

    def test_write_out_clustering_results(self):
        self.fail()

    def test_evaluate_predictor(self):
        self.fail()

    def test_evaluate_predictions(self):
        self.fail()
