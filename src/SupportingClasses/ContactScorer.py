"""
Created on Sep 1, 2018

@author: dmkonecki
"""
import os
import math
import numpy as np
import pandas as pd
from pymol import cmd
from time import time
from math import floor
from multiprocessing import Pool
from Bio import pairwise2
from Bio.PDB.Polypeptide import one_to_three
from Bio.SubsMat import MatrixInfo as matlist
from sklearn.metrics import (auc, roc_curve, precision_recall_curve, precision_score, recall_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from seaborn import heatmap, scatterplot
from SeqAlignment import SeqAlignment
try:
    from SupportingClasses.SeqAlignment import SeqAlignment as UpperSeqAlignment
    seq_aln_classes = (SeqAlignment, UpperSeqAlignment)
except ImportError:
    seq_aln_classes = (SeqAlignment, )
from PDBReference import PDBReference


class ContactScorer(object):
    """
    ContactScorer

    This class is meant to abstract the process of scoring a set of contact predictions away from the actual method
    (previously it was included in the prediction class).  This is being done for two main reasons. First, contact
    predictions are being made with several different methods and so their scoring should be performed consistently by
    another object or function. Second, this class will support several scoring methods such as the Precision at L/K
    found commonly in the literature as well as methods internal to the lab like the clustering Z-score derived from
    prior ET work.

    Attributes:
        query (str): The name of the query sequence/structure.
        query_alignment (str/SeqAlignment): Path to the alignment being evaluated in this contact scoring prediction
        task. This is eventually updated to a SeqAlignment object, after the alignment has been imported.
        query_structure (str/PDBReference): Path to the pdb file to use for evaluating the contact predictions made.
        This is eventually updated to a PDBReference once the pdb file has been imported.
        cutoff (float): Value to use as a distance cutoff for contact prediction.
        best_chain (str): The chain in the provided pdb which most closely matches the query sequence as specified by
        in the "chain" option of the __init__ function or determined by pairwise global alignment.
        query_pdb_mapping (dict): A mapping from the index of the query sequence to the index of the pdb chain's
        sequence for those positions which match according to a pairwise global alignment.
        _specific_mapping (dict): A mapping for the indices of the distance matrix used to determine residue contacts.
        distances (np.array): The distances between residues, used for determining those which are in contact in order
        to assess predictions.
        dist_type (str): The type of distance computed for assessement (expecting 'Any', 'CB' - Beta Carbon, 'CA' -
        alpha carbon.
        data (panda.DataFrame): A data frame containing a mapping of the sequence, structure, and predictions as well as
        labels to indicate which sequence separation category and top prediction category specific scores belong to.
        This is used for most evaluation methods.
    """

    def __init__(self, query, seq_alignment, pdb_reference, cutoff, chain=None):
        """
        __init__

        This function initializes a new ContactScorer, it accepts paths to an alignment and if available a pdb file to
        be used in the assessment of contact predictions for a given query. It also requires a cutoff which is used to
        denote which residues are actually in contact, based on their distance to one another.

        Args:
            query (str): The name of the query sequence/structure.
            seq_alignment (str/path, SeqAlignment): Path to the alignment being evaluated in this contact scoring
            prediction task or an already initialized SeqAlignment object.
            pdb_reference (str/path, PDBReference): The path to a PDB structure file, or an already initialized
            PDBReference object.
            cutoff (int or float): The distance between two residues at or below which a true contact is said to be
            occurring.
            chain (str): Which chain in the PDB structure to use for comparison and evaluation. If left blank the best
            chain will be identified by aligning the query sequence from seq_alignment against the chains in
            pdb_reference and the closest match will be selected.
        """
        self.query = query
        if isinstance(seq_alignment, os.PathLike):
            self.query_alignment = os.path.abspath(seq_alignment)
        else:
            self.query_alignment = seq_alignment
        if isinstance(pdb_reference, os.PathLike):
            self.query_structure = os.path.abspath(pdb_reference)
        else:
            self.query_structure = pdb_reference
        self.cutoff = cutoff
        self.best_chain = chain
        self.query_pdb_mapping = None
        self._specific_mapping = None
        self.distances = None
        self.dist_type = None
        self.data = None

    def __str__(self):
        """
        __str__

        Method over writing the default __str__ method, giving a simple summary of the data held by the ContactScorer.

        Returns:
            str: Simple string summarizing the contents of the ContactScorer.

        Usage Example:
        >>> scorer = ContactScorer(p53_sequence, p53_structure, query='P53', cutoff=8.0)
        >>> scorer.fit()
        >>> str(scorer)
        """
        if (self.best_chain is None) or (self.query_pdb_mapping is None):
            raise ValueError('Scorer not yet fitted.')
        return 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            self.query_alignment.seq_length, len(self.query_structure.chains), self.best_chain)

    def fit(self):
        """
        Fit

        This function maps sequence positions between the query sequence from the alignment and residues in PDB file.
        If there are multiple chains for the structure in the PDB, the one which matches the query sequence best
        (highest global alignment score) is used and recorded in the best_chain variable. This method updates the
        query_alignment, query_structure, best_chain, and query_pdb_mapping class attributes.
        """
        gap_open = -10
        gap_extend = -0.5
        if (self.best_chain is None) or (self.query_pdb_mapping is None):
            start = time()
            if self.query_alignment is None:
                raise ValueError('Scorer cannot be fit, because no alignment was provided.')
            else:
                if not isinstance(self.query_alignment, seq_aln_classes):
                    self.query_alignment = SeqAlignment(file_name=self.query_alignment, query_id=self.query)
                self.query_alignment.import_alignment()
                self.query_alignment = self.query_alignment.remove_gaps()
            if self.query_structure is None:
                raise ValueError('Scorer cannot be fit, because no PDB was provided.')
            else:
                if not isinstance(self.query_structure, PDBReference):
                    self.query_structure = PDBReference(pdb_file=self.query_structure)
                self.query_structure.import_pdb(structure_id=self.query)
            if self.best_chain is None:
                best_chain = None
                best_alignment = None
                for ch in self.query_structure.seq:
                    curr_align = pairwise2.align.globalds(self.query_alignment.query_sequence,
                                                          self.query_structure.seq[ch], matlist.blosum62, gap_open,
                                                          gap_extend)
                    if (best_alignment is None) or (best_alignment[0][2] < curr_align[0][2]):
                        best_alignment = curr_align
                        best_chain = ch
                self.best_chain = best_chain
            else:
                best_alignment = pairwise2.align.globalds(self.query_alignment.query_sequence,
                                                          self.query_structure.seq[self.best_chain], matlist.blosum62,
                                                          gap_open, gap_extend)
            print(pairwise2.format_alignment(*best_alignment[0]))
            if self.query_pdb_mapping is None:
                f_counter = 0
                p_counter = 0
                f_to_p_map = {}
                for i in range(len(best_alignment[0][0])):
                    if (best_alignment[0][0][i] != '-') and (best_alignment[0][1][i] != '-'):
                        f_to_p_map[f_counter] = p_counter
                    if best_alignment[0][0][i] != '-':
                        f_counter += 1
                    if best_alignment[0][1][i] != '-':
                        p_counter += 1
                self.query_pdb_mapping = f_to_p_map
            end = time()
            print('Mapping query sequence and pdb took {} min'.format((end - start) / 60.0))
        if (self.data is None) or (not self.data.columns.isin(
                ['Seq Pos 1', 'Seq AA 1', 'Seq Pos 2', 'Seq AA 2', 'Seq Separation', 'Seq Separation Category',
                 'Struct Pos 1', 'Struct AA 1', 'Struct Pos 2', 'Struct AA 2']).all()):
            start = time()
            data_dict = {'Seq Pos 1': [], 'Seq AA 1': [], 'Seq Pos 2': [], 'Seq AA 2': [], 'Struct Pos 1': [],
                         'Struct AA 1': [], 'Struct Pos 2': [], 'Struct AA 2': []}
            for pos1 in range(self.query_alignment.seq_length):
                char1 = self.query_alignment.query_sequence[pos1]
                pos_other = list(range(pos1 + 1, self.query_alignment.seq_length))
                char_other = list(self.query_alignment.query_sequence[pos1+1:])
                if pos1 in self.query_pdb_mapping:
                    struct_pos1 = self.query_structure.pdb_residue_list[self.best_chain][self.query_pdb_mapping[pos1]]
                    struct_char1 = self.query_structure.residue_pos[self.best_chain][struct_pos1]
                else:
                    struct_pos1 = '-'
                    struct_char1 = '-'
                struct_pos_other = []
                struct_char_other = []
                for pos2 in pos_other:
                    if pos2 in self.query_pdb_mapping:
                        struct_pos2 = self.query_structure.pdb_residue_list[self.best_chain][self.query_pdb_mapping[pos2]]
                        struct_char2 = self.query_structure.residue_pos[self.best_chain][struct_pos2]
                    else:
                        struct_pos2 = '-'
                        struct_char2 = '-'
                    struct_pos_other.append(struct_pos2)
                    struct_char_other.append(struct_char2)
                data_dict['Seq Pos 1'] += [pos1] * len(pos_other)
                data_dict['Seq AA 1'] += [char1] * len(pos_other)
                data_dict['Seq Pos 2'] += pos_other
                data_dict['Seq AA 2'] += char_other
                data_dict['Struct Pos 1'] += [struct_pos1] * len(pos_other)
                data_dict['Struct AA 1'] += [struct_char1] * len(pos_other)
                data_dict['Struct Pos 2'] += struct_pos_other
                data_dict['Struct AA 2'] += struct_char_other
            data_df = pd.DataFrame(data_dict)

            def determine_sequence_separation_category(sep):
                if sep < 6:
                    category = 'Neighbors'
                elif sep < 12:
                    category = 'Short'
                elif sep < 24:
                    category = 'Medium'
                else:
                    category = 'Long'
                return category

            data_df['Seq Separation'] = data_df['Seq Pos 2'] - data_df['Seq Pos 1']
            data_df['Seq Separation Category'] = data_df['Seq Separation'].apply(
                lambda x: determine_sequence_separation_category(x))
            self.data = data_df
            end = time()
            print('Constructing internal representation took {} min'.format((end - start) / 60.0))

    def _get_all_coords(self, residue):
        """
        Get All Coords

        This method retrieves the 3D coordinates for all atoms in a residue.

        Args:
            residue (Bio.PDB.Residue): The residue for which to return coordinates.
        Returns:
            list: A list of lists where each sub lists contains the x, y, and z coordinates of a given atom in a
            residue.
        """
        all_coords = []
        cmd.select('curr_res', f'best_chain and resi {residue.get_id()[1]}')
        model = cmd.get_model('curr_res')
        if len(model.atom) > 0:
            for atm in model.atom:
                all_coords.append(np.array([atm.coord[0], atm.coord[1], atm.coord[2]], dtype=np.float32))
        else:
            raise ValueError(f'No atoms found for Structure: {self.query} Chain: {self.best_chain} Residue: {residue.get_id()[1]}')
        return all_coords

    def _get_c_alpha_coords(self, residue):
        """
        Get C Alpha Coords

        This method retrieves the 3D coordinates for the C alpha atom of a residue. If the C alpha atom is not specified
        for the given residue _get_all_coords is called instead.

        Args:
            residue (Bio.PDB.Residue): The residue for which to return coordinates.
        Returns:
            list: A list of lists where the only sub list contains the x, y, and z coordinates of the C alpha atom.
        """
        c_alpha_coords = []
        cmd.select('curr_res_ca', f'best_chain and resi {residue.get_id()[1]} and name CA')
        model = cmd.get_model('curr_res_ca')
        if len(model.atom) > 0:
            for atm in model.atom:
                c_alpha_coords.append(np.array([atm.coord[0], atm.coord[1], atm.coord[2]], dtype=np.float32))
        else:
            c_alpha_coords = self._get_all_coords(residue)
        return c_alpha_coords

    def _get_c_beta_coords(self, residue):
        """
        Get C Beta Coords

        This method retrieves the 3D coordinates for the C beta atom of a residue. If the C beta atom is not specified
        for the given residue _get_c_alpha_coords is called instead.

        Args:
            residue (Bio.PDB.Residue): The residue for which to return coordinates.
        Returns:
            list: A list of lists where the only sub list contains the x, y, and z coordinates of the C beta atom.
        """
        c_beta_coords = []
        cmd.select('curr_res_cb', f'best_chain and resi {residue.get_id()[1]} and name CB')
        model = cmd.get_model('curr_res_cb')
        if len(model.atom) > 0:
            for atm in model.atom:
                c_beta_coords.append(np.array([atm.coord[0], atm.coord[1], atm.coord[2]], dtype=np.float32))
        else:
            c_beta_coords = self._get_c_alpha_coords(residue)
        return c_beta_coords

    def _get_coords(self, residue, method='Any'):
        """
        Get Coords

        This method serves as a switch statement, its only purpose is to reduce code duplication when calling one of
        the three specific get (all, alpha, or beta) coord functions.

        Args:
            residue (Bio.PDB.Residue): The residue for which to extract the specified atom or atoms' coordinates.
            method (str): Which method of coordinate extraction to use, expected values are 'Any', 'CA' for alpha
            carbon, or 'CB' for beta carbon.
        Returns:
            list: A list of lists where each list is a single set of x, y, and z coordinates for an atom in the
            specified residue.
        """

        if method == 'Any':
            return self._get_all_coords(residue)
        elif method == 'CA':
            return self._get_c_alpha_coords(residue)
        elif method == 'CB':
            return self._get_c_beta_coords(residue)
        else:
            raise ValueError("method variable must be either 'Any', 'CA', or 'CB' but {} was provided".format(method))

    def measure_distance(self, method='Any', save_file=None):
        """
        Measure Distance

        This method measures the distances between residues in the chain identified as the best match to the query by
        the fit function. Distance can be measured in different ways, currently three different options are supported.
        This function updates the class attributes distances and dist_type.

        Args:
            method (str): Method by which to compute distances, currently options are:
                'Any' - this measures the distance between two residues as the shortest distance between any two atoms
                in the two residues structure.
                'CA' - this measures the distance as the space between the alpha carbons of two residues. If the alpha
                carbon for a given residue is not available in the PDB structure, then the distance between any atom of
                that residue and the alpha carbons of all other residues is measured.
                'CB' - this measures the distance as the space between the beta carbons of the two residues. If the beta
                carbon of a given residue is not available in the PDB structure the alpha carbon is used, or all atoms
                in the residue if the alpha carbon is also not annotated.
            save_file (str or os.path): The path to a file where the computed distances can be stored, or may have been
            stored on a previous run (optional).
    """
        start = time()
        if self.query_structure is None:
            raise ValueError('Distance cannot be measured, because no PDB was provided.')
        elif ((self.distances is not None) and (self.dist_type == method) and
              (pd.Series(['Distance', 'Contact (within {}A cutoff)'.format(self.cutoff)]).isin(self.data.columns).all())):
            return
        elif (save_file is not None) and os.path.exists(save_file):
            dists = np.load(save_file + '.npz')['dists']
        else:
            if self.best_chain is None:
                self.fit()
            self.data['Distance'] = '-'
            # Open and load PDB structure in pymol, needed for get_coord methods
            cmd.load(self.query_structure.file_name, self.query)
            cmd.select('best_chain', f'{self.query} and chain {self.best_chain}')
            dists = np.zeros((self.query_structure.size[self.best_chain], self.query_structure.size[self.best_chain]))
            coords = {}
            counter = 0
            pos = {}
            key = {}
            # Loop over all residues in the pdb
            for res_num1 in self.query_structure.pdb_residue_list[self.best_chain]:
                residue = self.query_structure.structure[0][self.best_chain][res_num1]
                # Loop over residues to calculate distance between all residues i and j
                if res_num1 not in coords:
                    pos[res_num1] = counter
                    key[counter] = res_num1
                    counter += 1
                    coords[res_num1] = np.vstack(self._get_coords(residue, method))
                for j in range(pos[res_num1]):
                    res_num2 = key[j]
                    if res_num2 not in pos:
                        continue
                    # Getting the 3d coordinates for every atom in each residue.
                    # iterating over all pairs to find all distances
                    res1 = (coords[res_num1] - coords[res_num2][:, np.newaxis])
                    norms = np.linalg.norm(res1, axis=2)
                    dists[pos[res_num1], pos[res_num2]] = dists[pos[res_num2], pos[res_num1]] = np.min(norms)
            # Delete loaded structure from pymol session
            cmd.delete(self.query)
            if save_file is not None:
                np.savez(save_file, dists=dists)
        indices = self.data.loc[(self.data['Struct Pos 1'] != '-') & (self.data['Struct Pos 2'] != '-')].index
        self.data.loc[indices, 'Distance'] = self.data.loc[indices].apply(
            lambda x: dists[pos[x['Struct Pos 1']], pos[x['Struct Pos 2']]], axis=1)
        self.data['Contact (within {}A cutoff)'.format(self.cutoff)] = self.data['Distance'].apply(
            lambda x: '-' if x == '-' else x <= self.cutoff)
        end = time()
        print('Computing the distance matrix based on the PDB file took {} min'.format((end - start) / 60.0))
        self.distances = dists
        self.dist_type = method

    def find_pairs_by_separation(self, category='Any', mappable_only=False):
        """
        Find Pairs By Separation

        This method returns all pairs of residues falling into a given category of sequence separation.

        Args:
            category (str): The category for which to return residue pairs. At the moment the following categories are
            supported:
                Neighbors - Residues 1 to 5 sequence positions apart.
                Short - Residues 6 to 12 sequences positions apart.
                Medium - Residues 13 to 24 sequences positions apart.
                Long - Residues more than 24 sequence positions apart.
                Any - Any/All pairs of residues.
            mappable_only (boolean): If True only pairs which are mappable to the PDB structure provided to the scorer
            will be returned.
        Returns:
            list: A list of tuples where the tuples are pairs of residue positions which meet the category criteria.
        """
        if category not in {'Neighbors', 'Short', 'Medium', 'Long', 'Any'}:
            raise ValueError("Category was {} must be one of the following 'Neighbors', 'Short', 'Medium', 'Long', "
                             "'Any'".format(category))
        if self.data is None:
            self.fit()
        if category == 'Any':
            category_subset = self.data
        else:
            category_subset = self.data.loc[self.data['Seq Separation Category'] == category, :]
        if mappable_only:
            unmappable_index = category_subset.index[(category_subset['Struct Pos 1'] == '-') |
                                                     (category_subset['Struct Pos 2'] == '-')]
            final_subset = category_subset.drop(unmappable_index)
        else:
            final_subset = category_subset
        final_df = final_subset[['Seq Pos 1', 'Seq Pos 2']]
        pairs = final_df.to_records(index=False).tolist()
        return pairs

    def map_predictions_to_pdb(self, ranks, predictions, coverages, threshold=0.5): # , category='Any'):
        """
        Map Predictions To PDB

        This method accepts a set of predictions and uses the mapping between the query sequence and the best PDB chain
        to extract the comparable predictions and distances.

        Args:
            ranks (np.array): An array of ranks for predicted contacts between protein residues with size nxn where n
            is the length of the query sequence used when initializing the ContactScorer.
            predictions (np.array): An array of prediction scores for contacts between protein residues with size nxn
            where n is the length of the query sequence used when initializing the ContactScorer.
            coverages (np.array): An array of coverage scores for predicted contacts between protein residues with size
            nxn where n is the length of the query sequence used when initializing the ContactScorer.
            threshold (float): The cutoff for coverage scores up to which scores (inclusive) are considered true.
        """
        # Add predictions to the internal data representation, simultaneously mapping them to the structural data.
        if self.data is None:
            self.fit()
        indices = np.triu_indices(n=self.query_alignment.seq_length, k=1)
        self.data['Rank'] = ranks[indices]
        self.data['Score'] = predictions[indices]
        self.data['Coverage'] = coverages[indices]
        self.data['True Prediction'] = self.data['Coverage'].apply(lambda x: x <= threshold)

    def _identify_relevant_data(self, category='Any', n=None, k=None):
        """
        Map Predictions To PDB

        This method accepts a set of predictions and uses the mapping between the query sequence and the best PDB chain
        to extract the comparable predictions and distances.

        Args:
            category (str/list): The category for which to return residue pairs. At the moment the following categories
            are supported:
                Neighbors - Residues 1 to 5 sequence positions apart.
                Short - Residues 6 to 12 sequences positions apart.
                Medium - Residues 13 to 24 sequences positions apart.
                Long - Residues more than 24 sequence positions apart.
                Any - Any/All pairs of residues.
            In order to return a combination of labels, a list may be provided which contains any of the strings from
            the above set of categories (e.g. ['Short', 'Medium', 'Long']).
            k (int): This value should only be specified if n is not specified. This is the number that L, the length of
            the query sequence, will be divided by to give the number of predictions to test.
            n (int): This value should only be specified if k is not specified. This is the number of predictions to
            test.
        Returns:
            np.array: A set of predictions which can be scored because they map successfully to the PDB.
            np.array: A set of distances which can be used for scoring because they map successfully to the query
            sequence.
        """
        # Defining for which of the pairs of residues there are both cET-MIp  scores and distance measurements from
        # the PDB Structure.
        if self.data is None:
            self.fit()
        if not pd.Series(['Distance', 'Contact (within {}A cutoff)'.format(self.cutoff)]).isin(self.data.columns).all():
            raise ValueError('measure_distance must be called before a specific evaluation is performed so that'
                             'contacts can be identified to compare to the predictions.')
        if not pd.Series(['Rank', 'Score', 'Coverage']).isin(self.data.columns).all():
            raise ValueError('Ranks, Scores, and Coverage values must be provided through map_predictions_to_pdb,'
                             'before a specific evaluation can be made.')
        if (k is not None) and (n is not None):
            raise ValueError('Both k and n were set for score_recall which is not a valid option.')
        if (category == 'Any') or ('Any' in category):
            category_subset = self.data
        elif type(category) == list:
            category_subset = self.data.loc[self.data['Seq Separation Category'].isin(category), :]
        else:
            category_subset = self.data.loc[self.data['Seq Separation Category'] == category, :]
        unmappable_index = category_subset.index[(category_subset['Struct Pos 1'] == '-') |
                                                 (category_subset['Struct Pos 2'] == '-')]
        final_subset = category_subset.drop(unmappable_index)
        final_df = final_subset.sort_values(by='Coverage')
        final_df['Top Predictions'] = final_df['Coverage'].rank(method='dense')
        if k:
            n = int(floor(self.query_alignment.seq_length / float(k)))
        elif n is None:
            n = self.data.shape[0]
        else:
            pass
        ind = final_df['Top Predictions'] <= n
        final_df = final_df.loc[ind, :]
        return final_df

    def score_auc(self, category='Any'):
        """
        Score AUC

        This function accepts a matrix of predictions and uses it to compute an overall AUROC when compared to the
        distances between residues computed for the PDB structure. It uses the cutoff defined when initializing the
        ContactScorer to determine the set of true positives and the mapping from query sequence to pdb determined
        when the fit function is performed.

        Args:
            category (str): The sequence separation category to score, the options are as follows:
                 Neighbors : Residues 1 to 5 sequence positions apart.
                Short : Residues 6 to 12 sequences positions apart.
                Medium : Residues 13 to 24 sequences positions apart.
                Long : Residues more than 24 sequence positions apart.
                Any : Any/All pairs of residues.
        Returns:
            np.array: The list of true positive rate values calculated when computing the roc curve.
            np.array: The list of false positive rate value calculated when computing the roc curve.
            float: The auroc determined for the roc curve.
        """
        # Checks are performed in _identify_relevant_data
        df = self._identify_relevant_data(category=category)
        # AUC computation
        if (df is not None) and (np.sum(~ df['Contact (within {}A cutoff)'.format(self.cutoff)].isnull()) !=
                                 np.sum(~ df['Coverage'].isnull())):
            raise ValueError("Lengths do not match between query sequence and the aligned pdb chain.")
        fpr, tpr, _thresholds = roc_curve(df['Contact (within {}A cutoff)'.format(self.cutoff)].values.astype(bool),
                                          1.0 - df['Coverage'].values, pos_label=True)
        auroc = auc(fpr, tpr)
        return tpr, fpr, auroc

    def plot_auc(self, auc_data, title=None, file_name=None, output_dir=None):
        """
        Plot AUC

        This function plots and saves the AUROC.  The image will be stored in the png format with dpi=300 using a name
        specified by the ContactScorer query name, cutoff, clustering constant, and date.

        Args:
            auc_data (dictionary): AUC values generated by score_auc.
            title (str): The title for the AUC plot.
            file_name (str): The file name under which to save this figure.
            output_dir (str): The full path to where the AUC plot image should be stored. If None (default) the plot
            will be stored in the current working directory.
        """
        # If there is no AUC data return without plotting
        if (auc_data[0] is None) and (auc_data[1] is None) and (auc_data[2] in {None, '-', 'NA'}):
            return
        if file_name is None:
            file_name = '{}_Cutoff{}A_roc.png'.format(self.query, self.cutoff)
        if not file_name.endswith('.png'):
            file_name = file_name + '.png'
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        # If the figure has already been plotted return
        if os.path.isfile(file_name):
            return
        plt.plot(auc_data[1], auc_data[0], label='(AUC = {0:.2f})'.format(auc_data[2]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if title is None:
            title = 'Ability to predict positive contacts in {}'.format(self.query)
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(file_name, format='png', dpi=300, fontsize=8)
        plt.close()

    def score_precision_recall(self, category='Any'):
        """
        Score Precision Recall

        This function accepts a matrix of predictions and uses it to compute an overall precision and recall when
        compared to the distances between residues computed for the PDB structure. It uses the cutoff defined when
        initializing the ContactScorer to determine the set of true positives and the mapping from query sequence to pdb
        determined when the fit function is performed.

        Args:
            category (str): The sequence separation category to score, the options are as follows:
                 Neighbors : Residues 1 to 5 sequence positions apart.
                Short : Residues 6 to 12 sequences positions apart.
                Medium : Residues 13 to 24 sequences positions apart.
                Long : Residues more than 24 sequence positions apart.
                Any : Any/All pairs of residues.
        Returns:
            np.array: The list of precision values calculated at each point along the sorted predictions.
            np.array: The list of recall values calculated at each point along the sorted predictions.
            float: The auprc determined for the precision recall curve.
        """
        # Checks are performed in _identify_relevant_data
        df = self._identify_relevant_data(category=category)
        # AUPRC computation
        if (df is not None) and (np.sum(~ df['Contact (within {}A cutoff)'.format(self.cutoff)].isnull()) !=
                                 np.sum(~ df['Coverage'].isnull())):
            raise ValueError("Lengths do not match between query sequence and the aligned pdb chain.")
        precision, recall, _thresholds = precision_recall_curve(
            df['Contact (within {}A cutoff)'.format(self.cutoff)].values.astype(bool),
            1.0 - df['Coverage'].values, pos_label=True)
        recall, precision = zip(*sorted(zip(recall, precision)))
        recall, precision = np.array(recall), np.array(precision)
        auprc = auc(recall, precision)
        return precision, recall, auprc

    def plot_auprc(self, auprc_data, title=None, file_name=None, output_dir=None):
        """
        Plot AUPRC

        This function plots and saves the AUPRC.  The image will be stored in the png format with dpi=300 using a name
        specified by the ContactScorer query name, cutoff, clustering constant, and date.

        Args:
            auprc_data (dictionary): AUPRC values generated by the score_precision_recall method.
            title (str): The title for the AUPRC plot.
            file_name (str): The file name under which to save this figure.
            output_dir (str): The full path to where the AUPRC plot image should be stored. If None (default) the plot
            will be stored in the current working directory.
        """
        # If there is no AUC data return without plotting
        if (auprc_data[0] is None) and (auprc_data[1] is None) and (auprc_data[2] in {None, '-', 'NA'}):
            return
        if file_name is None:
            file_name = '{}_Cutoff{}A_auprc.png'.format(self.query, self.cutoff)
        if not file_name.endswith('.png'):
            file_name = file_name + '.png'
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        # If the figure has already been plotted return
        if os.path.isfile(file_name):
            return
        plt.plot(auprc_data[1], auprc_data[0], label='(AUC = {0:.2f})'.format(auprc_data[2]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if title is None:
            title = 'Ability to predict positive contacts in {}'.format(self.query)
        plt.title(title)
        plt.legend(loc="lower left")
        plt.savefig(file_name, format='png', dpi=300, fontsize=8)
        plt.close()

    def score_precision(self, category='Any', k=None, n=None):
        """
        Score Precision

        This method can be used to calculate the precision of the predictions. The intention is that this method be used
        to compute precision for the top L/k or top n residue pairs, where L is the length of the query sequence and k
        is a number less than or equal to L and n is a specific number of predictions to test. Predictions in the top
        L/k or n are given a label of 1 if they are > threshold and are given a label of 0 otherwise. The true positive
        set is determined by taking the PDB measured distances for the top L/k or n residue pairs and setting them to 1
        if they are <= the cutoff provided when initializing this ContactScorer, and 0 otherwise. Precision tests that
        the ranking of residues correctly predicts structural contacts, it is given by tp / (tp + fp) as implemented by
        sklearn.

        Args:
            category (str): The sequence separation category to score, the options are as follows:
                 Neighbors - Residues 1 to 5 sequence positions apart.
                Short - Residues 6 to 12 sequences positions apart.
                Medium - Residues 13 to 24 sequences positions apart.
                Long - Residues more than 24 sequence positions apart.
                Any - Any/All pairs of residues.
            k (int): This value should only be specified if n is not specified. This is the number that L, the length of
            the query sequence, will be divided by to give the number of predictions to test.
            n (int): This value should only be specified if k is not specified. This is the number of predictions to
            test.
        Returns:
            float: The precision value computed for the predictions provided.
        """
        # Checks are performed in _identify_relevant_data
        df = self._identify_relevant_data(category=category, n=n, k=k)
        # Precision computation
        if (df is not None) and (np.sum(~ df['Contact (within {}A cutoff)'.format(self.cutoff)].isnull()) !=
                                 np.sum(~ df['Coverage'].isnull())):
            raise ValueError("Lengths do not match between query sequence and the aligned pdb chain.")
        precision = precision_score(df['Contact (within {}A cutoff)'.format(self.cutoff)].values.astype(bool),
                                    df['True Prediction'].values, pos_label=True)
        return precision

    def score_recall(self, category='Any', k=None, n=None):
        """
        Score Recall

        This method can be used to calculate the recall of the predictions. The intention is that this method be used
        to compute recall for the top L/k or top n residue pairs, where L is the length of the query sequence and k
        is a number less than or equal to L and n is a specific number of predictions to test. Predictions in the top
        L/k or n are given a label of 1 if they are >0 and are given a label of 0 otherwise. The true positive set is
        determined by taking the PDB measured distances for the top L/k or n residue pairs and setting them to 1 if they
        are <= the cutoff provided when initializing this ContactScorer, and 0 otherwise. Recall tests that the ranking
        of residues to predict all structural contacts, it is given by tp / (tp + fn) as implemented by sklearn.

        Args:
            category (str): The sequence separation category to score, the options are as follows:
                 Neighbors - Residues 1 to 5 sequence positions apart.
                Short - Residues 6 to 12 sequences positions apart.
                Medium - Residues 13 to 24 sequences positions apart.
                Long - Residues more than 24 sequence positions apart.
                Any - Any/All pairs of residues.
            k (int): This value should only be specified if n is not specified. This is the number that L, the length of
            the query sequence, will be divided by to give the number of predictions to test.
            n (int): This value should only be specified if k is not specified. This is the number of predictions to
            test.
        Returns:
            float: The recall value computed for the predictions provided.
        """
        # Checks are performed in _identify_relevant_data
        df = self._identify_relevant_data(category=category, n=n, k=k)
        # Recall computation
        if (df is not None) and (np.sum(~ df['Contact (within {}A cutoff)'.format(self.cutoff)].isnull()) !=
                                 np.sum(~ df['Coverage'].isnull())):
            raise ValueError("Lengths do not match between query sequence and the aligned pdb chain.")
        recall = recall_score(df['Contact (within {}A cutoff)'.format(self.cutoff)].values.astype(bool),
                              df['True Prediction'].values, pos_label=True)
        return recall

    def score_f1(self, category='Any', k=None, n=None):
        """
        Score F1

        This method can be used to calculate the f1 score of the predictions. The intention is that this method be used
        to compute the f1 score for the top L/k or top n residue pairs, where L is the length of the query sequence and
        k is a number less than or equal to L and n is a specific number of predictions to test. Predictions in the top
        L/k or n are given a label of 1 if they are >0 and are given a label of 0 otherwise. The true positive set is
        determined by taking the PDB measured distances for the top L/k or n residue pairs and setting them to 1 if they
        are <= the cutoff provided when initializing this ContactScorer, and 0 otherwise. F1 score is the weighted
        average of precision and recall (2 * (precision * recall) / (precision + recall)) as implemented by sklearn.

        Args:
            category (str): The sequence separation category to score, the options are as follows:
                 Neighbors - Residues 1 to 5 sequence positions apart.
                Short - Residues 6 to 12 sequences positions apart.
                Medium - Residues 13 to 24 sequences positions apart.
                Long - Residues more than 24 sequence positions apart.
                Any - Any/All pairs of residues.
            k (int): This value should only be specified if n is not specified. This is the number that L, the length of
            the query sequence, will be divided by to give the number of predictions to test.
            n (int): This value should only be specified if k is not specified. This is the number of predictions to
            test.
        Returns:
            float: The F1 score value computed for the predictions provided.
        """
        # Checks are performed in _identify_relevant_data
        df = self._identify_relevant_data(category=category, n=n, k=k)
        # F1 computation
        if (df is not None) and (np.sum(~ df['Contact (within {}A cutoff)'.format(self.cutoff)].isnull()) !=
                                 np.sum(~ df['Coverage'].isnull())):
            raise ValueError("Lengths do not match between query sequence and the aligned pdb chain.")
        f1 = f1_score(df['Contact (within {}A cutoff)'.format(self.cutoff)].values.astype(bool),
                      df['True Prediction'].values, pos_label=True)
        return f1

    def score_clustering_of_contact_predictions(self, bias=True, file_path='./z_score.tsv', w_and_w2_ave_sub=None,
                                                processes=1):
        """
        Score Clustering Of Contact Predictions

        This method employs the _clustering_z_score method to score all pairs for which predictions are made. A z-score
        of '-' means that the pair did not map to the provided PDB while 'NA' means that the sigma computed for that
        pair was equal to 0. The residues belonging to each pair are added to a set in order of pair covariance score.
        That set of residues is evaluated for clustering z-score after each pair is added.

        Args:
            bias (int or bool): option to calculate z_scores with bias (True) or no bias (False). If bias is used a j-i
            factor accounting for the sequence separation of residues, as well as their distance, is added to the
            calculation.
            file_path (str): path where the z-scoring results should be written to.
            w_and_w2_ave_sub (dict): A dictionary of the precomputed scores for E[w] and E[w^2] also returned by this
            function.
            processes (int): How many processes may be used in computing the clustering Z-scores.
        Returns:
            pd.DataFrame: Table holding residue I of a pair, residue J of a pair, the covariance score for that pair,
            the clustering Z-Score, the w score, E[w], E[w^2], sigma, and the number of residues of interest up to that
            point.
            dict: The parts of E[w^2] which can be pre-calculated and reused for later computations (i.e. cases 1, 2,
            and 3).
            float: The area under the curve defined by the z-scores and the protein coverage.
        """
        start = time()
        if self.query_structure is None:
            print('Z-Scores cannot be measured, because no PDB was provided.')
            return pd.DataFrame(), None, None
            # If data has already been computed load and return it without recomputing.
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, sep='\t', header=0, index_col=False)
            df_sub = df.replace([None, '-', 'NA'], np.nan)
            df_sub = df_sub.dropna()[['Num_Residues', 'Z-Score']]
            df_sub.drop_duplicates(inplace=True)
            df_sub['Coverage'] = df_sub['Num_Residues'] / float(self.query_alignment.seq_length)
            df_sub.sort_values(by='Coverage', ascending=True, inplace=True)
            if len(df_sub['Coverage']) == 0:
                au_scw_z_score_curve = None
            else:
                au_scw_z_score_curve = auc(df_sub['Coverage'].astype(float).values,
                                           df_sub['Z-Score'].astype(float).values)
            return df, None, au_scw_z_score_curve
        # Set up dataframe for storing results.
        data_df = self._identify_relevant_data(category='Any').loc[:, ['Seq Pos 1', 'Seq Pos 2', 'Coverage']]
        data_df.rename(columns={'Seq Pos 1': 'Res_i', 'Seq Pos 2': 'Res_j', 'Coverage': 'Covariance_Score'},
                       inplace=True)
        sorted_residues = list(zip(data_df['Res_i'], data_df['Res_j']))
        data = {'Z-Score': [], 'W': [], 'W_Ave': [], 'W2_Ave': [], 'Sigma': [], 'Num_Residues': []}
        # Set up structures to track unique sets of residues
        unique_sets = {}
        to_score = []
        residues_of_interest = set([])
        counter = -1
        prev_size = 0
        # Identify unique sets of residues to score
        for pair in sorted_residues:
            residues_of_interest.update(pair)
            curr_size = len(residues_of_interest)
            if curr_size > prev_size:
                counter += 1
                unique_sets[counter] = [pair]
                to_score.append(list(residues_of_interest))
                prev_size = curr_size
            else:
                unique_sets[counter].append(pair)
        # Compute the precomputable part (w_ave_pre, w2_ave_sub)
        if w_and_w2_ave_sub is None:
            w_and_w2_ave_sub = {'w_ave_pre': 0, 'Case1': 0, 'Case2': 0, 'Case3': 0}
            pool = Pool(processes=processes, initializer=init_compute_w_and_w2_ave_sub,
                        initargs=(self.distances, self.query_structure.pdb_residue_list[self.best_chain], bias))
            res = pool.map_async(compute_w_and_w2_ave_sub, range(self.distances.shape[0]))
            pool.close()
            pool.join()
            for cases in res.get():
                for key in cases:
                    w_and_w2_ave_sub[key] += cases[key]
        # Compute all other Z-scores
        pool2 = Pool(processes=processes, initializer=init_clustering_z_score,
                     initargs=(bias, w_and_w2_ave_sub, self.query_structure, self.query_pdb_mapping, self.distances,
                               self.best_chain))
        res = pool2.map(clustering_z_score, to_score)
        pool2.close()
        pool2.join()
        # Variables to track for computing the Area Under the SCW Z-Score curve
        x_coverage = []
        y_z_score = []
        for counter, curr_res in enumerate(res):
            curr_len = len(unique_sets[counter])
            data['Z-Score'] += [curr_res[6]] * curr_len
            data['W'] += [curr_res[7]] * curr_len
            data['W_Ave'] += [curr_res[8]] * curr_len
            data['W2_Ave'] += [curr_res[9]] * curr_len
            data['Sigma'] += [curr_res[10]] * curr_len
            data['Num_Residues'] += [curr_res[11]] * curr_len
            if curr_res[6] not in {None, '-', 'NA'}:
                y_z_score.append(curr_res[6])
                x_coverage.append(float(curr_res[11]) / self.query_alignment.seq_length)
        data_df['Z-Score'] = data['Z-Score']
        data_df['W'] = data['W']
        data_df['W_Ave'] = data['W_Ave']
        data_df['W2_Ave'] = data['W2_Ave']
        data_df['Sigma'] = data['Sigma']
        data_df['Num_Residues'] = data['Num_Residues']
        if len(x_coverage) == 0:
            au_scw_z_score_curve = None
        else:
            au_scw_z_score_curve = auc(x_coverage, y_z_score)
        # Identify all of the pairs which include unmappable positions and set their Z-scores to the appropriate value.
        data_df_unmapped = self.data.loc[self.data['Distance'] == '-',
                                         ['Seq Pos 1', 'Seq Pos 2', 'Coverage']].sort_values(by='Coverage')
        data_df_unmapped.rename(columns={'Seq Pos 1': 'Res_i', 'Seq Pos 2': 'Res_j', 'Coverage': 'Covariance_Score'},
                                inplace=True)
        data_df_unmapped['Z-Score'] = '-'
        data_df_unmapped['W'] = None
        data_df_unmapped['W_Ave'] = None
        data_df_unmapped['W2_Ave'] = None
        data_df_unmapped['Sigma'] = None
        data_df_unmapped['Num_Residues'] = None
        # Combine the mappable and unmappable index dataframes.
        df = data_df.append(data_df_unmapped)
        # Write out DataFrame
        df[['Res_i', 'Res_j', 'Covariance_Score', 'Z-Score', 'W', 'W_Ave', 'Sigma', 'Num_Residues']].to_csv(
            path_or_buf=file_path, sep='\t', header=True, index=False)
        end = time()
        print('Compute SCW Z-Score took {} min'.format((end - start) / 60.0))
        return df, w_and_w2_ave_sub, au_scw_z_score_curve

    def write_out_clustering_results(self, today, output_dir, file_name=None, prefix=None):
        """
        Write Out Clustering Results

        This method writes the covariation scores to file along with the structural validation data if available.

        Args:
            today (date/str): Today's date.
            output_dir (str): The full path to where the output file should be stored. If None (default) the file will
            be stored in the current working directory.
            file_name (str): The filename under which to save the results.
            prefix (str): A string to prepend to the specified file_name.
        """
        start = time()
        if file_name is None:
            file_name = "{}_{}.Covariance_vs_Structure.txt".format(today, self.query)
        if prefix is not None and prefix != '':
            file_name = prefix + file_name
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        if os.path.isfile(file_name):
            end = time()
            print('Contact prediction scores and structural validation data already written {} min'.format(
                (end - start) / 60.0))
            return
        header = ['Pos1', '(AA1)', 'Pos2', '(AA2)', 'Raw_Score', 'Coverage_Score', 'Residue_Dist', 'Within_Threshold']
        df = self.data[['Seq Pos 1', 'Seq AA 1', 'Seq Pos 2', 'Seq AA 2', 'Score', 'Coverage', 'Distance',
                        'Contact (within {}A cutoff)'.format(self.cutoff)]]
        df = df.rename(columns={'Seq Pos 1': 'Pos1', 'Seq AA 1': '(AA1)', 'Seq Pos 2': 'Pos2', 'Seq AA 2': '(AA2)',
                                'Score': 'Raw_Score', 'Coverage': 'Coverage_Score', 'Distance': 'Residue_Dist',
                                'Contact (within {}A cutoff)'.format(self.cutoff): 'Within_Threshold'})
        df.to_csv(file_name, sep='\t', header=True, index=False, columns=header)
        end = time()
        print('Writing the contact prediction scores and structural validation data to file took {} min'.format(
            (end - start) / 60.0))

    def evaluate_predictor(self, predictor, verbosity, out_dir, dist='Any', biased_w2_ave=None,
                           unbiased_w2_ave=None, processes=1, threshold=0.5,  file_prefix='Scores_', plots=True):
        """
        Evaluate Predictor

        This method can be used to perform a number of validations at once for the predictions made by a given
        predictor.

        Args:
            predictor (ETMIPC/ETMIPWrapper/DCAWrapper/EVCouplingsWrapper): A predictor which has already calculated its
            covariance scores.
            verbosity (int): What level of output to produce.
                1. Tests the AUROC, AUPRC, and AUTPRFDRC of contact prediction at different levels of sequence
                separation.
                2. Tests the precision, recall, and f1 score of  contact prediction at different levels of sequence
                separation and top prediction levels (L, L/2 ... L/10).
                3. Tests the clustering Z-score of the predictions and writes them to file as well as plotting Z-Scores
                against residue count.
            out_dir (str/path): The path at which to save
            dist (str): Which type of distance computation to use to determine if residues are in contact, for further
            details see the measure_distance method. Current choices are:
                Any - Measures the minimum distance between two residues considering all of their atoms.
                CB - Measures the distance between two residues using their Beta Carbons as the measuring point.
                CA - Measures the distance between two residues using their Alpha Carbons as the measuring point.
            biased_w2_ave (dict): A dictionary of the precomputed scores for E[w^2] for biased z-score computation also
            returned by this function.
            unbiased_w2_ave (dict): A dictionary of the precomputed scores for E[w^2] for unbaised z-score
            computation also returned by this function.
            processes (int): The number of processes to use when computing the clustering z-score (if specified).
            threshold (float): Value above which a prediction will be labeled 1 (confident/true) when computing
            precision, recall, and f1 scores (this should be the expected coverage value).
            file_prefix (str): string to prepend before filenames.
            plots (boolean): Whether to create and save plots associated with the scores computed.
        Returns:
            pandas.DataFrame: A DataFrame containing the specified amount of information (see verbosity) for the raw
            scores provided when evaluating this predictor. Possible column headings include: 'Time',
            'Sequence_Separation', 'Distance', 'AUROC', 'Precision (L)', 'Precision (L/2)', 'Precision (L/3)',
            'Precision (L/4)', 'Precision (L/5)', 'Precision (L/6)', 'Precision (L/7)', 'Precision (L/8)',
            'Precision (L/9)', and 'Precision (L/10)'.
            dict: A dictionary of the precomputed scores for E[w^2] for biased z-score computation.
            dict: A dictionary of the precomputed scores for E[w^2] for unbiased z-score computation.
        """
        # if not isinstance(predictor, Predictor):
        #     import inspect
        #     print(inspect.getmro(type(predictor)))
        #     print(Predictor)
        #     raise TypeError('To evaluate a predictor it must have type Predictor, please use a valid Predictor.')
        score_fn = os.path.join(out_dir, '{}_Evaluation_Dist-{}.txt'.format(file_prefix, dist))
        # If the evaluation has already been performed load the data and return it
        if os.path.isfile(score_fn):
            score_df = pd.read_csv(score_fn, sep='\t', header=0, index_col=False)
            return score_df, None, None
        columns = ['Sequence_Separation', 'Distance', 'Top K Predictions', 'AUROC', 'AUPRC', 'AUTPRFDRC',
                   'Precision', 'Recall', 'F1 Score', 'Max Biased Z-Score', 'AUC Biased Z-Score',
                   'Max Unbiased Z-Score', 'AUC Unbiased Z-Score']
        # Retrieve coverages or computes them from the scorer
        ranks = predictor.rankings
        coverages = predictor.coverages
        score_stats, b_w2_ave, u_w2_ave = self.evaluate_predictions(
            scores=predictor.scores, coverages=coverages, ranks=ranks, verbosity=verbosity, out_dir=out_dir, dist=dist,
            file_prefix=file_prefix, biased_w2_ave=biased_w2_ave, unbiased_w2_ave=unbiased_w2_ave, processes=processes,
            threshold=threshold, plots=plots)
        if (biased_w2_ave is None) and (b_w2_ave is not None):
            biased_w2_ave = b_w2_ave
        if (unbiased_w2_ave is None) and (u_w2_ave is not None):
            unbiased_w2_ave = u_w2_ave
        if score_stats == {}:
            score_df = None
        else:
            score_df = pd.DataFrame(score_stats)
            score_df.to_csv(path_or_buf=score_fn, columns=[x for x in columns if x in score_stats], sep='\t',
                            header=True, index=False)
        return score_df, biased_w2_ave, unbiased_w2_ave

    def evaluate_predictions(self, verbosity, out_dir, scores, coverages, ranks, dist='Any', file_prefix='',
                             biased_w2_ave=None, unbiased_w2_ave=None, processes=1, threshold=0.5, plots=True):
        """
        Evaluate Predictions

        This function evaluates a matrix of covariance predictions to the specified verbosity level.

        Args:
            scores (np.array): The predicted scores for pairs of residues in a sequence alignment.
            verbosity (int): What level of output to produce.
                1. Tests the AUROC, AUPRC, and AUTPRFDRC of contact prediction at different levels of sequence
                separation.
                2. Tests the precision, recall, and f1 score of  contact prediction at different levels of sequence
                separation and top prediction levels (L, L/2 ... L/10).
                3. Tests the clustering Z-score of the predictions and writes them to file as well as plotting Z-Scores
                against residue count.
            out_dir (str/path): The path at which to save
            dist (str): Which type of distance computation to use to determine if residues are in contact, choices are:
                Any - Measures the minimum distance between two residues considering all of their atoms.
                CB - Measures the distance between two residues using their Beta Carbons as the measuring point.
                CA - Measures the distance between two residues using their Alpha Carbons as the measuring point.
            file_prefix (str): string to prepend before filenames.
            biased_w2_ave (dict): A dictionary of the precomputed scores for E[w^2] for biased z-score computation also
            returned by this function.
            unbiased_w2_ave (dict): A dictionary of the precomputed scores for E[w^2] for unbaised z-score
            computation also returned by this function.
            processes (int): The number of processes to use when computing the clustering z-score (if specified).
            threshold (float): Value above which a prediction will be labeled 1 (confident/true) when computing
            precision, recall, and f1 scores.
            plots (boolean): Whether to create and save plots associated with the scores computed.
        Returns:
            dict. The stats computed for this matrix of scores, if a dictionary of stats was passed in the current stats
            are added to the previous ones.
            dict. A dictionary of the precomputed scores for E[w^2] for biased z-score computation.
            dict. A dictionary of the precomputed scores for E[w^2] for unbaised z-score computation.
        """
        stats = {}
        self.fit()
        self.measure_distance(method=dist)
        self.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=threshold)
        duplicate = 1
        # Verbosity 1
        stats['AUROC'] = []
        stats['AUPRC'] = []
        stats['Distance'] = []
        stats['Sequence_Separation'] = []
        for separation in ['Any', 'Neighbors', 'Short', 'Medium', 'Long', ['Neighbors', 'Short', 'Medium'],
                           ['Neighbors', 'Short'], ['Short', 'Medium', 'Long'], ['Medium', 'Long']]:
            # AUC Evaluation
            auc_roc = self.score_auc(category=separation)
            auc_prc = self.score_precision_recall(category=separation)
            if plots:
                self.plot_auc(auc_data=auc_roc, title='AUROC Evaluation', output_dir=out_dir,
                              file_name=file_prefix + 'AUROC_Evaluation_Dist-{}_Separation-{}'.format(dist, separation))
                self.plot_auprc(auprc_data=auc_prc, title='AUPRC Evaluation', output_dir=out_dir,
                                file_name=file_prefix + 'AUPRC_Evaluation_Dist-{}_Separation-{}'.format(
                                    dist, separation))
            if verbosity >= 2:
                duplicate = 10
                if 'F1 Score' not in stats:
                    stats['F1 Score'] = []
                    stats['Precision'] = []
                    stats['Recall'] = []
                    stats['Top K Predictions'] = []
                for k in range(1, 11):
                    if k == 1:
                        top_preds_label = 'L'
                    else:
                        top_preds_label = 'L/{}'.format(k)
                    f1 = self.score_f1(k=k, category=separation)
                    precision = self.score_precision(k=k, category=separation)
                    recall = self.score_recall(k=k, category=separation)
                    stats['F1 Score'].append(f1)
                    stats['Precision'].append(precision)
                    stats['Recall'].append(recall)
                    stats['Top K Predictions'].append(top_preds_label)
            stats['AUROC'] += [auc_roc[2]] * duplicate
            stats['AUPRC'] += [auc_prc[2]] * duplicate
            stats['Distance'] += [dist] * duplicate
            stats['Sequence_Separation'] += ((['-'.join(separation)] if isinstance(separation, list) else [separation])
                                             * duplicate)
        duplicate *= 5
        if verbosity >= 3:
            # Score Prediction Clustering
            z_score_fn = os.path.join(out_dir, file_prefix + 'Dist-{}_{}_ZScores.tsv')
            z_score_plot_fn = os.path.join(out_dir, file_prefix + 'Dist-{}_{}_ZScores.png')
            z_score_biased, b_w2_ave, b_scw_z_auc = self.score_clustering_of_contact_predictions(
                1.0 - coverages, bias=True, file_path=z_score_fn.format(dist, 'Biased'), w2_ave_sub=biased_w2_ave,
                processes=processes)
            if (biased_w2_ave is None) and (b_w2_ave is not None):
                biased_w2_ave = b_w2_ave
            stats['Max Biased Z-Score'] = [np.max(pd.to_numeric(z_score_biased['Z-Score'], errors='coerce'))] * duplicate
            stats['AUC Biased Z-Score'] = [b_scw_z_auc] * duplicate
            z_score_unbiased, u_w2_ave, u_scw_z_auc = self.score_clustering_of_contact_predictions(
                1.0 - coverages, bias=False, file_path=z_score_fn.format(dist, 'Unbiased'), w2_ave_sub=unbiased_w2_ave,
                processes=processes)
            if (unbiased_w2_ave is None) and (u_w2_ave is not None):
                unbiased_w2_ave = u_w2_ave
            stats['Max Unbiased Z-Score'] = [np.max(pd.to_numeric(z_score_unbiased['Z-Score'], errors='coerce'))] * duplicate
            stats['AUC Unbiased Z-Score'] = [u_scw_z_auc] * duplicate
            if plots:
                plot_z_scores(z_score_biased, z_score_plot_fn.format(dist, 'Biased'))
                plot_z_scores(z_score_unbiased, z_score_plot_fn.format(dist, 'Unbiased'))
        return stats, biased_w2_ave, unbiased_w2_ave


def write_out_contact_scoring(today, alignment, scores, coverages, cluster_scores=None, branch_scores=None,
                              file_name=None, output_dir=None):
    """
    Write out contact scoring

    This method writes the results of covariation scoring to file.

    Args:
        today (date): Todays date.
        alignment (SeqAlignment): Alignment associated with the scores being written to file.
        scores (numpy.array): A matrix which represents the integration of coupling scores across all clusters
        defined at that clustering constant.
        coverages (numpy.array): This dictionary maps clustering constants to a matrix of normalized coupling scores
        between 0 and 100, computed from the scores matrices.
        cluster_scores (numpy.array): The coupling scores for all positions in the query sequences at the specified
        clustering constant created by hierarchical clustering.
        branch_scores (numpy.array): This dictionary maps clustering constants to a matrix which combines the
        scores from the whole_mip_matrix, all lower clustering constants, and this clustering constant.
        file_name (str): The name with which to save the file. If None the following string template will be used:
        "{}_{}.all_scores.txt".format(today, self.query_alignment.query_id.split('_')[1])
        output_dir (str): The full path to where the output file should be stored. If None (default) the file will be
        stored in the current working directory.
    """
    start = time()
    header = ['Pos1', 'AA1', 'Pos2', 'AA2']
    if cluster_scores is not None:
        header += ['Raw_Score_{}'.format(i) for i in map(str, [k + 1 for k in sorted(cluster_scores.keys())])]
    if branch_scores is not None:
        header += ['Integrated_Score', 'Final_Score', 'Coverage_Score']
    else:
        header += ['Final_Score', 'Coverage_Score']
    file_dict = {key: [] for key in header}
    for i in range(0, alignment.seq_length):
        for j in range(i + 1, alignment.seq_length):
            res1 = i + 1
            res2 = j + 1
            file_dict['Pos1'].append(res1)
            file_dict['AA1'].append(one_to_three(alignment.query_sequence[i]))
            file_dict['Pos2'].append(res2)
            file_dict['AA2'].append(one_to_three(alignment.query_sequence[j]))
            if cluster_scores is not None:
                for c in cluster_scores:
                    file_dict['Raw_Score_{}'.format(c + 1)].append(round(cluster_scores[c][i, j], 4))
            file_dict['Final_Score'].append(round(scores[i, j], 4))
            if branch_scores is not None:
                file_dict['Integrated_Score'].append(round(branch_scores[i, j], 4))
            file_dict['Coverage_Score'].append(round(coverages[i, j], 4))
    if file_name is None:
        file_name = "{}_{}.all_scores.txt".format(today, alignment.query_id.split('_')[1])
    if output_dir:
        file_name = os.path.join(output_dir, file_name)
    df = pd.DataFrame(file_dict)
    df.to_csv(file_name, sep='\t', header=True, index=False, columns=header)
    end = time()
    print('Writing the contact prediction scores to file took {} min'.format((end - start) / 60.0))


def plot_z_scores(df, file_path=None):
    """
    Plot Z-Scores

    This method accepts a DataFrame of residue positions and z-scores produced by
    score_clustering_of_contact_predictions to plot a scatter plot.

    Args:
        df (pd.DataFrame): DataFrame containing at least the 'Num_Residues' and 'Z-Score' columns produced after
        running the score_clustering_of_contact_predictions method.
        file_path (str): Path at which to save the plot produced by this call.
    """
    # If there is no data to plot return
    if df.empty:
        return
    if file_path is None:
        file_path = './zscore_plot.png'
    # If the figure has already been plotted return
    if os.path.isfile(file_path):
        return
    plotting_data = df.loc[~df['Z-Score'].isin(['-', 'NA']), ['Num_Residues', 'Z-Score']]
    scatterplot(x='Num_Residues', y='Z-Score', data= plotting_data)
    plt.savefig(file_path)
    plt.close()


def heatmap_plot(name, data_mat, output_dir=None):
    """
    Heatmap Plot

    This method creates a heatmap using the Seaborn plotting package. The intended data are covariance raw scores or
    coverage scores.

    Args:
        name (str): Name used as the title of the plot and the filename for the saved figure.
        data_mat (np.array): A matrix of scores. This input should either be the score or coverage matrices from a
        predictor like the ETMIPC/ETMIPWrapper/DCAWrapper classes.
        output_dir (str): The full path to where the heatmap plot image should be stored. If None (default) the plot
        will be stored in the current working directory.
    """
    image_name = name.replace(' ', '_') + '.png'
    if output_dir:
        image_name = os.path.join(output_dir, image_name)
    # If the figure has already been plotted return
    if os.path.isfile(image_name):
        return
    dm_max = np.max(data_mat)
    dm_min = np.min(data_mat)
    plot_max = max([dm_max, abs(dm_min)])
    heatmap(data=data_mat, cmap='jet', center=0.0, vmin=-1 * plot_max,
            vmax=plot_max, cbar=True, square=True)
    plt.title(name)
    plt.savefig(image_name)
    plt.close()


def surface_plot(name, data_mat, output_dir=None):
    """
    Surface Plot

    This method creates a surface plot using the matplotlib plotting package. The data used is expected to come from the
    scores or coverage data from a covariance/contact predictor.

    Args:
        name (str): Name used as the title of the plot and the filename for the saved figure.
        data_mat (np.array): A matrix of scores. This input should either be the coverage or score matrices from a
        predictor like the ETMIPC/ETMIPWrapper/DCAWrapper classes.
        output_dir (str): The full path to where the surface plot image should be stored. If None (default) the plot
        will be stored in the current working directory.
    """
    image_name = name.replace(' ', '_') + '.png'
    if output_dir:
        image_name = os.path.join(output_dir, image_name)
    # If the figure has already been plotted return
    if os.path.isfile(image_name):
        return
    dm_max = np.max(data_mat)
    dm_min = np.min(data_mat)
    plot_max = max([dm_max, abs(dm_min)])
    x = y = np.arange(max(data_mat.shape))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, data_mat, cmap='jet', linewidth=0, antialiased=False)
    ax.set_zlim(-1 * plot_max, plot_max)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(image_name)
    plt.close()


def init_compute_w_and_w2_ave_sub(dists, structure_res_num, bias_bool):
    """
    Init Compute w and w^2 Average Sub-problems

    Method initializes shared values used by all processes working on parts of the w and w^2 average sub-problems.

    Args:
        dists (np.array): The array of distances between residues in the tested structure.
        structure_res_num (list/array): The residue numbers for each position in the structure as assigned in the PDB.
        bias_bool (int or bool): Option to calculate z_scores with bias (True) or no bias (False). If bias is used a j-i
        factor accounting for the sequence separation of residues, as well as their distance, is added to the
        calculation.
    """
    global distances
    distances = dists
    global w2_ave_residues
    w2_ave_residues = structure_res_num
    global bias
    bias = bias_bool
    global cutoff
    # The distance for structural clustering has historically been 4.0 Angstroms and can be found in all previous
    # versions of the code and paper descriptions, it had previously been set as a variable but this changes the meaning
    # of the result and makes it impossible to compare to previous work.
    cutoff = 4.0


def compute_w_and_w2_ave_sub(res_i):
    """
    Compute W and W^2 Average Sub-problems

    Solves the cases of the w and w^2 average sub-problem for a specific position in the protein. These can be combined
    across all positions to get the complete solution (after the pool of workers has finished).

    Args:
        res_i (int): The position in the protein for which to compute the parts of the w and w^2 average sub-problems.
    Returns:
        dict: The parts of E[w] (w_ave before multiplication with the pi1 coefficient, i.e. w_ave_pre) and E[w^2] (cases
         1, 2, and 3) which can be pre-calculated and reused for later computations.
    """
    cases = {'w_ave_pre': 0, 'Case1': 0, 'Case2': 0, 'Case3': 0}
    for res_j in range(res_i + 1, distances.shape[1]):
        if distances[res_i][res_j] >= cutoff:
            continue
        if bias:
            s_ij = np.abs(w2_ave_residues[res_j] - w2_ave_residues[res_i])
        else:
            s_ij = 1
        cases['w_ave_pre'] += s_ij
        for res_x in range(distances.shape[0]):
            for res_y in range(res_x + 1, distances.shape[1]):
                if distances[res_x][res_y] >= cutoff:
                    continue
                if bias:
                    s_xy = np.abs(w2_ave_residues[res_y] - w2_ave_residues[res_x])
                else:
                    s_xy = 1
                if (res_i == res_x and res_j == res_y) or (res_i == res_y and res_j == res_x):
                    curr_case = 'Case1'
                elif (res_i == res_x) or (res_j == res_y) or (res_i == res_y) or (res_j == res_x):
                    curr_case = 'Case2'
                else:
                    curr_case = 'Case3'
                cases[curr_case] += s_ij * s_xy
    return cases


def init_clustering_z_score(bias_bool, w_and_w2_ave_sub, curr_pdb, map_to_structure, residue_dists, best_chain):
    """
    Init Clustering Z-Score

    This method initializes a set of processes in a multiprocessing pool so that they can compute the clustering Z-Score
    with minimal data duplication.

    Args:
        bias_bool (int or bool): Option to calculate z_scores with bias (True) or no bias (False). If bias is used a j-i
        factor accounting for the sequence separation of residues, as well as their distance, is added to the
        calculation.
        w_and_w2_ave_sub (dict): A dictionary of the precomputed scores for E[w] and E[w^2].
        curr_pdb (PDBReference): Instance representing the PDB from which the structural data for this analysis comes.
        map_to_structure (dict): A mapping from the index of the query sequence to the index of the pdb chain's
        sequence for those positions which match according to a pairwise global alignment.
        residue_dists (np.array): The distances between residues, used for determining those which are in contact in
        order to assess predictions.
        best_chain (str): The chain being considered during this analysis.
    """
    global bias
    bias = bias_bool
    global w_and_w2_ave_sub_dict
    w_and_w2_ave_sub_dict = w_and_w2_ave_sub
    global query_structure
    query_structure = curr_pdb
    global query_pdb_mapping
    query_pdb_mapping = map_to_structure
    global distances
    distances = residue_dists
    global cutoff
    # The distance for structural clustering has historically been 4.0 Angstroms and can be found in all previous
    # versions of the code and paper descriptions, it had previously been set as a variable but this changes the meaning
    # of the result and makes it impossible to compare to previous work.
    cutoff = 4.0
    global query_chain
    query_chain = best_chain


def clustering_z_score(res_list):
    """
    Clustering Z Score

    Calculate z-score (z_S) for residue selection res_list=[1,2,...]
        z_S = (w-<w>_S)/sigma_S
    The steps are:
        1. Calculate Selection Clustering Weight (SCW) 'w'
        2. Calculate mean SCW (<w>_S) in the ensemble of random selections of len(res_list) residues
        3. Calculate mean square SCW (<w^2>_S) and standard deviation (sigma_S) Reference 2

    Args:
        res_list (list): a list of int's of protein residue numbers, e.g. ET residues (residues of interest)
    Returns:
        float: The z-score calculated for the residues of interest for the PDB provided with this ContactScorer.
        float: The w (clustering) score calculated for the residues of interest for the PDB provided with this
        ContactScorer.
        float: The E[w] (clustering expectation) score calculated over all residues of the PDB provided with this
        ContactScorer.
        float: The E[w^2] score calculated over all pairs of pairs of residues for the PDB provided with this
        ContactScorer.
        float: The sigma score calculated over all pairs of pairs of residues for the PDB provided with his
        ContactScorer.
        int: The number of residues in the list of interest.

    This method was formalized by Ivana Mihalek (Reference 1) and adapted from code written by Rhonald Lua in Python
    (Reference 2) which was adapted from code written by Angela Wilkins (Reference 3).

    References:
        1. I. Mihalek, I. Res, H. Yao, O. Lichtarge, Combining Inference from Evolution and Geometric Probability in
        Protein Structure Evaluation, Journal of Molecular Biology, Volume 331, Issue 1, 2003, Pages 263-279,
        ISSN 0022-2836, https://doi.org/10.1016/S0022-2836(03)00663-6.
        (http://www.sciencedirect.com/science/article/pii/S0022283603006636)
        2. Lua RC, Lichtarge O. PyETV: a PyMOL evolutionary trace viewer to analyze functional site predictions in
        protein complexes. Bioinformatics. 2010;26(23):2981-2982. doi:10.1093/bioinformatics/btq566.
        3. Wilkins AD, Lua R, Erdin S, Ward RM, Lichtarge O. Sequence and structure continuity of evolutionary
        importance improves protein functional site discovery and annotation. Protein Sci. 2010;19(7):1296-311.
    """
    res_list = sorted(res_list)
    if query_structure is None:
        print('Z-Score cannot be measured, because no PDB was provided.')
        return None, None, None, None, None, None, '-', None, None, None, None, len(res_list)
    # Check that there is a valid bias values
    if bias is not True and bias is not False:
        raise ValueError('Bias term may be True or False, but {} was provided'.format(bias))
    # Make sure a query_pdb_mapping exists
    if query_pdb_mapping is None:
        raise ValueError('ContactScorer instance must be fit before attempting to perform structural clustering'
                         'weighted z-scoring.')
    # Make sure all residues in res_list are mapped to the PDB structure in use
    if not all(res in query_pdb_mapping for res in res_list):
        print('At least one residue of interest is not present in the PDB provided')
        print(', '.join([str(x) for x in res_list]))
        print(', '.join([str(x) for x in query_pdb_mapping.keys()]))
        return None, None, None, None, None, None, '-', None, None, None, None, len(res_list)
    positions = list(range(distances.shape[0]))
    a = distances < cutoff
    a[positions, positions] = 0
    s_i = np.in1d(positions, [query_pdb_mapping[r] for r in res_list])
    s_ij = np.outer(s_i, s_i)
    s_ij[positions, positions] = 0
    if bias:
        converted_positions = np.array(query_structure.pdb_residue_list[query_chain])
        bias_ij = np.subtract.outer(converted_positions, converted_positions)
        bias_ij = np.abs(bias_ij)
    else:
        bias_ij = np.ones(s_ij.shape, dtype=np.float64)
    w = np.sum(np.tril(a * s_ij * bias_ij))
    # Calculate w, <w>_S, and <w^2>_S.
    # Use expressions (3),(4),(5),(6) in Reference.
    m = len(res_list)
    l = len(query_structure.seq[query_chain])
    pi1 = m * (m - 1.0) / (l * (l - 1.0))
    pi2 = pi1 * (m - 2.0) / (l - 2.0)
    pi3 = pi2 * (m - 3.0) / (l - 3.0)
    w_ave = w_and_w2_ave_sub_dict['w_ave_pre'] * pi1
    w2_ave = ((pi1 * w_and_w2_ave_sub_dict['Case1']) + (pi2 * w_and_w2_ave_sub_dict['Case2']) +
              (pi3 * w_and_w2_ave_sub_dict['Case3']))
    sigma = math.sqrt(w2_ave - w_ave * w_ave)
    # Response to Bioinformatics reviewer 08/24/10
    if sigma == 0:
        return a, m, l, pi1, pi2, pi3, 'NA', w, w_ave, w2_ave, sigma, m
    z_score = (w - w_ave) / sigma
    return a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, m
