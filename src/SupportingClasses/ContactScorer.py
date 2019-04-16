"""
Created on Sep 1, 2018

@author: dmkonecki
"""
import os
import math
import datetime
import numpy as np
import pandas as pd
from time import time
from math import floor
from scipy.stats import rankdata
from multiprocessing import Pool
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.PDB.Polypeptide import one_to_three
from sklearn.metrics import auc, roc_curve, precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from seaborn import heatmap, scatterplot
from SeqAlignment import SeqAlignment
from PDBReference import PDBReference


class ContactScorer(object):
    """
    ContactScorer

    This class is meant to abstract the process of scoring a set of contact predictions away from the actual method
    (previously it was included in the ETMIPC class).  This is being done for two main reasons. First, contact
    predictions are being made with several different methods and so their scoring should be performed consistently by
    an other object or function. Second, there are many ways to score contact predictions and at the moment only
    overall AUROC is being used. This class will support several other scoring methods such as the Precision at L/K
    found commonly in the literature as well as methods internal to the lab like the clustering Z-score derived from
    prior ET work.

    Attributes:
        query (str): The name of the query structure. When '>query_' is prepended, this should be the name of the query
        in the provided alignment.
        query_alignment (str/SeqAlignment): Path to the alignment being evaluated in this contact scoring prediction
        task. This is eventually updated to a SeqAlignment object, after the alignment has been imported.
        query_structure (str/PDBReference): Path to the pdb file to use for evaluating the contact predictions made.
        This is eventually updated to a PDBReference once the pdb file has been imported.
        cutoff (float): Value to use as a distance cutoff for contact prediction.
        best_chain (str): The chain in the provided pdb which most closely matches the query sequence as determined by
        pairwise global alignment.
        query_pdb_mapping (dict): A mapping from the index of the query sequence to the index of the pdb chain's
        sequence for those positions which match according to a pairwise global alignment.
        _specific_mapping (dict): A mapping for the indices of the distance matrix used to determine residue contacts.
        distances (np.array): The distances between residues, used for determining those which are in contact in order
        to assess predictions.
        dist_type (str): The type of distance computed for assessement (expecting 'Any', 'CB' - Beta Carbon, 'CA' -
        alpha carbon.
    """

    def __init__(self, query, seq_alignment, pdb_reference, cutoff):
        """
        __init__

        This function initializes a new ContactScorer, it accepts paths to an alignment and if available a pdb file to be
        used in the assessment of contact predictions for a given query. It also requires a cutoff which is used to
        denote which residues are actually in contact, based on their distance to one another.

        Args:
            query (str): The name of the query structure. When '>query_' is prepended, this should be the name of the
            query in the provided alignment.
            seq_alignment (str/path, SeqAlignment): Path to the alignment being evaluated in this contact scoring
            prediction task or an already initialized SeqAlignment object.
            pdb_reference (str/path, PDBReference): The object containing the PDB structure of interest, or an already
            initialized PDBReference object.
            cutoff (int or float): The distance between two residues at which a true contact is said to be occurring.
        """
        self.query = query
        if type(seq_alignment) is SeqAlignment:
            self.query_alignment = seq_alignment
        else:
            self.query_alignment = os.path.abspath(seq_alignment)
        if type(pdb_reference) is PDBReference:
            self.query_structure = pdb_reference
        else:
            self.query_structure = os.path.abspath(pdb_reference)
        self.cutoff = cutoff
        self.best_chain = None
        self.query_pdb_mapping = None
        self._specific_mapping = None
        self.distances = None
        self.dist_type = None

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
        if (self.best_chain is None) or (self.query_pdb_mapping is None):
            start = time()
            if self.query_alignment is None:
                raise ValueError('Scorer cannot be fit, because no alignment was provided.')
            else:
                if isinstance(self.query_alignment, str):
                    self.query_alignment = SeqAlignment(file_name=self.query_alignment, query_id=self.query)
                self.query_alignment.import_alignment()
                self.query_alignment.remove_gaps()
            if self.query_structure is None:
                raise ValueError('Scorer cannot be fit, because no PDB was provided.')
            else:
                if isinstance(self.query_structure, str):
                    self.query_structure = PDBReference(pdb_file=self.query_structure)
                self.query_structure.import_pdb(structure_id=self.query)
            if self.best_chain is None:
                best_chain = None
                best_alignment = None
                for ch in self.query_structure.seq:
                    curr_align = pairwise2.align.globalxs(self.query_alignment.query_sequence,
                                                          self.query_structure.seq[ch], -1, 0)
                    if (best_alignment is None) or (best_alignment[0][2] < curr_align[0][2]):
                        best_alignment = curr_align
                        best_chain = ch
                print(format_alignment(*best_alignment[0]))
                self.best_chain = best_chain
            else:
                best_alignment = pairwise2.align.globalxs(self.query_alignment.query_sequence,
                                                          self.query_structure.seq[self.best_chain], -1, 0)
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

    @staticmethod
    def _get_all_coords(residue):
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
        for atom in residue:
            all_coords.append(atom.get_coord())
        return all_coords

    @staticmethod
    def _get_c_alpha_coords(residue):
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
        try:
            c_alpha_coords.append(residue['CA'].get_coord())
        except KeyError:
            c_alpha_coords += ContactScorer._get_all_coords(residue)
        return c_alpha_coords

    @staticmethod
    def _get_c_beta_coords(residue):
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
        try:
            c_beta_coords.append(residue['CB'].get_coord())
        except KeyError:
            c_beta_coords = ContactScorer._get_c_alpha_coords(residue)
        return c_beta_coords

    @staticmethod
    def _get_coords(residue, method='Any'):
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
            return ContactScorer._get_all_coords(residue)
        elif method == 'CA':
            return ContactScorer._get_c_alpha_coords(residue)
        elif method == 'CB':
            return ContactScorer._get_c_beta_coords(residue)
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
            stored on a previous run.
    """
        start = time()
        if self.query_structure is None:
            raise ValueError('Distance cannot be measured, because no PDB was provided.')
        elif (self.distances is not None) and (self.dist_type == method):
            return
        elif (save_file is not None) and os.path.exists(save_file):
            dists = np.load(save_file + '.npz')['dists']
        else:
            if self.best_chain is None:
                self.fit()
            dists = np.zeros((self.query_structure.size[self.best_chain], self.query_structure.size[self.best_chain]))
            coords = {}
            counter = 0
            pos = {}
            key = {}
            # Loop over all residues in the pdb
            for residue in self.query_structure.structure[0][self.best_chain]:
                # Loop over residues to calculate distance between all residues i and j
                key1 = residue.get_id()[1]
                if key1 not in coords:
                    pos[key1] = counter
                    key[counter] = key1
                    counter += 1
                    coords[key1] = np.vstack(ContactScorer._get_coords(residue, method))
                for j in range(pos[key1]):
                    key2 = key[j]
                    if key2 not in pos:
                        continue
                    # Getting the 3d coordinates for every atom in each residue.
                    # iterating over all pairs to find all distances
                    res1 = (coords[key1] - coords[key2][:, np.newaxis])
                    norms = np.linalg.norm(res1, axis=2)
                    dists[pos[key1], pos[key2]] = dists[pos[key2], pos[key1]] = np.min(norms)
            if save_file is not None:
                np.savez(save_file, dists=dists)
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
        pairs = []
        for i in range(self.query_alignment.seq_length):
            if mappable_only and (i not in self.query_pdb_mapping):
                continue
            for j in range(i + 1, self.query_alignment.seq_length):
                if mappable_only and (j not in self.query_pdb_mapping):
                    continue
                separation = j - i
                if category == 'Neighbors' and (separation < 1 or separation >= 6):
                    continue
                elif category == 'Short' and (separation < 6 or separation >=13):
                    continue
                elif category == 'Medium' and (separation < 13 or separation >= 24):
                    continue
                elif category == 'Long' and separation < 24:
                    continue
                else:
                    pass
                pairs.append((i, j))
        return pairs

    def _map_predictions_to_pdb(self, predictions, category='Any'):
        """
        Map Predictions To PDB

        This method accepts a set of predictions and uses the mapping between the query sequence and the best PDB chain
        to extract the comparable predictions and distances.

        Args:
            predictions (np.array): An array of predictions for contacts between protein residues with size nxn where n
            is the length of the query sequence used when initializing the ContactScorer.
        Returns:
            np.array: A set of predictions which can be scored because they map successfully to the PDB.
            np.array: A set of distances which can be used for scoring because they map successfully to the query
            sequence.
        """
        # Defining for which of the pairs of residues there are both cET-MIp  scores and distance measurements from
        # the PDB Structure.
        if self.distances is None:
            raise ValueError('Distances not yet measured!')
        if self._specific_mapping is None:
            self._specific_mapping = {}
        if category not in self._specific_mapping:
            if predictions.shape[0] != self.query_alignment.seq_length:
                raise ValueError('Size of predictions ({}) does not match expectations based on query sequence ({})!'.format(predictions.shape[0], self.query_alignment.seq_length))
            indices = np.triu_indices(self.query_alignment.seq_length, 1)
            mappable_pos = np.array(self.query_pdb_mapping.keys())
            x_mappable = np.in1d(indices[0], mappable_pos)
            y_mappable = np.in1d(indices[1], mappable_pos)
            final_mappable = x_mappable & y_mappable
            indices = (indices[0][final_mappable], indices[1][final_mappable])
            # Mapping indices used for predictions so that they can be used to retrieve correct distances from PDB
            # distances matrix.
            keys = sorted(self.query_pdb_mapping.keys())
            values = [self.query_pdb_mapping[k] for k in keys]
            replace = np.array([keys, values])
            mask1 = np.in1d(indices[0], replace[0, :])
            indices[0][mask1] = replace[1, np.searchsorted(replace[0, :], indices[0][mask1])]
            mask2 = np.in1d(indices[1], replace[0, :])
            indices[1][mask2] = replace[1, np.searchsorted(replace[0, :], indices[1][mask2])]
            # Keep only data for the specified category
            pairs = self.find_pairs_by_separation(category=category, mappable_only=True)
            indices_to_keep = []
            for pair in pairs:
                pair_i = np.where(indices[0] == pair[0])
                pair_j = np.where(indices[1] == pair[1])
                overlap = np.intersect1d(pair_i, pair_j)
                if len(overlap) != 1:
                    # from IPython import embed
                    # embed()
                    raise ValueError('Something went wrong while computing overlaps.')
                indices_to_keep.append(overlap[0])
            self._specific_mapping[category] = (indices, indices_to_keep)
        else:
            indices, indices_to_keep = self._specific_mapping[category]
        mapped_predictions = predictions[indices]
        mapped_predictions = np.array(mapped_predictions[indices_to_keep])
        mapped_distances = self.distances[indices]
        mapped_distances = np.array(mapped_distances[indices_to_keep])
        return mapped_predictions, mapped_distances

    def score_auc(self, predictions, category='Any'):
        """
        Score AUC

        This function accepts a matrix of predictions and uses it to compute an overall AUROC when compared to the
        distances between residues computed for the PDB structure. It uses the cutoff defined when initializing the
        ContactScorer to determine the set of true positives and the mapping from query sequence to pdb determined
        when the fit function is performed.

        Args:
            predictions (np.array): An array of predictions for contacts between protein residues with size nxn where n
            is the length of the query sequence used when initializing the ContactScorer.
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
        if self.query_structure is None:
            print('AUC cannot be measured, because no PDB was provided.')
            return None, None, '-'
        if self.query_pdb_mapping is None:
            self.fit()
        mapped_predictions, mapped_distances = self._map_predictions_to_pdb(predictions, category=category)
        # AUC computation
        if (mapped_distances is not None) and (len(mapped_predictions) != len(mapped_distances)):
            raise ValueError("Lengths do not match between query sequence and the aligned pdb chain.")
        y_true1 = ((mapped_distances <= self.cutoff) * 1)
        fpr, tpr, _thresholds = roc_curve(y_true1, mapped_predictions, pos_label=1)
        auroc = auc(fpr, tpr)
        return tpr, fpr, auroc

    def plot_auc(self, auc_data, title=None, file_name=None, output_dir=None):
        """
        Plot AUC

        This function plots and saves the AUROC.  The image will be stored in the eps format with dpi=1000 using a name
        specified by the ContactScorer query name, cutoff, clustering constant, and date.

        Args:
            auc_data (dictionary): AUC values stored in the ETMIPC class, used to identify the specific values for the
            specified clustering constant (clus).
            title (str): The title for the AUC plot.
            file_name (str): The file name under which to save this figure.
            output_dir (str): The full path to where the AUC plot image should be stored. If None (default) the plot
            will be stored in the current working directory.
        """
        # If there is no AUC data return without plotting
        if (auc_data[0] is None) and (auc_data[1] is None) and (auc_data[2] in {None, '-', 'NA'}):
            return
        if file_name is None:
            file_name = '{}_Cutoff{}A_roc.eps'.format(self.query, self.cutoff)
        if not file_name.endswith('.eps'):
            file_name = file_name + '.eps'
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
        plt.savefig(file_name, format='eps', dpi=1000, fontsize=8)
        plt.close()

    def score_precision(self, predictions, k=None, n=None, category='Any'):
        """
        Score Precision

        This method can be used to calculate the precision of the predictions. The intention is that this method be used
        to compute precision for the top L/k or top n residue pairs, where L is the length of the query sequence and k
        is a number less than or equal to L and n is a specific number of predictions to test. Predictions in the top
        L/k or n are given a label of 1 if they are >0 and are given a label of 0 otherwise. The true positive set is
        determined by taking the PDB measured distances for the top L/k or n residue pairs and setting them to 1 if they
        are <= the cutoff provided when initializing this ContactScorer, and 0 otherwise. Precision tests that the
        ranking of residues correctly predicts structural contacts, it is given by tp / (tp + fp) as implemented by
        sklearn.

        Args:
            predictions (np.array): An array of predictions for contacts between protein residues with size nxn where n
            is the length of the query sequence used when initializing the ContactScorer.
            k (int): This value should only be specified if n is not specified. This is the number that L, the length of
            the query sequence, will be divided by to give the number of predictions to test.
            n (int): This value should only be specified if k is not specified. This is the number of predictions to
            test.
            category (str): The sequence separation category to score, the options are as follows:
                 Neighbors - Residues 1 to 5 sequence positions apart.
                Short - Residues 6 to 12 sequences positions apart.
                Medium - Residues 13 to 24 sequences positions apart.
                Long - Residues more than 24 sequence positions apart.
                Any - Any/All pairs of residues.
        Returns:
            float: The precision value computed for the predictions provided.
        """
        if self.query_structure is None:
            print('Precision cannot be measured, because no PDB was provided.')
            return '-'
        mapped_predictions, mapped_distances = self._map_predictions_to_pdb(predictions, category=category)
        ranks = rankdata((np.zeros(mapped_predictions.shape) - mapped_predictions), method='dense')
        if (k is not None) and (n is not None):
            raise ValueError('Both k and n were set for score_precision which is not a valid option.')
        elif k:
            n = int(floor(self.query_alignment.seq_length / float(k)))
        elif n is None:
            n = mapped_distances.shape[0]
        else:
            pass
        ind = np.where(ranks <= n)
        top_predictions = mapped_predictions[ind]
        y_pred1 = ((top_predictions > 0.0) * 1)
        top_distances = mapped_distances[ind]
        y_true1 = ((top_distances <= self.cutoff) * 1)
        precision = precision_score(y_true1, y_pred1)
        return precision

    def score_clustering_of_contact_predictions(self, predictions, bias=True, file_path='./z_score.tsv',
                                                w2_ave_sub=None, processes=1):
        """
        Score Clustering Of Contact Predictions

        This method employs the _clustering_z_score method to score all pairs for which predictions are made. A z-score
        of '-' means that the pair did not map to the provided PDB while 'NA' means that the sigma computed for that
        pair was equal to 0. The residues belonging to each pair are added to a set in order of pair covariance score.
        That set of residues is evaluated for clustering z-score after each pair is added.

        Args:
            predictions (numpy.array): An array of predictions for contacts between protein residues with size nxn where
            n is the length of the query sequence used when initializing the ContactScorer.
            bias (int or bool): option to calculate z_scores with bias (True) or no bias (False). If bias is used a j-i
            factor accounting for the sequence separation of residues, as well as their distance, is added to the
            calculation.
            file_path (str): path where the z-scoring results should be written to.
            w2_ave_sub (dict): A dictionary of the precomputed scores for E[w^2] also returned by this function.
        Returns:
            pd.DataFrame: Table holding residue I of a pair, residue J of a pair, the covariance score for that pair,
            the clustering Z-Score, the w score, E[w], E[w^2], sigma, and the number of residues of interest up to that
            point.
            dict: The parts of E[w^2] which can be precalculated and reused for later computations (i.e. cases 1, 2, and
            3).
        """
        start = time()
        if self.query_structure is None:
            print('Z-Scores cannot be measured, because no PDB was provided.')
            return pd.DataFrame()
            # If data has already been computed load and return it without recomputing.
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, sep='\t', header=0, index_col=False)
            return df, None
        # Identify unmappable positions (do not appear in the PDB).
        unmappable_residues = set(range(predictions.shape[0])) - set(self.query_pdb_mapping)
        # Identify all uniqu positions in the prediction set (assuming they are square and symmetrical).
        indices = np.triu_indices(predictions.shape[0], k=1)
        # Set up dataframe for storing results.
        data = {'Res_i': [], 'Res_j': [], 'Covariance_Score': [], 'Z-Score': [], 'W': [], 'W_Ave': [], 'W2_Ave': [],
                'Sigma': [], 'Num_Residues': []}
        # Identify all of the pairs which include unmappable positions and set their Z-scores to the appropriate value.
        unmappable_x = np.isin(indices[0], list(unmappable_residues))
        unmappable_y = np.isin(indices[1], list(unmappable_residues))
        unmappable_all = unmappable_x | unmappable_y
        unmappable_indices = (indices[0][unmappable_all], indices[1][unmappable_all])
        unmappable_pairs = zip(unmappable_indices[0], unmappable_indices[1])
        # Identify all mappable positions and retrieve their scores
        mappable_all = ~ unmappable_all
        mappable_indices = (indices[0][mappable_all], indices[1][mappable_all])
        mappable_pairs = zip(mappable_indices[0], mappable_indices[1])
        scores = predictions[mappable_indices]
        # Sort the scores in preparation for Z-scoring
        sorted_scores, sorted_residues = zip(*sorted(zip(scores, mappable_pairs), reverse=True))
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
        # Compute the precomputable part (w2_ave_sub)
        if w2_ave_sub is None:
            w2_ave_sub = {'Case1': 0, 'Case2': 0, 'Case3': 0}
            pool = Pool(processes=processes, initializer=init_compute_w2_ave_sub,
                        initargs=(self.distances, bias, self.cutoff))
            res = pool.map_async(compute_w2_ave_sub, range(self.distances.shape[0]))
            pool.close()
            pool.join()
            for cases in res.get():
                for key in cases:
                    w2_ave_sub[key] += cases[key]
        # Compute all other Z-scores
        pool2 = Pool(processes=processes, initializer=init_clustering_z_score,
                     initargs=(bias, w2_ave_sub, self.query_structure, self.query_pdb_mapping, self.distances,
                              self.cutoff, self.query_alignment))
        res = pool2.map(clustering_z_score, to_score)
        pool2.close()
        pool2.join()
        # Store the values in the data frame
        sorted_mappable_indices = map(list, zip(*sorted_residues))
        data['Res_i'] += list(sorted_mappable_indices[0])
        data['Res_j'] += list(sorted_mappable_indices[1])
        data['Covariance_Score'] += list(predictions[sorted_mappable_indices])
        for counter, curr_res in enumerate(res):
            curr_len = len(unique_sets[counter])
            data['Z-Score'] += [curr_res[0]] * curr_len
            data['W'] += [curr_res[1]] * curr_len
            data['W_Ave'] += [curr_res[2]] * curr_len
            data['W2_Ave'] += [curr_res[3]] * curr_len
            data['Sigma'] += [curr_res[4]] * curr_len
            data['Num_Residues'] += [len(to_score[counter])] * curr_len
        # Add values for unmappable positions
        data['Res_i'] += list(unmappable_indices[0])
        data['Res_j'] += list(unmappable_indices[1])
        data['Covariance_Score'] += list(predictions[unmappable_indices])
        len_unmappable = len(unmappable_pairs)
        data['Z-Score'] += ['-'] * len_unmappable
        data['W'] += [None] * len_unmappable
        data['W_Ave'] += [None] * len_unmappable
        data['W2_Ave'] += [None] * len_unmappable
        data['Sigma'] += [None] * len_unmappable
        data['Num_Residues'] += [None] * len_unmappable
        # Write out DataFrame
        df = pd.DataFrame(data)
        df[['Res_i', 'Res_j', 'Covariance_Score', 'Z-Score', 'W', 'W_Ave', 'Sigma', 'Num_Residues']].to_csv(
            path_or_buf=file_path, sep='\t', header=True, index=False)
        end = time()
        print('Compute SCW Z-Score took {} min'.format((end - start) / 60.0))
        return df, w2_ave_sub

    def write_out_clustering_results(self, today, raw_scores, coverage_scores, output_dir, file_name=None, prefix=None):
        """
        Write out clustering results

        This method writes the covariation scores to file along with the structural validation data if available.

        Args:
            today (date/str): Today's date.
            raw_scores (np.array): A matrix  the raw values from the covariance/contact prediction process.
            coverage_scores (dict): A matrix of the coverage_scores computed based on the raw_scores matrices.
            file_name (str): The filename under which to save the results.
            output_dir (str): The full path to where the output file should be stored. If None (default) the plot will
            be stored in the current working directory.
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
        file_dict = {key: [] for key in header}
        mapped_chain = self.best_chain
        for i in range(0, self.query_alignment.seq_length):
            for j in range(i + 1, self.query_alignment.seq_length):
                file_dict['(AA1)'].append('({})'.format(one_to_three(self.query_alignment.query_sequence[i])))
                file_dict['(AA2)'].append('({})'.format(one_to_three(self.query_alignment.query_sequence[j])))
                file_dict['Raw_Score'].append(round(raw_scores[i, j], 4))
                if coverage_scores is not None:
                    file_dict['Coverage_Score'].append(round(coverage_scores[i, j], 4))
                else:
                    file_dict['Coverage_Score'].append(np.nan)
                if (self.query_structure is None) and (self.query_pdb_mapping is None):
                    file_dict['Pos1'].append(i + 1)
                    file_dict['Pos2'].append(j + 1)
                    r = '-'
                    dist = '-'
                else:
                    if (i in self.query_pdb_mapping) or (j in self.query_pdb_mapping):
                        if i in self.query_pdb_mapping:
                            mapped1 = self.query_pdb_mapping[i]
                            file_dict['Pos1'].append(self.query_structure.pdb_residue_list[mapped_chain][mapped1])
                        else:
                            file_dict['Pos1'].append('-')
                        if j in self.query_pdb_mapping:
                            mapped2 = self.query_pdb_mapping[j]
                            file_dict['Pos2'].append(self.query_structure.pdb_residue_list[mapped_chain][mapped2])
                        else:
                            file_dict['Pos2'].append('-')
                        if (i in self.query_pdb_mapping) and (j in self.query_pdb_mapping):
                            dist = round(self.distances[mapped1, mapped2], 4)
                        else:
                            dist = float('NaN')
                    else:
                        file_dict['Pos1'].append('-')
                        file_dict['Pos2'].append('-')
                        dist = float('NaN')
                    if dist <= self.cutoff:
                        r = 1
                    elif np.isnan(dist):
                        r = '-'
                    else:
                        r = 0
                file_dict['Residue_Dist'].append(dist)
                file_dict['Within_Threshold'].append(r)
        df = pd.DataFrame(file_dict)
        df.to_csv(file_name, sep='\t', header=True, index=False, columns=header)
        end = time()
        print('Writing the contact prediction scores and structural validation data to file took {} min'.format(
            (end - start) / 60.0))

    def evaluate_predictor(self, predictor, verbosity, out_dir, dist='Any', biased_w2_ave=None,
                           unbiased_w2_ave=None, today=None, processes=1):
        """
        Evaluate Predictor

        Args:
            predictor (ETMIPC/ETMIPWrapper/DCAWrapper): A predictor which has already calculated its covariance scores.
            verbosity (int): What level of output to produce.
                1. Writes scores for all tested clustering constants.
                2. Tests the clustering Z-score of the predictions and writes them to file as well as plotting Z-Scores
                against residue count.
                3. Tests the AUROC of contact prediction at different levels of sequence separation and plots the
                resulting curves to file.
                4. Tests the precision of  contact prediction at different levels of sequence separation and list
                lengths (L, L/2 ... L/10).
                5. Produces heatmaps and surface plots of scores.
                In all cases a file is written out with the final evaluation of the scores, if no PDB is provided, this
                means only times will be recorded.'
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
        Returns:
            pandas.DataFrame: A DataFrame containing the specified amount of information (see verbosity) for the raw
            scores provided when evaluating this predictor. Possible column headings include: 'Time',
            'Sequence_Separation', 'Distance', 'AUROC', 'Precision (L)', 'Precision (L/2)', 'Precision (L/3)',
            'Precision (L/4)', 'Precision (L/5)', 'Precision (L/6)', 'Precision (L/7)', 'Precision (L/8)',
            'Precision (L/9)', and 'Precision (L/10)'.
            dict: A dictionary of the precomputed scores for E[w^2] for biased z-score computation.
            dict: A dictionary of the precomputed scores for E[w^2] for unbiased z-score computation.
        """
        if today is None:
            today = str(datetime.date.today())
        score_fn = os.path.join(out_dir, 'Score_Evaluation_Dist-{}.txt'.format(dist))
        # If the evaluation has already been performed load the data and return it
        if os.path.isfile(score_fn):
            score_df = pd.read_csv(score_fn, sep='\t', header=0, index_col=False)
            return score_df, None, None
        score_stats = None
        columns = ['Time', 'Sequence_Separation', 'Distance', 'AUROC', 'Precision (L)', 'Precision (L/2)',
                   'Precision (L/3)', 'Precision (L/4)', 'Precision (L/5)', 'Precision (L/6)', 'Precision (L/7)',
                   'Precision (L/8)', 'Precision (L/9)', 'Precision (L/10)', 'Max Biased Z-Score',
                   'Max Unbiased Z-Score']
        if isinstance(predictor.scores, dict):
            columns = ['K'] + columns
            for c in predictor.scores:
                c_out_dir = os.path.join(out_dir, str(c))
                if not os.path.isdir(c_out_dir):
                    os.mkdir(c_out_dir)
                score_stats, b_w2_ave, u_w2_ave = self.evaluate_predictions(
                    scores=predictor.get_scores(branch=c), verbosity=verbosity, out_dir=c_out_dir, dist=dist,
                    file_prefix='Scores_K-{}_'.format(c), stats=score_stats, biased_w2_ave=biased_w2_ave,
                    unbiased_w2_ave=unbiased_w2_ave, today=today, processes=processes)
                if (biased_w2_ave is None) and (b_w2_ave is not None):
                    biased_w2_ave = b_w2_ave
                if (unbiased_w2_ave is None) and (u_w2_ave is not None):
                    unbiased_w2_ave = u_w2_ave
                if score_stats != {}:
                    if 'K' not in score_stats:
                        score_stats['K'] = []
                        score_stats['Time'] = []
                    diff_len = len(score_stats['AUROC']) - len(score_stats['K'])
                    c_array = [c] * diff_len
                    time_array = [predictor.time[c]] * diff_len
                    score_stats['K'] += c_array
                    score_stats['Time'] += time_array
        else:
            score_stats, b_w2_ave, u_w2_ave = self.evaluate_predictions(
                scores=predictor.scores, verbosity=verbosity, out_dir=out_dir, dist=dist, file_prefix='Scores_',
                stats=score_stats, biased_w2_ave=biased_w2_ave, unbiased_w2_ave=unbiased_w2_ave, today=today,
                processes=processes)
            if (biased_w2_ave is None) and (b_w2_ave is not None):
                biased_w2_ave = b_w2_ave
            if (unbiased_w2_ave is None) and (u_w2_ave is not None):
                unbiased_w2_ave = u_w2_ave
            if score_stats != {}:
                if 'Time' not in score_stats:
                    score_stats['Time'] = []
                diff_len = len(score_stats['AUROC']) - len(score_stats['Time'])
                time_array = [predictor.time] * diff_len
                score_stats['Time'] += time_array
        if score_stats == {}:
            score_df = None
        else:
            score_df = pd.DataFrame(score_stats)
            score_df.to_csv(path_or_buf=score_fn, columns=[x for x in columns if x in score_stats], sep='\t',
                            header=True, index=False)
        return score_df, biased_w2_ave, unbiased_w2_ave

    def evaluate_predictions(self, verbosity, out_dir, scores, coverages=None, dist='Any', file_prefix='', stats=None,
                             biased_w2_ave=None, unbiased_w2_ave=None, today=None, processes=1):
        """
        Evaluate Predictions

        This function evaluates a matrix of covariance predictions to the specified verbosity level.

        Args:
            scores (np.array): The predicted scores for pairs of residues in a sequence alignment.
            coverages (np.array): The coverage adjusted scores for predictions on pairs of residues in a sequence
            alignment.
            verbosity (int): What level of output to produce.
                1. Writes scores for all tested clustering constants.
                2. Tests the clustering Z-score of the predictions and writes them to file as well as plotting Z-Scores
                against residue count.
                3. Tests the AUROC of contact prediction at different levels of sequence separation and plots the
                resulting curves to file.
                4. Tests the precision of  contact prediction at different levels of sequence separation and list
                lengths (L, L/2 ... L/10).
                5. Produces heatmaps and surface plots of scores.
                In all cases a file is written out with the final evaluation of the scores, if no PDB is provided, this
                means only times will be recorded.'
            out_dir (str/path): The path at which to save
            dist (str): Which type of distance computation to use to determine if residues are in contact, choices are:
                Any - Measures the minimum distance between two residues considering all of their atoms.
                CB - Measures the distance between two residues using their Beta Carbons as the measuring point.
                CA - Measures the distance between two residues using their Alpha Carbons as the measuring point.
            file_prefix (str): string to prepend before filenames.
            stats (dict): A dictionary of previously computed statistics and scores from a predictor.
            biased_w2_ave (dict): A dictionary of the precomputed scores for E[w^2] for biased z-score computation also
            returned by this function.
            unbiased_w2_ave (dict): A dictionary of the precomputed scores for E[w^2] for unbaised z-score
            computation also returned by this function.
        Returns:
            dict. The stats computed for this matrix of scores, if a dictionary of stats was passed in the current stats
            are added to the previous ones.
            dict. A dictionary of the precomputed scores for E[w^2] for biased z-score computation.
            dict. A dictionary of the precomputed scores for E[w^2] for unbaised z-score computation.
        """
        if stats is None:
            stats = {}
        if today is None:
            today = str(datetime.date.today())
        self.fit()
        self.measure_distance(method=dist)
        # Verbosity 1
        self.write_out_clustering_results(today=today, raw_scores=scores, coverage_scores=coverages,
                                          output_dir=out_dir, prefix=file_prefix)
        if verbosity >= 2:
            # Score Prediction Clustering
            z_score_fn = os.path.join(out_dir, file_prefix + 'Dist-{}_{}_ZScores.tsv')
            z_score_plot_fn = os.path.join(out_dir, file_prefix + 'Dist-{}_{}_ZScores.eps')
            z_score_biased, b_w2_ave = self.score_clustering_of_contact_predictions(
                scores, bias=True, file_path=z_score_fn.format(dist, 'Biased'), w2_ave_sub=biased_w2_ave,
                processes=processes)
            if (biased_w2_ave is None) and (b_w2_ave is not None):
                biased_w2_ave = b_w2_ave
            plot_z_scores(z_score_biased, z_score_plot_fn.format(dist, 'Biased'))
            stats['Max Biased Z-Score'] = np.max(z_score_biased.loc[z_score_biased['Z-Score'] != 'NA', 'Z-Score'])
            z_score_unbiased, u_w2_ave = self.score_clustering_of_contact_predictions(
                scores, bias=False, file_path=z_score_fn.format(dist, 'Unbiased'), w2_ave_sub=unbiased_w2_ave,
                processes=processes)
            if (unbiased_w2_ave is None) and (u_w2_ave is not None):
                unbiased_w2_ave = u_w2_ave
            plot_z_scores(z_score_unbiased, z_score_plot_fn.format(dist, 'Unbiased'))
            stats['Max Unbiased Z-Score'] = np.max(z_score_unbiased.loc[z_score_unbiased['Z-Score'] != 'NA', 'Z-Score'])
        # Evaluating scores
        if verbosity >= 3:
            if 'AUROC' not in stats:
                stats['AUROC'] = []
                stats['Distance'] = []
                stats['Sequence_Separation'] = []
            for separation in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
                # AUC Evaluation
                auc_roc = self.score_auc(scores, category=separation)
                self.plot_auc(auc_data=auc_roc, title='AUROC Evaluation', output_dir=out_dir,
                              file_name=file_prefix + 'AUROC_Evaluation_Dist-{}_Separation-{}'.format(dist, separation))
                stats['AUROC'].append(auc_roc[2])
                stats['Distance'].append(dist)
                stats['Sequence_Separation'].append(separation)
                # Precision Evaluation
                if verbosity >= 4:
                    for k in range(1, 11):
                        if k == 1:
                            precision_label = 'Precision (L)'
                        else:
                            precision_label = 'Precision (L/{})'.format(k)
                        if precision_label not in stats:
                            if len(stats['AUROC']) > 1:
                                stats[precision_label] = ['-'] * (len(stats['AUROC']) - 1)
                            else:
                                stats[precision_label] = []
                        precision = self.score_precision(predictions=scores, k=k, category=separation)
                        stats[precision_label].append(precision)
        if verbosity >= 5:
            heatmap_plot(name=file_prefix.replace('_', ' ') + 'Dist-{} Heatmap'.format(dist), data_mat=scores,
                         output_dir=out_dir)
            surface_plot(name=file_prefix.replace('_', ' ') + 'Dist-{} Surface'.format(dist), data_mat=scores,
                         output_dir=out_dir)
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

    This method accepts a DataFrame residue positions and z-scores produced by score_clustering_of_contact_predictions
    to plot a scatter plot.

    Args:
        df (pd.DataFrame): DataFrame containing at least the 'Num_Residues' and 'Z-Score' columns produced after
        running the score_clustering_of_contact_predictions method.
        file_path (str): Path at which to save the plot produced by this call.
    """
    # If there is no data to plot return
    if df.empty:
        return
    if file_path is None:
        file_path = './zscore_plot.eps'
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
        data_mat (np.array): A matrix of scores. This input should either be the score or coveragematrices from a
        predictor like the ETMIPC/ETMIPWrapper/DCAWrapper classes.
        output_dir (str): The full path to where the heatmap plot image should be stored. If None (default) the plot
        will be stored in the current working directory.
    """
    image_name = name.replace(' ', '_') + '.eps'
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
    image_name = name.replace(' ', '_') + '.eps'
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


def init_compute_w2_ave_sub(dists, bias_bool, custom_cutoff):
    """
    Init Compute w^2 Average Sub

    Method initializes shared values used by all processes working on parts of the w^2 average sub-problems.

    Args:
        dists (np.array): The array of distances between residues in the tested structure.
        bias_bool (int or bool): Option to calculate z_scores with bias (True) or no bias (False). If bias is used a j-i
        factor accounting for the sequence separation of residues, as well as their distance, is added to the
        calculation.
        custom_cutoff (float): Separation at which to consider residues clustered. In previous publications a value of 4
        Angstroms has been used.
    """
    global distances
    distances = dists
    global bias
    bias = bias_bool
    global cutoff
    cutoff = custom_cutoff


def compute_w2_ave_sub(res_i):
    """
    Compute W^2 Average Sub

    Solves the cases of the w^2 average sub-problem for a specific position in the protein. These can be combined across
    all positions to get the complete solution (after the pool of workers has finished).

    Args:
        res_i (int): The position in the protein for which to compute the parts of the w^2 average sub-problems.
    Returns:
        dict: The parts of E[w^2] which can be precalculated and reused for later computations (i.e. cases 1, 2, and
        3).
    """
    cases = {'Case1': 0, 'Case2': 0, 'Case3': 0}
    for res_j in range(res_i + 1, distances.shape[1]):
        if distances[res_i][res_j] >= cutoff:
            continue
        if bias:
            s_ij = res_j - res_i
        else:
            s_ij = 1
        for res_x in range(distances.shape[0]):
            for res_y in range(res_x + 1, distances.shape[1]):
                if distances[res_x][res_y] >= cutoff:
                    continue
                if bias:
                    s_xy = (res_y - res_x)
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


def init_clustering_z_score(bias_bool, w2_ave_sub_dict, curr_pdb, map_to_structure, residue_dists, dist_threshold,
                            seq_aln):
    """
    Init Clustering Z-Score

    This method initializes a set of processes in a multiprocessing pool so that they can compute the clustering Z-Score
    with minimal data duplication.

    Args:
        bias_bool (int or bool): Option to calculate z_scores with bias (True) or no bias (False). If bias is used a j-i
        factor accounting for the sequence separation of residues, as well as their distance, is added to the
        calculation.
        w2_ave_sub_dict (dict): A dictionary of the precomputed scores for E[w^2] also returned by this function.
        curr_pdb (PDBReference): Instance representing the PDB from which the structural data for this analysis comes.
        map_to_structure (dict): A mapping from the index of the query sequence to the index of the pdb chain's
        sequence for those positions which match according to a pairwise global alignment.
        residue_dists (np.array): The distances between residues, used for determining those which are in contact in
        order to assess predictions.
        dist_threshold (float): Value to use as a distance cutoff for contact prediction.
        seq_aln (SeqAlignment): SeqAlignment object being evaluated in this contact scoring prediction task.
    """
    global bias
    bias = bias_bool
    global w2_ave_sub
    w2_ave_sub = w2_ave_sub_dict
    global query_structure
    query_structure = curr_pdb
    global query_pdb_mapping
    query_pdb_mapping = map_to_structure
    global distances
    distances = residue_dists
    global cutoff
    cutoff = dist_threshold
    global query_alignment
    query_alignment = seq_aln


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
    # start = time()
    if query_structure is None:
        print('Z-Score cannot be measured, because no PDB was provided.')
        return '-', None, None, None, None, None
    # Check that there is a valid bias values
    if bias is not True and bias is not False:
        raise ValueError('Bias term may be True or False, but {} was provided'.format(bias))
    # Make sure a query_pdb_mapping exists
    if query_pdb_mapping is None:
        # self.fit()
        raise ValueError('ContactScorer instance must be fit before attempting to perform structural clustering'
                         'weighted z-scoring.')
    # Make sure all residues in res_list are mapped to the PDB structure in use
    if not all(res in query_pdb_mapping for res in res_list):
        print('At least one residue of interest is not present in the PDB provided')
        print(', '.join([str(x) for x in res_list]))
        print(', '.join([str(x) for x in query_pdb_mapping.keys()]))
        return '-', None, None, None, None, None
    positions = range(distances.shape[0])
    a = distances < cutoff
    a[positions, positions] = 0
    s_i = np.in1d(positions, res_list)
    s_ij = np.outer(s_i, s_i)
    s_ij[positions, positions] = 0
    if bias:
        bias_ij = np.subtract.outer(positions, positions)
        bias_ij[np.triu_indices(distances.shape[0], 1)] *= -1
    else:
        bias_ij = np.ones(s_ij.shape)
    w = np.sum(np.tril(a * s_ij * bias_ij))
    # Calculate w, <w>_S, and <w^2>_S.
    # Use expressions (3),(4),(5),(6) in Reference.
    m = len(res_list)
    l = query_alignment.seq_length
    pi1 = m * (m - 1.0) / (l * (l - 1.0))
    pi2 = pi1 * (m - 2.0) / (l - 2.0)
    pi3 = pi2 * (m - 3.0) / (l - 3.0)
    w_ave = np.sum(np.tril(a * bias_ij)) * pi1
    w2_ave = (pi1 * w2_ave_sub['Case1']) + (pi2 * w2_ave_sub['Case2']) + (pi3 * w2_ave_sub['Case3'])
    sigma = math.sqrt(w2_ave - w_ave * w_ave)
    # Response to Bioinformatics reviewer 08/24/10
    if sigma == 0:
        return 'NA', w, w_ave, w2_ave, sigma, w2_ave_sub
    z_score = (w - w_ave) / sigma
    # end = time()
    # print('Clustering Z-Score took {} min to compute'.format((end - start) / 60.0))
    return z_score, w, w_ave, w2_ave, sigma