"""
Created on Sep 1, 2018

@author: dmkonecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from math import floor
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.PDB.Polypeptide import one_to_three
from sklearn.metrics import auc, roc_curve, precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from seaborn import heatmap


class ContactScorer(object):
    """
    ContactScorer

    This class is meant to abstract the process of scoring a set of contact predictions away from the actual method
    (at the moment it is included in the ETMIPC class).  This is beind done for two main reasons. First, contact
    predictions are being made with several different methods and so their scoring should be performed consistently by
    an other object or function. Second, there are many ways to score contact predictions and at the moment only
    overall AUROC is being used. This class will support several other scoring methods such as the Precision at L/K
    found commonly in the literature as well as methods internal to the lab like the clustering Z-score derived from
    prior ET work.
    """

    def __init__(self, seq_alignment, pdb_reference, cutoff):
        """
        Init

        This function overwrite the default __init__ function. It accepts a SeqAlignment and PDBReference object which
        it will use to map between an alignment and a strucutre, compute distances, and ultimately score predictions
        made on contacts within a structure.

        Args:
            seq_alignment (SupportingClasses.SeqAlignment): The object containing the alignment of interest.
            pdb_reference (SupportingClasses.PDBReference): The object containing the PDB structure of interest.
            cutoff (int or float): The distance between two residues at which a true contact is said to be occuring.
        """
        self.query_alignment = seq_alignment
        self.query_structure = pdb_reference
        self.cutoff = cutoff
        self.best_chain = None
        self.query_pdb_mapping = None
        self.distances = None

    def __str__(self):
        """
        Str

        Method over writing the default __str__ method, giving a simple summary of the data held by the ContactScorer.

        Returns:
            str. Simple string summarizing the contents of the ContactScorer.

        Usage Example:
        >>> scorer = ContactScorer(p53_sequence, p53_structure)
        >>> str(scorer)
        """
        return 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            self.query_alignment.query_sequence, self.query_structure.seq, self.best_chain)

    def fit(self):
        """
        Fit

        This function maps sequence positions between the query sequence from the alignment and residues in PDB file.
        If there are multiple chains for the structure in the PDB, the one which matches the query sequence best
        (highest global alignment score) is used and recorded in the best_chain variable. This method updates the
        query_pdb_mapping class variable.
        """
        start = time()
        best_chain = None
        best_alignment = None
        for ch in self.query_structure.seq:
            curr_align = pairwise2.align.globalxs(self.query_alignment.query_sequence,
                                                  self.query_structure.seq[ch], -1, 0)
            if (best_alignment is None) or (best_alignment[0][2] < curr_align[0][2]):
                best_alignment = curr_align
                best_chain = ch
        print(format_alignment(*best_alignment[0]))
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
        end = time()
        print('Mapping query sequence and pdb took {} min'.format((end - start) / 60.0))
        self.best_chain = best_chain
        self.query_pdb_mapping = f_to_p_map

    @staticmethod
    def _get_all_coords(residue):
        """
        Get All Coords

        This method retrieves the 3D coordinates for all atoms in a residue.

        Args:
            residue (Bio.PDB.Residue): The residue for which to return coordinates.
        Returns:
            list. A list of lists where each sub lists contains the x, y, and z coordinates of a given atom in a
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

        This method retrieves the 3D coordinates for the Calpha atom of a residue. If the Calpha atom is not specified
        for the given residue _get_all_coords is called instead.

        Args:
            residue (Bio.PDB.Residue): The residue for which to return coordinates.
        Returns:
            list. A list of lists where the only sub list contains the x, y, and z coordinates of the Calpha atom.
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

        This method retrieves the 3D coordinates for the Cbeta atom of a residue. If the Cbeta atom is not specified
        for the given residue _get_c_alpha_coords is called instead.

        Args:
            residue (Bio.PDB.Residue): The residue for which to return coordinates.
        Returns:
            list. A list of lists where the only sub list contains the x, y, and z coordinates of the Cbeta atom.
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

        This method servers as a switch statement, its only purpose is to reduce code duplication when calling one of
        the three specific get (all, alpha, or beta) coord functions.

        Args:
            residue (Bio.PDB.Residue): The residue for which to extract the specified atom or atoms' coordinates.
            method (str): Which method of coordinate extraction to use, expected values are 'Any', 'CA' for alpha
            carbon, or 'CB' for beta carbon.
        Returns:
            list. A list of lists where each list is a single set of x, y, and z coordinates for an atom in the
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
        The first option is 'Any', this measures the distance between two residues as the shortest distance between any
        two atoms in the two residues structure. The second option is 'CA', this measures the distance as the space
        between the alpha carbons of two residues. If the alpha carbon for a given residue is not available in the PDB
        structure, then the distance between any atom of that residue and the alpha carbons of all other resiudes is
        measured. The third option is 'CB', this measures the distance as the space between the beta carbons of the two
        residues. If the beta carbon of a given residue is not available in the PDB structure the alpha carbon is used,
        or all atoms in the residue if the alpha carbon is also not annotated. This function updates the class variable
        distances.

        Args:
            method (str): Method by which to compute distances, see method description for more details on the options
            and what they mean.
            save_file (str or os.path): The path to a file where the computed distances can be stored, or may have been
            stored on a previous run.
    """
        start = time()
        if (save_file is not None) and os.path.exists(save_file):
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

    def _map_predictions_to_pdb(self, predictions):
        """
        Map Predictions To PDB

        This method accepts a set of predictions and uses the mapping between the query sequence and the best PDB chain
        to extract the comparable predictions and distances.

        Args:
            predictions (np.array): An array of predictions for contacts between protein residues with size nxn where n
            is the length of the query sequence used when initializing the ContactScorer.
        Returns:
            np.array. A set of predictions which can be scored because they map successfully to the PDB.
            np.array. A set of distances which can be used for scoring because they map successfully to the query
            sequence.
        """
        # Defining for which of the pairs of residues there are both cET-MIp  scores and distance measurements from
        # the PDB Structure.
        indices = np.triu_indices(predictions.shape[0], 1)
        mappable_pos = np.array(self.query_pdb_mapping.keys())
        x_mappable = np.in1d(indices[0], mappable_pos)
        y_mappable = np.in1d(indices[1], mappable_pos)
        final_mappable = x_mappable & y_mappable
        indices = (indices[0][final_mappable], indices[1][final_mappable])
        mapped_predictions = predictions[indices]
        # Mapping indices used for predictions so that they can be used to retrieve correct distances from PDB
        # distances matrix.
        keys = sorted(self.query_pdb_mapping.keys())
        values = [self.query_pdb_mapping[k] for k in keys]
        replace = np.array([keys, values])
        mask1 = np.in1d(indices[0], replace[0, :])
        indices[0][mask1] = replace[1, np.searchsorted(replace[0, :], indices[0][mask1])]
        mask2 = np.in1d(indices[1], replace[0, :])
        indices[1][mask2] = replace[1, np.searchsorted(replace[0, :], indices[1][mask2])]
        mapped_distances = self.distances[indices]
        return mapped_predictions, mapped_distances

    def score_auc(self, predictions):
        """
        Score AUC

        This function accepts a matrix of predictions and uses it to compute an overall AUROC when compared to the
        distances between residues computed for the PDB structure. It uses the cutoff defined when initializing the
        ContactScorer to determine the set of true positives and the mapping from query sequence to pdb determined
        when the fit function is performed.

        Args:
            predictions (np.array): An array of predictions for contacts between protein residues with size nxn where n
            is the length of the query sequence used when initializing the ContactScorer.
        Returns:
            np.array. The list of true positive rate values calculated when computing the roc curve.
            np.array. The list of false positive rate value calculated when computing the roc curve.
            float. The auroc determined for the roc curve.
        """
        if self.query_pdb_mapping is not None:
            mapped_predictions, mapped_distances = self._map_predictions_to_pdb(predictions)
            # AUC computation
            if (mapped_distances is not None) and (len(mapped_predictions) != len(mapped_distances)):
                raise ValueError("Lengths do not match between query sequence and the aligned pdb chain.")
            y_true1 = ((mapped_distances <= self.cutoff) * 1)
            fpr, tpr, _thresholds = roc_curve(y_true1, mapped_predictions, pos_label=1)
            auroc = auc(fpr, tpr)
        else:
            fpr = tpr = auroc = None
        return tpr, fpr, auroc

    def plot_auc(self, query_name, auc_data, title=None, file_name=None, output_dir=None):
        """
        Plot AUC

        This function plots and saves the AUCROC.  The image will be stored in
        the eps format with dpi=1000 using a name specified by the query name,
        cutoff, clustering constant, and date.

        Parameters:
        -----------
        query_name: str
            Name of the query protein
        clus: int
            Number of clusters created
        today: date
            The days date
        aucs : dictionary
            AUC values stored in the ETMIPC class, used to identify the specific
            values for the specified clustering constant (clus).
        output_dir : str
            The full path to where the AUC plot image should be stored. If None
            (default) the plot will be stored in the current working directory.
        """
        start = time()
        plt.plot(auc_data[0], auc_data[1], label='(AUC = {0:.2f})'.format(auc_data[2]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if title is None:
            title = 'Ability to predict positive contacts in {}'.format(query_name)
        plt.title(title)
        plt.legend(loc="lower right")
        if file_name is None:
            file_name = '{}_Cutoff{}A_roc.eps'.format(query_name, self.cutoff)
        print(file_name)
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        print(file_name)
        plt.savefig(file_name, format='eps', dpi=1000, fontsize=8)
        plt.close()
        end = time()
        print('Plotting the AUC plot took {} min'.format((end - start) / 60.0))

    def score_precision(self, predictions, k, n):
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
        Returns:
        """
        if (k is not None) and (n is not None):
            raise ValueError('Both k and n were set for score_precision which is not a valid option.')
        else:
            n = floor(self.query_alignment.seq_length / float(k))
        mapped_predictions, mapped_distances = self._map_predictions_to_pdb(predictions)
        sorted_predictions, sorted_distances = zip(sorted(zip(mapped_predictions, mapped_distances),
                                                          key=lambda pair: pair[0]))
        top_predictions = sorted_predictions[:n]
        y_pred1 = ((top_predictions > 0.0) * 1)
        top_distances = sorted_distances[:n]
        y_true1 = ((top_distances <= self.cutoff) * 1)
        precision = precision_score(y_true1, y_pred1)
        return precision

    def clustering_z_score(self):
        """

        :return:
        """
        raise NotImplemented()

    def write_out_contact_scoring(self, today, c_raw_scores, c_coverage, mip_matrix=None, c_raw_sub_scores=None,
                                  c_integrated_scores=None, file_name=None, output_dir=None):
        """
        Write out clustering scoring results

        This method writes the results of covariation scoring to file.

        Parameters:
        today: date
            Todays date.
        mip_matrix : np.ndarray
            Matrix scoring the coupling between all positions in the query
            sequence, as computed over all sequences in the input alignment.
        c_raw_sub_scores : numpy.array
            The coupling scores for all positions in the query sequences at the specified clustering constant created by
            hierarchical clustering.
        c_raw_scores : numpy.array
            A matrix which represents the integration of coupling scores across all clusters defined at that clustering
            constant.
        c_integrated_scores : numpy.array
            This dictionary maps clustering constants to a matrix which combines
            the scores from the whole_mip_matrix, all lower clustering constants,
            and this clustering constant.
        c_coverage : numpy.array
            This dictionary maps clustering constants to a matrix of normalized
            coupling scores between 0 and 100, computed from the
            summary_matrices.
        file_name : str
            The name with which to save the file. If None the following string template will be used:
            "{}_{}.all_scores.txt".format(today, self.query_alignment.query_id.split('_')[1])
        output_dir : str
            The full path to where the output file should be stored. If None
            (default) the plot will be stored in the current working directory.
        """
        start = time()
        header = ['Pos1', 'AA1', 'Pos2', 'AA2', 'OriginalScore']
        if c_raw_sub_scores is not None:
            header += ['Raw_Score_Sub_{}'.format(i) for i in map(str, range(1, c_raw_sub_scores.shape[0] + 1))]
        if c_integrated_scores is not None:
            header += ['Raw_Score', 'Integrated_Score', 'Coverage_Score']
        else:
            header += ['Raw_Score', 'Coverage_Score']
        file_dict = {key: [] for key in header}
        for i in range(0, self.query_alignment.seq_length):
            for j in range(i + 1, self.query_alignment.seq_length):
                res1 = i + 1
                res2 = j + 1
                file_dict['Pos1'].append(res1)
                file_dict['AA1'].append(one_to_three(self.query_alignment.query_sequence[i]))
                file_dict['Pos2'].append(res2)
                file_dict['AA2'].append(one_to_three(self.query_alignment.query_sequence[j]))
                file_dict['OriginalScore'].append(round(mip_matrix[i, j], 4))
                if c_raw_sub_scores is not None:
                    for c in range(c_raw_sub_scores.shape[0]):
                        file_dict['Raw_Score_Sub_{}'.format(c + 1)].append(round(c_raw_sub_scores[c, i, j], 4))
                file_dict['Raw_Score'].append(round(c_raw_scores[i, j], 4))
                if c_integrated_scores is not None:
                    file_dict['Integrated_Score'].append(round(c_integrated_scores[i, j], 4))
                file_dict['Coverage_Score'].append(round(c_coverage[i, j], 4))
        if file_name is None:
            file_name = "{}_{}.all_scores.txt".format(today, self.query_alignment.query_id.split('_')[1])
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        df = pd.DataFrame(file_dict)
        df.to_csv(file_name, sep='\t', header=True, index=False, columns=header)
        end = time()
        print('Writing the contact prediction scores to file took {} min'.format((end - start) / 60.0))

    def write_out_clustering_results(self, today, q_name, raw_scores, coverage_scores, file_name, output_dir):
        """
        Write out clustering results

        This method writes the covariation scores to file along with the structural validation data if available.

        Parameters:
        today: date
            Todays date.
        q_name: str
            The name of the query protein
        cutoff : float
            The distance used for proximity cutoff in the PDB structure.
        clus: int
            The number of clusters created
        alignment: SeqAlignment
            The sequence alignment object associated with the ETMIPC instance
            calling this method.
        pdb: PDBReference
            Object representing the pdb structure used in the current
            analysis.  This object is passed in to enable access to the
            sortedPDBDist variable.
        raw_scores : dict
            A dictionary of the clustering constants mapped to a matrix of the raw
            values from the whole MIp matrix through all clustering constants <=
            clus. See ETMIPC class description.
        coverage_scores : dict
            A dictionary of the clustering constants mapped to a matrix of the
            coverage_scores values computed on the raw_scores matrices. See ETMIPC class
            description.
        output_dir : str
            The full path to where the output file should be stored. If None
            (default) the plot will be stored in the current working directory.
        """
        start = time()
        header = ['Pos1', '(AA1)', 'Pos2', '(AA2)', 'Raw_Score', 'Coverage_Score', 'Residue_Dist', 'Within_Threshold']
        file_dict = {key: [] for key in header}
        mapped_chain = self.best_chain
        for i in range(0, self.query_alignment.seq_length):
            for j in range(i + 1, self.query_alignment.seq_length):
                file_dict['(AA1)'].append('({})'.format(one_to_three(self.query_alignment.query_sequence[i])))
                file_dict['(AA2)'].append('({})'.format(one_to_three(self.query_alignment.query_sequence[j])))
                file_dict['Raw_Score'].append(round(raw_scores[i, j], 4))
                file_dict['Coverage_Score'].append(round(coverage_scores[i, j], 4))
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
        if file_name is None:
            file_name = "{}_{}.etmipCVG.clustered.txt".format(today, q_name)
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        df = pd.DataFrame(file_dict)
        df.to_csv(file_name, sep='\t', header=True, index=False, columns=header)
        end = time()
        print('Writing the contact prediction scores and structural validation data to file took {} min'.format(
            (end - start) / 60.0))


def heatmap_plot(name, data_mat, output_dir=None):
    """
    Heatmap Plot

    This method creates a heatmap using the Seaborn plotting package. The
    data used can come from the summary_matrices or coverage data.

    Parameters:
    -----------
    name : str
        Name used as the title of the plot and the filename for the saved
        figure.
    rel_data : dict
        A dictionary of integers (k) mapped to matrices (scores). This input
        should either be the coverage or summary_matrices from the ETMIPC class.
    cluster : int
        The clustering constant for which to create a heatmap.
    output_dir : str
        The full path to where the heatmap plot image should be stored. If None
        (default) the plot will be stored in the current working directory.
    """
    start = time()
    dm_max = np.max(data_mat)
    dm_min = np.min(data_mat)
    plot_max = max([dm_max, abs(dm_min)])
    heatmap(data=data_mat, cmap='jet', center=0.0, vmin=-1 * plot_max,
            vmax=plot_max, cbar=True, square=True)
    plt.title(name)
    image_name = name.replace(' ', '_') + '.pdf'
    if output_dir:
        image_name = os.path.join(output_dir, image_name)
    plt.savefig(image_name)
    plt.clf()
    end = time()
    print('Plotting ETMIp-C heatmap took {} min'.format((end - start) / 60.0))


def surface_plot(name, data_mat, output_dir=None):
    """
    Surface Plot

    This method creates a surface plot using the matplotlib plotting
    package. The data used can come from the summary_matrices or coverage
    data.

    Parameters:
    -----------
    name : str
        Name used as the title of the plot and the filename for the saved
        figure.
    rel_data : dict
        A dictionary of integers (k) mapped to matrices (scores). This input
        should either be the coverage or summary_matrices from the ETMIPC class.
    cluster : int
        The clustering constant for which to create a heatmap.
    output_dir : str
        The full path to where the AUC plot image should be stored. If None
        (default) the plot will be stored in the current working directory.
    """
    start = time()
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
    image_name = name.replace(' ', '_') + '.pdf'
    if output_dir:
        image_name = os.path.join(output_dir, image_name)
    plt.savefig(image_name)
    plt.clf()
    end = time()
    print('Plotting ETMIp-C surface plot took {} min'.format((end - start) / 60.0))