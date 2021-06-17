"""
Created on Sep 1, 2018

@author: dmkonecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from math import floor
from datetime import datetime
from multiprocessing import Pool
from scipy.stats import hypergeom
# This is used implicitly not explicitly to enable surface plotting
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import (auc, roc_curve, precision_recall_curve, precision_score, recall_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from seaborn import scatterplot, lineplot, heatmap, scatterplot
from SupportingClasses.Predictor import Predictor
from SupportingClasses.utils import compute_rank_and_coverage
from Evaluation.Scorer import Scorer, init_scw_z_score_selection, scw_z_score_selection
from Evaluation.SelectionClusterWeighting import SelectionClusterWeighting


class ContactScorer(Scorer):
    """
    ContactScorer

    This class is meant to abstract the process of scoring a set of contact predictions away from the actual method
    (previously it was included in the prediction class).  This is being done for two main reasons. First, contact
    predictions are being made with several different methods and so their scoring should be performed consistently by
    another object or function. Second, this class will support several scoring methods such as the Precision at L/K
    found commonly in the literature as well as methods internal to the lab like the clustering Z-score derived from
    prior ET work.

    Attributes:
        cutoff (float): Value to use as a distance cutoff for contact prediction.
        query_pdb_mapper (SequencePDBMap): A mapping from the index of the query sequence to the index of the
        pdb chain's sequence for those positions which match according to a pairwise global alignment and vice versa.
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
        super().__init__(query, seq_alignment, pdb_reference, chain)
        self.cutoff = cutoff
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
        >>> scorer.fit
        >>> str(scorer)
        """
        if not self.query_pdb_mapper.is_aligned():
            raise ValueError('Scorer not yet fitted.')
        return f'Query Sequence of Length: {self.query_pdb_mapper.seq_aln.seq_length}\n' \
               f'PDB with {len(self.query_pdb_mapper.pdb_ref.chains)} Chains\n' \
               f'Best Sequence Match to Chain: {self.query_pdb_mapper.best_chain}'

    def fit(self):
        """
        Fit

        This function maps sequence positions between the query sequence from the alignment and residues in PDB file.
        If there are multiple chains for the structure in the PDB, the one which matches the query sequence best
        (highest global alignment score) is used and recorded in the seq_pdb_mapper attribute. This method updates the
        query_pdb_mapper and data class attributes.
        """
        super().fit()
        if (self.data is None) or (not self.data.columns.isin(
                ['Seq Pos 1', 'Seq AA 1', 'Seq Pos 2', 'Seq AA 2', 'Seq Separation', 'Seq Separation Category',
                 'Struct Pos 1', 'Struct AA 1', 'Struct Pos 2', 'Struct AA 2']).all()):
            start = time()
            data_dict = {'Seq Pos 1': [], 'Seq AA 1': [], 'Seq Pos 2': [], 'Seq AA 2': [], 'Struct Pos 1': [],
                         'Struct AA 1': [], 'Struct Pos 2': [], 'Struct AA 2': []}
            for pos1 in range(self.query_pdb_mapper.seq_aln.seq_length):
                char1 = self.query_pdb_mapper.seq_aln.query_sequence[pos1]
                pos_other = list(range(pos1 + 1, self.query_pdb_mapper.seq_aln.seq_length))
                char_other = list(self.query_pdb_mapper.seq_aln.query_sequence[pos1+1:])
                struct_pos1, struct_char1 = self.query_pdb_mapper.map_seq_position_to_pdb_res(pos1)
                if struct_pos1 is None:
                    struct_pos1, struct_char1 = '-', '-'
                struct_pos_other = []
                struct_char_other = []
                for pos2 in pos_other:
                    struct_pos2, struct_char2 = self.query_pdb_mapper.map_seq_position_to_pdb_res(pos2)
                    if struct_pos2 is None:
                        struct_pos2, struct_char2 = '-', '-'
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
        super().measure_distance(method=method, save_file=save_file)
        indices = self.data.loc[(self.data['Struct Pos 1'] != '-') & (self.data['Struct Pos 2'] != '-')].index
        self.data.loc[indices, 'Distance'] = self.data.loc[indices].apply(
            lambda x: self.distances[self.query_pdb_mapper.query_pdb_mapping[x['Seq Pos 1']],
                                     self.query_pdb_mapper.query_pdb_mapping[x['Seq Pos 2']]], axis=1)
        self.data['Contact (within {}A cutoff)'.format(self.cutoff)] = self.data['Distance'].apply(
            lambda x: '-' if x == '-' else x <= self.cutoff)

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

    def map_predictions_to_pdb(self, ranks, predictions, coverages, threshold=0.5):
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
        indices = np.triu_indices(n=self.query_pdb_mapper.seq_aln.seq_length, k=1)
        self.data['Rank'] = ranks[indices]
        self.data['Score'] = predictions[indices]
        self.data['Coverage'] = coverages[indices]
        self.data['True Prediction'] = self.data['Coverage'].apply(lambda x: x <= threshold)

    def _identify_relevant_data(self, category='Any', n=None, k=None, coverage_cutoff=None):
        """
        Identify Relevant Data

        This method accepts parameters describing a subset of the data stored in the ContactScorer instance and subsets
        it. The order of operations is:
            1. Filter to only predicted pairs in the specified sequence separation category.
            2. Filter to only predicted pairs which map to the best_chain of the provided PDB structure.
            3. Filter to the top predictions as specified by n, k, or coverage_cutoff

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
            k (int): This value should only be specified if n and coverage_cutoff are not specified. This is the number
            that L, the length of the query sequence, will be divided by to give the number of predictions to test.
            n (int): This value should only be specified if k and coverage_cutoff are not specified. This is the number
            of predictions to test.
            coverage_cutoff (float): This value should only be specified if n and k are not specified. This number
            determines how many predictions will be tested by considering predictions up to the point that the specified
            percentage of residues in best_chain are covered. If predictions are tied their residues are added to this
            calculation together, so if many residues are added by considering the next group of predictions, and the
            number of unique residues exceeds the specified percentage, none of the predictions in that group will be
            added.
        Returns:
            pd.DataFrame: A set of predictions which can be scored because they are in the specified sequence separation
            category, map successfully to the PDB, and meet the criteria set by n, k, or coverage cutoff for top
            predictions.
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
        parameter_count = (k is not None) + (n is not None) + (coverage_cutoff is not None)
        if parameter_count > 1:
            raise ValueError('Only one parameter should be set, either: n, k, or coverage_cutoff.')
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
        if coverage_cutoff:
            assert isinstance(coverage_cutoff, float), 'coverage_cutoff must be a float!'
            mappable_res = sorted(self.query_pdb_mapper.query_pdb_mapping.keys())
            position_remapping = {res: i for i, res in enumerate(mappable_res)}
            top_sequence_ranks = np.ones(self.query_pdb_mapper.seq_aln.seq_length) * np.inf
            residues_visited = set()
            groups = final_df.groupby('Top Predictions')
            for curr_rank in sorted(groups.groups.keys()):
                curr_group = groups.get_group(curr_rank)
                curr_residues = set(curr_group['Seq Pos 1']).union(set(curr_group['Seq Pos 2']))
                new_positions = curr_residues - residues_visited
                top_sequence_ranks[list(new_positions)] = curr_rank
                residues_visited |= new_positions
            top_residue_ranks = top_sequence_ranks[mappable_res]
            assert len(top_residue_ranks) <= len(mappable_res)
            _, top_coverage = compute_rank_and_coverage(seq_length=len(mappable_res), pos_size=1, rank_type='min',
                                                        scores=top_residue_ranks)
            final_df['Pos 1 Coverage'] = final_df['Seq Pos 1'].apply(lambda x: top_coverage[position_remapping[x]])
            final_df['Pos 2 Coverage'] = final_df['Seq Pos 2'].apply(lambda x: top_coverage[position_remapping[x]])
            n = final_df.loc[(final_df['Pos 1 Coverage'] > coverage_cutoff) |
                             (final_df['Pos 2 Coverage'] > coverage_cutoff), 'Top Predictions'].min() - 1
            if np.isnan(n):
                n = final_df['Top Predictions'].max()
                if np.isnan(n):
                    n = 0
        elif k:
            n = int(floor(self.query_pdb_mapper.seq_aln.seq_length / float(k)))
        elif n is None:
            n = self.data.shape[0]
        else:
            pass
        ind = final_df['Top Predictions'] <= n
        final_df = final_df.loc[ind, :]
        return final_df

    def characterize_pair_residue_coverage(self, out_dir, fn=None):
        """
        Characterize Pair Residue Coverage

        This method iterates over pairs and characterizes how many unique residues have been observed at each pair
        ranking. This means that if many pairs are tied in ranking many residues may be added at once.

        Returns:
            pandas.DataFrame: A dataframe including the ranks and coverages of pairs, as well as the corresponding
            number of pairs and residues added at each rank and their cumulative sum at each rank, the coverage of
            residues added at each rank and the maximum residues coverage observed up to that rank.
            str: The path where the dataframe has been writen to file.
            str: The path to a figure plotting pair coverage along the x-axis against the maximum residue coverage
            observed up to that point along the y-axis.
        """
        if not pd.Series(['Rank', 'Score', 'Coverage']).isin(self.data.columns).all():
            raise ValueError('Ranks, Scores, and Coverage values must be provided through map_predictions_to_pdb,'
                             'before a specific evaluation can be made.')
        if fn is None:
            fn = 'Pair_vs_Residue_Coverage'
        tsv_path = os.path.join(out_dir, fn + '.tsv')
        plot_path = os.path.join(out_dir, fn + '.png')
        top_sequence_ranks = np.zeros(self.query_pdb_mapper.seq_aln.seq_length)
        rank_res = {}
        residues_visited = set()
        groups = self.data.groupby('Rank')
        for curr_rank in sorted(groups.groups.keys()):
            curr_group = groups.get_group(curr_rank)
            curr_residues = set(curr_group['Seq Pos 1']).union(set(curr_group['Seq Pos 2']))
            new_positions = curr_residues - residues_visited
            rank_res[curr_rank] = sorted(new_positions)
            top_sequence_ranks[list(new_positions)] = curr_rank
            residues_visited |= new_positions
        _, sequence_coverage = compute_rank_and_coverage(seq_length=self.query_pdb_mapper.seq_aln.seq_length,
                                                         pos_size=1, rank_type='min', scores=top_sequence_ranks)
        comparison_df = self.data[['Rank', 'Coverage']].drop_duplicates().sort_values(by='Rank')
        comparison_df.rename(columns={'Rank': 'Pair Rank', 'Coverage': 'Pair Coverage'}, inplace=True)
        comparison_df['Residues Added'] = comparison_df['Pair Rank'].apply(
            lambda x: ','.join([str(y) for y in rank_res[x]]))
        comparison_df['Num Residues Added'] = comparison_df['Pair Rank'].apply(lambda x: len(rank_res[x]))
        comparison_df['Num Pairs Added'] = comparison_df['Pair Rank'].apply(lambda x: (self.data['Rank'] == x).sum())
        comparison_df['Residue Coverage'] = '-'
        for ind in comparison_df.index:
            rank = comparison_df.loc[ind, 'Pair Rank']
            residues = rank_res[rank]
            ranks = np.unique(top_sequence_ranks[residues])
            assert len(ranks) <= 1
            coverages = np.unique(sequence_coverage[residues])
            assert len(coverages) <= 1
            try:
                comparison_df.loc[ind, 'Residue Coverage'] = coverages[0]
            except IndexError:
                comparison_df.loc[ind, 'Residue Coverage'] = np.nan
        comparison_df['Total Pairs Added'] = comparison_df['Num Pairs Added'].cumsum()
        comparison_df['Total Residues Added'] = comparison_df['Num Residues Added'].cumsum()
        comparison_df['Max Residue Coverage'] = comparison_df['Residue Coverage'].cummax()
        comparison_df['Max Residue Coverage'].fillna(method='ffill', inplace=True)
        comparison_df = comparison_df.astype({'Pair Rank': 'int32', 'Pair Coverage': 'float64',
                                              'Num Residues Added': 'float64', 'Num Pairs Added': 'float64',
                                              'Residue Coverage': 'float64', 'Total Pairs Added': 'float64',
                                              'Total Residues Added': 'float64', 'Max Residue Coverage': 'float64'})
        comparison_df.to_csv(tsv_path, sep='\t', header=True, index=False)
        plotting_df = pd.DataFrame({'Pair Coverage': [0], 'Max Residue Coverage': [0]}).append(
                                   comparison_df[['Pair Coverage', 'Max Residue Coverage']]).append(
            pd.DataFrame({'Pair Coverage': [1], 'Max Residue Coverage': [1]}))
        cov_comp_ax = scatterplot(x='Pair Coverage', y='Max Residue Coverage', color='k', markers='.', edgecolor="none",
                                  data=plotting_df)
        lineplot(x='Pair Coverage', y='Max Residue Coverage', color='k', markers=True, dashes=False, data=plotting_df,
                 ax=cov_comp_ax)
        cov_comp_ax.set_xlim(0.0, 1.0)
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cov_comp_ax.set_xticks(ticks)
        cov_comp_ax.set_xticklabels(ticks)
        cov_comp_ax.set_ylim(0.0, 1.0)
        cov_comp_ax.set_yticks(ticks)
        cov_comp_ax.set_yticklabels(ticks)
        cov_comp_ax.get_figure().savefig(plot_path, bbox_inches='tight', transparent=True, dpi=500)
        plt.close()
        return comparison_df, tsv_path, plot_path

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
            file_name = '{}_Cutoff{}A_roc.png'.format(self.query_pdb_mapper.query, self.cutoff)
        if not file_name.endswith('.png'):
            file_name = file_name + '.png'
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        # If the figure has already been plotted return
        if os.path.isfile(file_name):
            return
        plt.plot(auc_data[1], auc_data[0], label=f'(AUC = {auc_data[2]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if title is None:
            title = 'Ability to predict positive contacts in {}'.format(self.query_pdb_mapper.query)
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
            file_name = '{}_Cutoff{}A_auprc.png'.format(self.query_pdb_mapper.query, self.cutoff)
        if not file_name.endswith('.png'):
            file_name = file_name + '.png'
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        # If the figure has already been plotted return
        if os.path.isfile(file_name):
            return
        plt.plot(auprc_data[1], auprc_data[0], label=f'(AUC = {auprc_data[2]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if title is None:
            title = 'Ability to predict positive contacts in {}'.format(self.query_pdb_mapper.query)
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

    def score_clustering_of_contact_predictions(self, biased=True, file_path='./z_score.tsv', scw_scorer=None,
                                                processes=1):
        """
        Score Clustering Of Contact Predictions

        This method employs the SelectionClusterWeighting class and support functions in the Scorer class to score all
        pairs for which predictions are made. A Z-Score of '-' means that the pair did not map to the provided PDB while
        'NA' means that the sigma computed for that pair was equal to 0. The residues belonging to each pair are added
        to a set in order of pair covariance score. That set of residues is evaluated for selection cluster weighting
        z-score after each pair is added.

        Args:
            biased (int or bool): option to calculate z_scores with bias (True) or no bias (False). If bias is used a j-i
            factor accounting for the sequence separation of residues, as well as their distance, is added to the
            calculation.
            file_path (str): path where the z-scoring results should be written to.
            scw_scorer (SelectionClusterWeighting): A SelectionClusterWeighting instance with the w_ave and w2_ave
            background already computed. If None, a new SelectionClusterWeighting object will be initialized using the
            SequencePDBMap held by the current ContactScorer instance and the background values will be computed before
            scoring the ranked residues.
            processes (int): How many processes may be used in computing the clustering Z-scores.
        Returns:
            pd.DataFrame: Table holding residue I of a pair, residue J of a pair, the covariance score for that pair,
            the clustering Z-Score, the w score, E[w], E[w^2], sigma, and the number of residues of interest up to that
            point.
            SelectionClusterWeighting: The SelectionClusterWeighting object used in this computation, which stores the
            precomputable parts of E[w^2], which can reused for later computations (i.e. cases 1, 2, and 3).
            float: The area under the curve defined by the z-scores and the protein coverage.
        """
        start = time()
        if self.query_pdb_mapper.pdb_ref is None:
            print('Z-Scores cannot be measured, because no PDB was provided.')
            return pd.DataFrame(), None, None
            # If data has already been computed load and return it without recomputing.
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, sep='\t', header=0, index_col=False)
            df_sub = df.replace([None, '-', 'NA'], np.nan)
            df_sub.sort_values(by='Pair Coverage', ascending=True, inplace=True)
            df_sub = df_sub.dropna()
            df_sub.drop_duplicates(subset='Pair Coverage', keep='last', inplace=True)
            if len(df_sub['Residue Coverage']) == 0:
                au_scw_z_score_curve = None
            else:
                au_scw_z_score_curve = auc(df_sub['Residue Coverage'].astype(float).values,
                                           df_sub['Z-Score'].astype(float).values)
            return df, None, au_scw_z_score_curve
        # Set up dataframe for storing results.
        if scw_scorer is None:
            scw_scorer = SelectionClusterWeighting(seq_pdb_map=self.query_pdb_mapper, pdb_dists=self.distances,
                                                   biased=biased)
            # Compute the precomputable part (w_ave_pre, w2_ave_sub)
            scw_scorer.compute_background_w_and_w2_ave(processes=processes)
        else:
            assert scw_scorer.biased == biased, 'SelectionClusterWeighting scorer does not match the biased parameter.'
        data_df = self._identify_relevant_data(category='Any', coverage_cutoff=1.0).loc[
                  :, ['Seq Pos 1', 'Seq Pos 2', 'Score', 'Coverage', 'Pos 1 Coverage', 'Pos 2 Coverage']]
        data_df['Max Pos Coverage'] = data_df[['Pos 1 Coverage', 'Pos 2 Coverage']].max(axis=1)
        data_df['Cumulative Coverage'] = data_df['Max Pos Coverage'].cummax()
        data_df.drop(columns=['Pos 1 Coverage', 'Pos 2 Coverage', 'Max Pos Coverage'], inplace=True)
        data_df.rename(columns={'Seq Pos 1': 'Res_i', 'Seq Pos 2': 'Res_j', 'Score': 'Covariance Score',
                                'Coverage': 'Pair Coverage', 'Cumulative Coverage': 'Residue Coverage'}, inplace=True)
        data = {'Z-Score': [], 'W': [], 'W_Ave': [], 'W2_Ave': [], 'Sigma': [], 'Num_Residues': []}
        # Set up structures to track unique sets of residues
        residues_visited = set()
        unique_sets = {}
        to_score = []
        # Identify unique sets of residues to score
        groups = data_df.groupby('Pair Coverage')
        counter = 0
        for curr_cov in sorted(groups.groups.keys()):
            curr_group = groups.get_group(curr_cov)
            curr_residues = set(curr_group['Res_i']).union(set(curr_group['Res_j']))
            new_positions = curr_residues - residues_visited
            unique_sets[counter] = curr_group[['Res_i', 'Res_j']].apply(tuple, axis=1).tolist()
            counter += 1
            residues_visited |= new_positions
            to_score.append(sorted(residues_visited))
        # Compute all other Z-scores
        unique_res = []
        sel_pbar = tqdm(total=len(to_score), unit='selection')

        def retrieve_scw_z_score_res(return_tuple):
            """
            Retrieve SCW Z-Score Result

            This function serves to update the progress bar for each selection scored by the SCW Z-Score. It also
            stores only the portion of the returned result needed for the output of this method.

            Args:
                return_tuple (tuple): A tuple consisting of all of the returns from
                SelectionClusterWeighting.scw_z_score_selection.
            """
            unique_res.append((return_tuple[11], return_tuple[6], return_tuple[7], return_tuple[8], return_tuple[9],
                               return_tuple[10]))
            sel_pbar.update(1)
            sel_pbar.refresh()

        pool2 = Pool(processes=processes, initializer=init_scw_z_score_selection,
                     initargs=(scw_scorer, ))
        for selection in to_score:
            pool2.apply_async(scw_z_score_selection, (selection, ), callback=retrieve_scw_z_score_res)
        pool2.close()
        pool2.join()
        # Ensure the ordering of results is correct by sorting by the number of residues in each analysis.
        unique_res = sorted(unique_res)
        # Variables to track for computing the Area Under the SCW Z-Score curve
        x_coverage = []
        y_z_score = []
        span_start = 0
        for counter, curr_res in enumerate(unique_res):
            curr_len = len(unique_sets[counter])
            data['Num_Residues'] += [curr_res[0]] * curr_len
            data['Z-Score'] += [curr_res[1]] * curr_len
            data['W'] += [curr_res[2]] * curr_len
            data['W_Ave'] += [curr_res[3]] * curr_len
            data['W2_Ave'] += [curr_res[4]] * curr_len
            data['Sigma'] += [curr_res[5]] * curr_len
            if curr_res[1] not in {None, '-', 'NA'}:
                y_z_score.append(curr_res[1])
                x_coverage.append(data_df['Residue Coverage'].iloc[span_start: span_start+curr_len].max())
            span_start += curr_len
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
                                         ['Seq Pos 1', 'Seq Pos 2', 'Score', 'Coverage']].sort_values(by='Coverage')
        data_df_unmapped.rename(columns={'Seq Pos 1': 'Res_i', 'Seq Pos 2': 'Res_j', 'Score': 'Covariance Score',
                                         'Coverage': 'Pair Coverage'}, inplace=True)
        data_df_unmapped['Residue Coverage'] = '-'
        data_df_unmapped['Z-Score'] = '-'
        data_df_unmapped['W'] = None
        data_df_unmapped['W_Ave'] = None
        data_df_unmapped['W2_Ave'] = None
        data_df_unmapped['Sigma'] = None
        data_df_unmapped['Num_Residues'] = None
        # Combine the mappable and unmappable index dataframes.
        df = data_df.append(data_df_unmapped)
        # Write out DataFrame
        df[['Res_i', 'Res_j', 'Covariance Score', 'Pair Coverage', 'Residue Coverage', 'Z-Score', 'W', 'W_Ave', 'Sigma',
            'Num_Residues']].to_csv(
            path_or_buf=file_path, sep='\t', header=True, index=False)
        end = time()
        print('Compute SCW Z-Score took {} min'.format((end - start) / 60.0))
        return df, scw_scorer, au_scw_z_score_curve

    def write_out_covariation_and_structural_data(self, output_dir=None, file_name=None):
        """
        Write Out Clustering Results

        This method writes the covariation scores to file along with the structural validation data if available.

        Args:
            output_dir (str): The full path to where the output file should be stored. If None (default) the file will
            be stored in the current working directory.
            file_name (str): The filename under which to save the results.
        """
        start = time()
        if file_name is None:
            file_name = f"{datetime.now().strftime('%m_%d_%Y')}_{self.query_pdb_mapper.query}" \
                        ".Covariance_vs_Structure.tsv"
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        if os.path.isfile(file_name):
            end = time()
            print('Contact prediction scores and structural validation data already written {} min'.format(
                (end - start) / 60.0))
            return
        header = ['Pos1', '(AA1)', 'Pos2', '(AA2)', 'Raw_Score', 'Coverage_Score', 'PDB Pos1', '(PDB AA1)', 'PDB Pos2',
                  '(PDB AA2)', 'Residue_Dist', 'Within_Threshold']
        df = self.data[['Seq Pos 1', 'Seq AA 1', 'Seq Pos 2', 'Seq AA 2', 'Score', 'Coverage', 'Struct Pos 1',
                        'Struct AA 1', 'Struct Pos 2', 'Struct AA 2', 'Distance',
                        f'Contact (within {self.cutoff}A cutoff)']]
        df = df.rename(columns={'Seq Pos 1': 'Pos1', 'Seq AA 1': '(AA1)', 'Seq Pos 2': 'Pos2', 'Seq AA 2': '(AA2)',
                                'Score': 'Raw_Score', 'Coverage': 'Coverage_Score', 'Struct Pos 1': 'PDB Pos1',
                                'Struct AA 1': '(PDB AA1)', 'Struct Pos 2': 'PDB Pos2', 'Struct AA 2': '(PDB AA2)',
                                'Distance': 'Residue_Dist',
                                f'Contact (within {self.cutoff}A cutoff)': 'Within_Threshold'})
        df.to_csv(file_name, sep='\t', header=True, index=False, columns=header)
        end = time()
        print('Writing the contact prediction scores and structural validation data to file took {} min'.format(
            (end - start) / 60.0))

    def select_and_color_residues(self, out_dir, category='Any', n=None, k=None, residue_coverage=None, fn=None):
        """
        Select and Color Residues

        This method selects the correct subset of data and uses the PDBReference to create a pse file and text file of
        commands coloring the residues by their coverage score.

        Args:
            out_dir: The directory in which to save the pse file generated and the text file with the commands used to
            generate it.
            category (str/list): The category for which to return residue pairs. At the moment the following categories
            are supported:
                Neighbors - Residues 1 to 5 sequence positions apart.
                Short - Residues 6 to 12 sequences positions apart.
                Medium - Residues 13 to 24 sequences positions apart.
                Long - Residues more than 24 sequence positions apart.
                Any - Any/All pairs of residues.
            In order to return a combination of labels, a list may be provided which contains any of the strings from
            the above set of categories (e.g. ['Short', 'Medium', 'Long']).
            k (int): This value should only be specified if n and coverage_cutoff are not specified. This is the number
            that L, the length of the query sequence, will be divided by to give the number of predictions to test.
            n (int): This value should only be specified if k and coverage_cutoff are not specified. This is the number
            of predictions to test.
            residue_coverage (float): This value should only be specified if n and k are not specified. This number
            determines how many predictions will be tested by considering predictions up to the point that the specified
            percentage of residues in best_chain are covered. If predictions are tied their residues are added to this
            calculation together, so if many residues are added by considering the next group of predictions, and the
            number of unique residues exceeds the specified percentage, none of the predictions in that group will be
            added.
            fn (str):  A string specifying the filename for the pse and commands files being written out which will have
            the format:
                f{fn}.pse
                f{fn}_all_pymol_commands.txt
            If None a default will be used as specified in PDBReference color_structure().
        Returns:
            str: The path to the created pse file.
            str: The path to the text file containing all commands used to generate the pse file.
            list: The residues in the structure which have been colored by this method using the provided data.
        """
        relevant_data = self._identify_relevant_data(category=category, n=n, k=k, coverage_cutoff=residue_coverage)
        if residue_coverage is None:
            mappable_res = sorted(self.query_pdb_mapper.query_pdb_mapping.keys())
            top_sequence_ranks = np.ones(self.query_pdb_mapper.seq_aln.seq_length) * np.inf
            residues_visited = set()
            groups = relevant_data.groupby('Top Predictions')
            for curr_rank in sorted(groups.groups.keys()):
                curr_group = groups.get_group(curr_rank)
                curr_residues = set(curr_group['Seq Pos 1']).union(set(curr_group['Seq Pos 2']))
                new_positions = curr_residues - residues_visited
                top_sequence_ranks[list(new_positions)] = curr_rank
                residues_visited |= new_positions
            top_residue_ranks = top_sequence_ranks[mappable_res]
            assert len(top_residue_ranks) <= len(mappable_res)
            _, top_coverage = compute_rank_and_coverage(seq_length=len(mappable_res), pos_size=1, rank_type='min',
                                                        scores=top_residue_ranks)
            relevant_data['Pos 1 Coverage'] = relevant_data['Seq Pos 1'].apply(lambda x: top_coverage[x])
            relevant_data['Pos 2 Coverage'] = relevant_data['Seq Pos 2'].apply(lambda x: top_coverage[x])
        coloring_data = pd.concat([relevant_data[['Struct Pos 1', 'Pos 1 Coverage']].rename(
            columns={'Struct Pos 1': 'RESIDUE_Index', 'Pos 1 Coverage': 'Coverage'}),
            relevant_data[['Struct Pos 2', 'Pos 2 Coverage']].rename(
                columns={'Struct Pos 2': 'RESIDUE_Index', 'Pos 2 Coverage': 'Coverage'})], ignore_index=True)
        coloring_data = coloring_data.drop_duplicates()
        return self.query_pdb_mapper.pdb_ref.color_structure(
            chain_id=self.query_pdb_mapper.best_chain, data=coloring_data, data_type='Coverage', data_direction='min',
            color_map='ET', out_dir=out_dir, fn=fn)

    def select_and_display_pairs(self, out_dir, category='Any', n=None, k=None, residue_coverage=None, fn=None):
        """
        Select and Display Pairs

        This method selects the correct subset of data and uses the PDBReference to create a pse file and text file of
        commands coloring the residues by their coverage score and the connection between pairs by the the color of the
        lowest coverage residue.

        Args:
            out_dir: The directory in which to save the pse file generated and the text file with the commands used to
            generate it.
            category (str/list): The category for which to return residue pairs. At the moment the following categories
            are supported:
                Neighbors - Residues 1 to 5 sequence positions apart.
                Short - Residues 6 to 12 sequences positions apart.
                Medium - Residues 13 to 24 sequences positions apart.
                Long - Residues more than 24 sequence positions apart.
                Any - Any/All pairs of residues.
            In order to return a combination of labels, a list may be provided which contains any of the strings from
            the above set of categories (e.g. ['Short', 'Medium', 'Long']).
            k (int): This value should only be specified if n and coverage_cutoff are not specified. This is the number
            that L, the length of the query sequence, will be divided by to give the number of predictions to test.
            n (int): This value should only be specified if k and coverage_cutoff are not specified. This is the number
            of predictions to test.
            residue_coverage (float): This value should only be specified if n and k are not specified. This number
            determines how many predictions will be tested by considering predictions up to the point that the specified
            percentage of residues in best_chain are covered. If predictions are tied their residues are added to this
            calculation together, so if many residues are added by considering the next group of predictions, and the
            number of unique residues exceeds the specified percentage, none of the predictions in that group will be
            added.
            fn (str):  A string specifying the filename for the pse and commands files being written out which will have
            the format:
                f{fn}.pse
                f{fn}_all_pymol_commands.txt
            If None a default will be used as specified in PDBReference display_pairs().
        Returns:
            str: The path to the created pse file.
            str: The path to the text file containing all commands used to generate the pse file.
            list: The residues in the structure which have been colored by this method using the provided data.
            list: The pairs of residues in the structure which have been colored by this method using the provided data.
        """
        relevant_data = self._identify_relevant_data(category=category, n=n, k=k, coverage_cutoff=residue_coverage)
        if residue_coverage is None:
            mappable_res = sorted(self.query_pdb_mapper.query_pdb_mapping.keys())
            top_sequence_ranks = np.ones(self.query_pdb_mapper.seq_aln.seq_length) * np.inf
            residues_visited = set()
            groups = relevant_data.groupby('Top Predictions')
            for curr_rank in sorted(groups.groups.keys()):
                curr_group = groups.get_group(curr_rank)
                curr_residues = set(curr_group['Seq Pos 1']).union(set(curr_group['Seq Pos 2']))
                new_positions = curr_residues - residues_visited
                top_sequence_ranks[list(new_positions)] = curr_rank
                residues_visited |= new_positions
            top_residue_ranks = top_sequence_ranks[mappable_res]
            assert len(top_residue_ranks) <= len(mappable_res)
            _, top_coverage = compute_rank_and_coverage(seq_length=len(mappable_res), pos_size=1, rank_type='min',
                                                        scores=top_residue_ranks)
            relevant_data['Pos 1 Coverage'] = relevant_data['Seq Pos 1'].apply(lambda x: top_coverage[x])
            relevant_data['Pos 2 Coverage'] = relevant_data['Seq Pos 2'].apply(lambda x: top_coverage[x])
        coloring_data = relevant_data[['Struct Pos 1', 'Struct Pos 2', 'Pos 1 Coverage', 'Pos 2 Coverage', 'Rank']]
        coloring_data.rename(columns={'Struct Pos 1': 'RESIDUE_Index_1', 'Struct Pos 2': 'RESIDUE_Index_2'},
                             inplace=True)
        return self.query_pdb_mapper.pdb_ref.display_pairs(
            chain_id=self.query_pdb_mapper.best_chain, data=coloring_data, pair_col='Rank', res_col1='Pos 1 Coverage',
            res_col2='Pos 2 Coverage', data_direction='min', color_map='ET', out_dir=out_dir, fn=fn)

    def score_pdb_residue_identification(self, pdb_residues, n=None, k=None, coverage_cutoff=None):
        """
        Score PDB Residue Identification

        This function takes the top n or L/k pairs or the residues up to a specified coverage cutoff and determines the
        significance of their overlap with a set of residues on the structure provided to this ContactScorer instance.
        Significance is measured using the hypergoemteric test, the implementation here is adapted from:
        https://alexlenail.medium.com/understanding-and-implementing-the-hypergeometric-test-in-python-a7db688a7458
        This implementation evaluates the likelihood of choosing the number of given successes or more given the
        specified sample size (i.e. P(X>=1)).

        Args:
            pdb_residues (list): A list of the residues from the PDB structure used as reference for this scorer whose
            overlap with the scores should be tested.
            n (int): Specify n if you would like to evaluate the overlap of the top n pairs with the provided
            pdb_residues list.
            k (int): Specify k if you would like to evaluate the overlap of the top L/k (where L is the length of the
            query sequence) pairs with the provided pdb_residues list.
            coverage_cutoff (float): Specify coverage cutoff if you would like to evaluate pairs up to that percentage
            of the residues in the query sequence. For example if 0.3 is specified pairs will be added until 30% of the
            residues in the sequence are covered. If pairs are tied for rank the residues from those pairs are added at
            the same time. If adding pairs at a given rank would go over the specified coverage, those pairs are not
            added. The coverage is measured over all residues, not just those mappable to the structure.
        Return:
            int: The number of residues overlapping the provided residue list (the number of successes in the sample).
            int: The number of residues in the PDB chain (the population size).
            int: The number of residues in the provided residues list (the number of successes in the population).
            int: The number of residues passing the n, L/k, or coverage_cutoff threshold (the size of the sample).
            float: The hypergeometric p-value testing the likelihood of picking the number of residues which overlap the
            provided list of residues.
        """
        sub_df = self._identify_relevant_data(category='Any', n=n, k=k, coverage_cutoff=coverage_cutoff)
        top_pdb_residues = set(sub_df['Struct Pos 1']).union(set(sub_df['Struct Pos 2']))
        overlap = len(top_pdb_residues.intersection(set(pdb_residues)))
        # Perform hypergeometric test for correctly selecting the specified residue from the chain of interest.
        # X is still the number of drawn “successes” - The number of residues which overlap from the top pairs
        # M is the population size (previously N) - Size of PDB chain
        # n is the number of successes in the population (previously K) - Size of the list of pdb_residues passed in
        # N is the sample size (previously n) - The number of residues returned from the top n or L/k pairs,
        # or single residue coverage <= coverage cutoff.
        pvalue = hypergeom.sf(overlap - 1,
                              # self.query_pdb_mapper.pdb_ref.size[self.query_pdb_mapper.best_chain],
                              len(self.query_pdb_mapper.query_pdb_mapping),
                              len(pdb_residues),
                              len(top_pdb_residues))
        return (overlap, self.query_pdb_mapper.pdb_ref.size[self.query_pdb_mapper.best_chain], len(pdb_residues),
                len(top_pdb_residues), pvalue)

    def auroc_pdb_residue_identification(self, pdb_residues):
        """
        AUROC PDB Residue Identification

        This function takes the top n or L/k pairs or the residues up to a specified coverage cutoff and determines the
        AUROC of their recovery of a set of residues on the structure provided to this ContactScorer instance.

        Args:
            pdb_residues (list): A list of the residues from the PDB structure used as reference for this scorer whose
            overlap with the scores should be tested.
        Return:
            list: The True Positive Rate (tpr) scores measured while scoring the AUROC.
            list: The False Positive Rate (fpr) scores measured while scoring the AUROC.
            float: The AUROC score for the set of predictions mapped to this ContactScorer.
        """
        sub_df = self._identify_relevant_data(category='Any', coverage_cutoff=1.0)
        single_pos1_df = sub_df[['Struct Pos 1', 'Pos 1 Coverage']].rename(
            columns={'Struct Pos 1': 'Struct Pos', 'Pos 1 Coverage': 'Single Pos Coverage'})
        single_pos2_df = sub_df[['Struct Pos 2', 'Pos 2 Coverage']].rename(
            columns={'Struct Pos 2': 'Struct Pos', 'Pos 2 Coverage': 'Single Pos Coverage'})
        single_pos_df = single_pos1_df.append(single_pos2_df, ignore_index=True)
        single_pos_df.drop_duplicates(inplace=True)
        single_pos_df.sort_values(by='Single Pos Coverage', inplace=True)
        single_pos_df['Is Key Residue'] = single_pos_df['Struct Pos'].isin(pdb_residues)

        print(single_pos_df)
        fpr, tpr, _thresholds = roc_curve(single_pos_df['Is Key Residue'].values.astype(bool),
                                          1.0 - single_pos_df['Single Pos Coverage'].values, pos_label=True)
        auroc = auc(fpr, tpr)
        return tpr, fpr, auroc

    def evaluate_predictions(self, verbosity, out_dir, scores, coverages, ranks, dist='Any', file_prefix='',
                             biased_w2_ave=None, unbiased_w2_ave=None, processes=1, threshold=0.5, plots=True):
        """
        Evaluate Predictions

        This function evaluates a matrix of covariance predictions to the specified verbosity level.

        Args:
            scores (np.array): The predicted scores for pairs of residues in a sequence alignment.
            verbosity (int): What level of output to produce.
                1. Tests the AUROC and AUPRC of contact prediction at different levels of sequence separation.
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
            try:
                auc_roc = self.score_auc(category=separation)
            except IndexError:
                auc_roc = None, None, 'N/A'
            try:
                auc_prc = self.score_precision_recall(category=separation)
            except IndexError:
                auc_prc = None, None, 'N/A'
            if plots and ((auc_roc[2] != 'N/A') and (auc_prc[2] != 'N/A')):
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
        duplicate *= 9
        if verbosity >= 3:
            # Score Prediction Clustering
            z_score_fn = os.path.join(out_dir, file_prefix + 'Dist-{}_{}_ZScores.tsv')
            z_score_plot_fn = os.path.join(out_dir, file_prefix + 'Dist-{}_{}_ZScores.png')
            z_score_biased, b_w2_ave, b_scw_z_auc = self.score_clustering_of_contact_predictions(
                biased=True, file_path=z_score_fn.format(dist, 'Biased'), scw_scorer=biased_w2_ave,
                processes=processes)
            if (biased_w2_ave is None) and (b_w2_ave is not None):
                biased_w2_ave = b_w2_ave
            stats['Max Biased Z-Score'] = [np.max(pd.to_numeric(z_score_biased['Z-Score'], errors='coerce'))] * duplicate
            stats['AUC Biased Z-Score'] = [b_scw_z_auc] * duplicate
            z_score_unbiased, u_w2_ave, u_scw_z_auc = self.score_clustering_of_contact_predictions(
                biased=False, file_path=z_score_fn.format(dist, 'Unbiased'), scw_scorer=unbiased_w2_ave,
                processes=processes)
            if (unbiased_w2_ave is None) and (u_w2_ave is not None):
                unbiased_w2_ave = u_w2_ave
            stats['Max Unbiased Z-Score'] = [np.max(pd.to_numeric(z_score_unbiased['Z-Score'], errors='coerce'))] * duplicate
            stats['AUC Unbiased Z-Score'] = [u_scw_z_auc] * duplicate
            if plots:
                plot_z_scores(z_score_biased, z_score_plot_fn.format(dist, 'Biased'))
                plot_z_scores(z_score_unbiased, z_score_plot_fn.format(dist, 'Unbiased'))
        return stats, biased_w2_ave, unbiased_w2_ave

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
        if not isinstance(predictor, Predictor):
            raise TypeError('To evaluate a predictor it must have type Predictor, please use a valid Predictor.')
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


def heatmap_plot(name, data_mat, output_dir=None):
    """
    Heatmap Plot

    This method creates a heatmap using the Seaborn plotting package. The intended data are covariance raw scores or
    coverage scores.

    Args:
        name (str): Name used as the title of the plot and the filename for the saved figure.
        data_mat (np.array): A matrix of scores. This input should either be the score or coverage matrices from a
        predictor like the EvolutionaryTrace/ETMIPWrapper/DCAWrapper/EVCouplingsWrapper classes.
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
    scatterplot(x='Num_Residues', y='Z-Score', data=plotting_data)
    plt.savefig(file_path)
    plt.close()
