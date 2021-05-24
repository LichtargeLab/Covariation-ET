"""
Created on May 22, 2021

@author: dmkonecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from math import floor
from sklearn.metrics import auc
from multiprocessing import Pool
from scipy.stats import hypergeom
from SupportingClasses.utils import compute_rank_and_coverage
from Evaluation.Scorer import Scorer, init_scw_z_score_selection, scw_z_score_selection
from Evaluation.SelectionClusterWeighting import SelectionClusterWeighting


class SinglePositionScorer(Scorer):
    """
    SinglePositionScorer

    This class is meant to abstract the process of scoring a set of residue importance predictions.  This is being done
    for two main reasons. First, residue importance predictions are being made with several different methods and so
    their scoring should be performed consistently by another object or function. Second, this class will support
    several scoring methods such as the hypergoemtric test scoring of overlap with key sites as well as methods internal
    to the lab like the Selection Cluster Weighting Z-score derived from prior ET work.

    Attributes:
        query_pdb_mapper (SequencePDBMap): A mapping from the index of the query sequence to the index of the
        pdb chain's sequence for those positions which match according to a pairwise global alignment and vice versa.
        data (panda.DataFrame): A data frame containing a mapping of the sequence, structure, and predictions as well as
        labels to indicate which sequence separation category and top prediction category specific scores belong to.
        This is used for most evaluation methods.
    """

    def __init__(self, query, seq_alignment, pdb_reference, chain=None):
        """
        __init__

        This function initializes a new SinglePositionScorer, it accepts paths to an alignment and if available a pdb
        file to be used in the assessment of residue importance predictions for a given query.

        Args:
            query (str): The name of the query sequence/structure.
            seq_alignment (str/path, SeqAlignment): Path to the alignment being evaluated in this contact scoring
            prediction task or an already initialized SeqAlignment object.
            pdb_reference (str/path, PDBReference): The path to a PDB structure file, or an already initialized
            PDBReference object.
            chain (str): Which chain in the PDB structure to use for comparison and evaluation. If left blank the best
            chain will be identified by aligning the query sequence from seq_alignment against the chains in
            pdb_reference and the closest match will be selected.
        """
        super().__init__(query, seq_alignment, pdb_reference, chain)

    def fit(self):
        """
        Fit

        This function maps sequence positions between the query sequence from the alignment and residues in PDB file.
        If there are multiple chains for the structure in the PDB, the one which matches the query sequence best
        (highest global alignment score) is used and recorded in the seq_pdb_mapper attribute. This method updates the
        query_pdb_mapper and data class attributes.
        """
        super().fit()
        if (self.data is None) or (not self.data.columns.isin(['Seq Pos', 'Seq AA', 'Struct Pos', 'Struct AA']).all()):
            start = time()
            data_dict = {'Seq Pos': [], 'Seq AA': [], 'Struct Pos': [], 'Struct AA': []}
            for pos in range(self.query_pdb_mapper.seq_aln.seq_length):
                char = self.query_pdb_mapper.seq_aln.query_sequence[pos]
                struct_pos, struct_char = self.query_pdb_mapper.map_seq_position_to_pdb_res(pos)
                if struct_pos is None:
                    struct_pos, struct_char = '-', '-'
                data_dict['Seq Pos'].append(pos)
                data_dict['Seq AA'].append(char)
                data_dict['Struct Pos'].append(struct_pos)
                data_dict['Struct AA'].append(struct_char)
            data_df = pd.DataFrame(data_dict)
            self.data = data_df
            end = time()
            print('Constructing internal representation took {} min'.format((end - start) / 60.0))

    def map_predictions_to_pdb(self, ranks, predictions, coverages, threshold=0.5):
        """
        Map Predictions To PDB

        This method accepts a set of predictions and uses the mapping between the query sequence and the best PDB chain
        to extract the comparable predictions.

        Args:
            ranks (np.array): An array of ranks for predicted residue importance with size n where n is the length of
            the query sequence used when initializing the SinglePositionScorer.
            predictions (np.array): An array of prediction scores for residue importance with size n where n is the
            length of the query sequence used when initializing the SinglePositionScorer.
            coverages (np.array): An array of coverage scores for predicted residue importance with size n where n is
            the length of the query sequence used when initializing the SinglePositionScorer.
            threshold (float): The cutoff for coverage scores up to which scores (inclusive) are considered true.
        """
        # Add predictions to the internal data representation, simultaneously mapping them to the structural data.
        if self.data is None:
            self.fit()
        assert len(ranks) == len(self.data) and len(ranks) == len(predictions) and len(ranks) == len(coverages)
        self.data['Rank'] = ranks
        self.data['Score'] = predictions
        self.data['Coverage'] = coverages
        self.data['True Prediction'] = self.data['Coverage'].apply(lambda x: x <= threshold)

    def _identify_relevant_data(self, n=None, k=None, coverage_cutoff=None):
        """
        Identify Relevant Data

        This method accepts parameters describing a subset of the data stored in the SinglePositionScorer instance and
        subsets it. The order of operations is:
            2. Filter to only predicted residues which map to the best_chain of the provided PDB structure.
            3. Filter to the top predictions as specified by n, k, or coverage_cutoff

        Args:
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
            pd.DataFrame: A set of predictions which can be scored because they map successfully to the PDB and meet the
            criteria set by n, k, or coverage cutoff for top predictions.
        """
        # Defining for which of the pairs of residues there are both cET-MIp  scores and distance measurements from
        # the PDB Structure.
        if self.data is None:
            self.fit()
        if not pd.Series(['Rank', 'Score', 'Coverage']).isin(self.data.columns).all():
            raise ValueError('Ranks, Scores, and Coverage values must be provided through map_predictions_to_pdb,'
                             'before a specific evaluation can be made.')
        parameter_count = (k is not None) + (n is not None) + (coverage_cutoff is not None)
        if parameter_count > 1:
            raise ValueError('Only one parameter should be set, either: n, k, or coverage_cutoff.')
        unmappable_index = self.data.index[(self.data['Struct Pos'] == '-')]
        final_subset = self.data.drop(unmappable_index)
        final_df = final_subset.sort_values(by='Coverage')
        final_df['Top Predictions'] = final_df['Coverage'].rank(method='dense')
        if coverage_cutoff:
            assert isinstance(coverage_cutoff, float), 'coverage_cutoff must be a float!'
            mappable_res = sorted(self.query_pdb_mapper.query_pdb_mapping.keys())
            top_sequence_ranks = np.ones(self.query_pdb_mapper.seq_aln.seq_length) * np.inf
            residues_visited = set()
            groups = final_df.groupby('Top Predictions')
            for curr_rank in sorted(groups.groups.keys()):
                curr_group = groups.get_group(curr_rank)
                curr_residues = set(curr_group['Seq Pos'])
                new_positions = curr_residues - residues_visited
                top_sequence_ranks[list(new_positions)] = curr_rank
                residues_visited |= new_positions
            top_residue_ranks = top_sequence_ranks[mappable_res]
            assert len(top_residue_ranks) <= len(mappable_res)
            _, top_coverage = compute_rank_and_coverage(seq_length=len(mappable_res), pos_size=1, rank_type='min',
                                                        scores=top_residue_ranks)
            final_df['Pos Coverage'] = final_df['Seq Pos'].apply(lambda x: top_coverage[x])
            n = final_df.loc[(final_df['Pos Coverage'] > coverage_cutoff), 'Top Predictions'].min() - 1
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

    def score_pdb_residue_identification(self, pdb_residues, n=None, k=None, coverage_cutoff=None):
        """
        Score PDB Residue Identification

        This function takes the top n or L/k residues or the residues up to a specified coverage cutoff and determines
        the significance of their overlap with a set of residues on the structure provided to this SinglePositionScorer
        instance. Significance is measured using the hypergoemteric test, the implementation here is adapted from:
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
        sub_df = self._identify_relevant_data(n=n, k=k, coverage_cutoff=coverage_cutoff)
        top_pdb_residues = set(sub_df['Struct Pos'])
        overlap = len(top_pdb_residues.intersection(set(pdb_residues)))
        # Perform hypergeometric test for correctly selecting the specified residue from the chain of interest.
        # X is still the number of drawn “successes” - The number of residues which overlap from the top pairs
        # M is the population size (previously N) - Size of PDB chain
        # n is the number of successes in the population (previously K) - Size of the list of pdb_residues passed in
        # N is the sample size (previously n) - The number of residues returned from the top n or L/k pairs,
        # or single residue coverage <= coverage cutoff.
        pvalue = hypergeom.sf(overlap - 1,
                              len(self.query_pdb_mapper.query_pdb_mapping),
                              len(pdb_residues),
                              len(top_pdb_residues))
        return (overlap, self.query_pdb_mapper.pdb_ref.size[self.query_pdb_mapper.best_chain], len(pdb_residues),
                len(top_pdb_residues), pvalue)

    def select_and_color_residues(self, out_dir, n=None, k=None, residue_coverage=None, fn=None):
        """
        Select and Color Residues

        This method selects the correct subset of data and uses the PDBReference to create a pse file and text file of
        commands coloring the residues by their coverage score.

        Args:
            out_dir: The directory in which to save the pse file generated and the text file with the commands used to
            generate it.
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
        """
        relevant_data = self._identify_relevant_data(n=n, k=k, coverage_cutoff=residue_coverage)
        if residue_coverage is None:
            mappable_res = sorted(self.query_pdb_mapper.query_pdb_mapping.keys())
            top_sequence_ranks = np.ones(self.query_pdb_mapper.seq_aln.seq_length) * np.inf
            residues_visited = set()
            groups = relevant_data.groupby('Top Predictions')
            for curr_rank in sorted(groups.groups.keys()):
                curr_group = groups.get_group(curr_rank)
                curr_residues = set(curr_group['Seq Pos'])
                new_positions = curr_residues - residues_visited
                top_sequence_ranks[list(new_positions)] = curr_rank
                residues_visited |= new_positions
            top_residue_ranks = top_sequence_ranks[mappable_res]
            assert len(top_residue_ranks) <= len(mappable_res)
            _, top_coverage = compute_rank_and_coverage(seq_length=len(mappable_res), pos_size=1, rank_type='min',
                                                        scores=top_residue_ranks)
            relevant_data['Pos Coverage'] = relevant_data['Seq Pos'].apply(lambda x: top_coverage[x])
        coloring_data = relevant_data[['Struct Pos', 'Pos Coverage']].rename(
            columns={'Struct Pos': 'RESIDUE_Index', 'Pos Coverage': 'Coverage'})
        coloring_data = coloring_data.drop_duplicates()
        self.query_pdb_mapper.pdb_ref.color_structure(chain_id=self.query_pdb_mapper.best_chain, data=coloring_data,
                                                      data_type='Coverage', data_direction='min', color_map='ET',
                                                      out_dir=out_dir, fn=fn)

    def score_clustering_of_important_residues(self, biased=True, file_path='./z_score.tsv', scw_scorer=None,
                                               processes=1):
        """
        Score Clustering Of Important Residues

        This method employs the SelectionClusterWeighting class and support functions in the Scorer class to score all
        residues for which predictions are made. A Z-Score of '-' means that the residue did not map to the provided PDB
        while 'NA' means that the sigma computed for that residue was equal to 0. Each residue is added to a set in
        order of predicted importance. That set of residues is evaluated for selection cluster weighting z-score after
        each residue is added.

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
            pd.DataFrame: Table holding the residue, predicted importance of that residue, the clustering Z-Score,
            the w score, E[w], E[w^2], sigma, and the number of residues of interest up to that point.
            SelectionClusterWeighting: The SelectionClusterWeighting object used in this computation, which stores the
            precomputable parts of E[w^2], which can reused for later computations (i.e. cases 1, 2, and 3).
            float: The area under the curve defined by the z-scores and the protein coverage.
        """
        start = time()
        if not self.query_pdb_mapper.is_aligned():
            raise AttributeError('Z-Scores cannot be measured, fit() must be called first!')
            # If data has already been computed load and return it without recomputing.
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, sep='\t', header=0, index_col=False)
            df_sub = df.replace([None, '-', 'NA'], np.nan)
            df_sub = df_sub.dropna()[['Num_Residues', 'Z-Score']]
            df_sub.drop_duplicates(inplace=True)
            df_sub['Coverage'] = df_sub['Num_Residues'] / float(self.query_pdb_mapper.seq_aln.seq_length)
            df_sub.sort_values(by='Coverage', ascending=True, inplace=True)
            if len(df_sub['Coverage']) == 0:
                au_scw_z_score_curve = None
            else:
                au_scw_z_score_curve = auc(df_sub['Coverage'].astype(float).values,
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
        data_df = self._identify_relevant_data().loc[:, ['Seq Pos', 'Score', 'Coverage']]
        data_df.rename(columns={'Seq Pos': 'Res', 'Score': 'Importance_Score', 'Coverage': 'Temp'}, inplace=True)
        _, data_df['Coverage'] = compute_rank_and_coverage(seq_length=len(self.query_pdb_mapper.query_pdb_mapping),
                                                           pos_size=1, rank_type='min',
                                                           scores=data_df['Temp'])
        data_df.drop(columns=['Temp'], inplace=True)
        data_df.sort_values(by='Coverage', inplace=True, ignore_index=True)
        # Set up structures to track unique sets of residues
        residues_visited = set()
        unique_sets = []
        to_score = []
        # Identify unique sets of residues to score
        groups = data_df.groupby('Coverage')
        for curr_cov in sorted(groups.groups.keys()):
            curr_group = groups.get_group(curr_cov)
            curr_residues = set(curr_group['Res'])
            new_positions = curr_residues - residues_visited
            unique_sets.append(sorted(new_positions))
            residues_visited |= new_positions
            to_score.append(sorted(residues_visited))
        data = {'Z-Score': [], 'W': [], 'W_Ave': [], 'W2_Ave': [], 'Sigma': [], 'Num_Residues': []}
        # Compute all other Z-scores
        pool2 = Pool(processes=processes, initializer=init_scw_z_score_selection,
                     initargs=(scw_scorer, ))
        res = pool2.map(scw_z_score_selection, to_score)
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
                curr_cov = data_df.loc[data_df['Res'].isin(unique_sets[counter]), 'Coverage'].unique()
                assert len(curr_cov) == 1
                x_coverage.append(curr_cov[0])
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
        data_df_unmapped = self.data.loc[~ self.data['Seq Pos'].isin(self.query_pdb_mapper.query_pdb_mapping),
                                         ['Seq Pos', 'Score', 'Coverage']].sort_values(by='Coverage')
        data_df_unmapped.drop(columns=['Coverage'])
        data_df_unmapped['Coverage'] = '-'
        data_df_unmapped.rename(columns={'Seq Pos': 'Res', 'Score': 'Importance_Score'},
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
        df[['Res', 'Importance_Score', 'Coverage', 'Z-Score', 'W', 'W_Ave', 'Sigma', 'Num_Residues']].to_csv(
            path_or_buf=file_path, sep='\t', header=True, index=False)
        end = time()
        print('Compute SCW Z-Score took {} min'.format((end - start) / 60.0))
        return df, scw_scorer, au_scw_z_score_curve
