"""
Created on August 10, 2019

@author: Daniel Konecki
"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from copy import deepcopy
from seaborn import heatmap
from multiprocessing import Pool
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator
from SupportingClasses.Trace import Trace
from SupportingClasses.Predictor import Predictor
from SupportingClasses.utils import compute_rank_and_coverage
from SupportingClasses.PhylogeneticTree import PhylogeneticTree
from SupportingClasses.PositionalScorer import PositionalScorer
from SupportingClasses.AlignmentDistanceCalculator import AlignmentDistanceCalculator


class EvolutionaryTrace(Predictor):
    """
    This class draws on all supporting classes and methods to implement the full Evolutionary Trace algorithm.

    This class inherits from the Predictor class which means it shares the attributes and initializer of that class,
    as well as implementing the calculate_scores method.

    Attributes:
        query (str): The sequence identifier for the sequence being analyzed.
        polymer_type (str): What kind of sequence information is being analyzed (.i.e. Protein or DNA).
        original_aln_fn (str): The path to the alignment to analyze.
        original_aln (SeqAlignment): A SeqAlignment object representing the alignment in original_aln_fn.
        et_distance (bool): Whether or not to use the Evolutionary Trace distance measure. This is a sequence similarity
        metric computed over the specified distance_model.
        distance_model (str): What type of distance model to use when computing the distance between sequences in the
        alignment (e.g. blosum62).
        distance_matrix (Bio.Phylo.TreeConstruction.DistanceMatrix): The distance matrix computed over the provided
        alignment using the specified distance_model (and potentially the Evolutionary Trace distance method).
        tree_building_method (str): Which tree construction methodology to use, current options are: 'et', 'upgma',
        'agglomerative', or 'custom'. Additional information can be found in the PhylogeneticTree class.
        tree_building_options (dict): Which options to use for tree construction with the specified method. Details for
        available options and their values can be found for each method in the PhylogeneticTree class.
        phylo_tree (PhylogeneticTree): The tree constructed from the distance_matrix using the specified
        tree_building_method and tree_building_options.
        phylo_tree_fn (str): Path to where the phylo_tree object will be written if specified (in nhx/Newick format).
        ranks (list/None): The ranks to analyze when performing the trace. If None, all ranks will be analyzed.
        assignments (dict): The rank and group level assignments. The root node, its terminal leaves, and its
        descendants are tracked for each rank (first key) and group (second key).
        non_gapped_aln_fn (str): Path to where the non-gapped alignment will be written (if specified in output_files).
        non_gapped_aln (SeqAlignment): SeqAlignment object representing the original alignment with all columns which
        are gaps in the query sequence removed.
        position_type (str): Whether the positions being analyzed are 'single' (a position specific impact analysis),
        or 'pair' (covariation analysis over pairs of positions).
        scoring_metric (str): Which method to use when scoring each group sub-alignment, available metrics are:
        identity, plain_entropy, mutual_information, normalized_mutual_information,
        average_product_corrected_mutual_information, and filtered_average_product_corrected_mutual_information. Method
        usage is constricted based on position type.
        gap_correction (float): Whether to correct final scores for gap content. If a value other than None is provided,
        columns whose gap content is greater than the specified float (should be between 0.0 and 1.0) will have their
        score replaced with the worst observed score during the trace (down weighting all highly gapped columns). The
        default value for this has traditionally been set to 0.6 for rvET but has not been used for intET or ET-MIp.
        trace (Trace): A Trace object used to organize data required for performing a trace and the methods to do so.
        scorer (PositionalScorer): A PositionalScorer object which uses the specified scoring metric to score groups and
        takes the combined group scores at a given rank to calculate a final rank score.
        rankings (np.array): The rank (lowest being best, highest being worst) of each single or paired position in the
        provided alignment as determined from the calculated scores.
        scores (np.array): The raw scores calculated for each single or paired position in the provided alignment while
        performing the trace. For some methods (identity, plain_entropy) the lower the score the better, while for
        others (mutual_information, normalized_mutual_information, average_product_corrected_mutual_information, and
        filtered_average_product_corrected_mutual_information) the higher the score the better.
        coverages (np.array): The percentage of scores at or better than the score for this single or paired position
        (i.e. the percentile rank).
        out_dir (str): The path where results of this analysis should be written to.
        output_files (set): Which files to write out, possible values include: 'original_aln', 'non-gap_aln', 'tree',
        'sub-alignment', 'frequency_tables', and 'scores'.
        processors (int): The number of CPU cores which this analysis can use while running.
        low_memory (bool): Whether or not to serialize files during execution in order to avoid keeping everything in
        memory (this is important for large alignments).
    """

    def __init__(self, query, polymer_type, aln_file, et_distance, distance_model, tree_building_method,
                 tree_building_options, ranks, position_type, scoring_metric, gap_correction, out_dir,
                 output_files, processors, low_memory):
        """
        Initialization for EvolutionaryTrace class:

        Args:
            query (str): The sequence identifier for the sequence being analyzed.
            polymer_type (str): What kind of sequence information is being analyzed (.i.e. Protein or DNA).
            aln_file (str): The path to the alignment to analyze.
            et_distance (bool): Whether or not to use the Evolutionary Trace distance measure. This is a sequence
            similarity metric computed over the specified distance_model.
            distance_model (str): What type of distance model to use when computing the distance between sequences in
            the alignment (e.g. blosum62).
            tree_building_method (str): Which tree construction methodology to use, current options are: 'et', 'upgma',
            'agglomerative', or 'custom'. Additional information can be found in the PhylogeneticTree class.
            tree_building_options (dict): Which options to use for tree construction with the specified method. Details
            for available options and their values can be found for each method in the PhylogeneticTree class.
            ranks (list/None): The ranks to analyze when performing the trace. If None, all ranks will be analyzed.
            position_type (str): Whether the positions being analyzed are 'single' (a position specific impact
            analysis), or 'pair' (covariation analysis over pairs of positions).
            scoring_metric (str): Which method to use when scoring each group sub-alignment, available metrics are:
            identity, plain_entropy, mutual_information, normalized_mutual_information,
            average_product_corrected_mutual_information, and filtered_average_product_corrected_mutual_information.
            Method usage is constricted based on position type.
            gap_correction (float): Whether to correct final scores for gap content. If a value other than None is
            provided, columns whose gap content is greater than the specified float (should be between 0.0 and 1.0) will
            have their score replaced with the worse observed score during the trace (down weighting all highly gapped
            columns). The default value for this has traditionally been set to 0.6 for rvET but has not been used for
            intET or ET-MIp.
            out_dir (str): The path where results of this analysis should be written to.
            output_files (set): Which files to write out, possible values include: 'original_aln', 'non-gap_aln',
            'tree', 'sub-alignment', 'frequency_tables', and 'scores'.
            processors (int): The number of CPU cores which this analysis can use while running.
            low_memory (bool): Whether or not to serialize files during execution in order to avoid keeping everything
            in memory (this is important for large alignments).
        """
        super().__init__(query, aln_file, polymer_type, out_dir)
        self.method = 'ET'
        self.et_distance = et_distance
        self.distance_model = distance_model
        self.distance_matrix = None
        self.tree_building_method = tree_building_method
        self.tree_building_options = tree_building_options
        self.phylo_tree = None
        self.phylo_tree_fn = None
        self.ranks = ranks
        self.assignments = None
        self.position_type = position_type
        self.scoring_metric = scoring_metric
        self.gap_correction = gap_correction
        self.trace = None
        self.scorer = None
        self.output_files = output_files
        self.processors = processors
        self.low_memory = low_memory

    def compute_distance_matrix_tree_and_assignments(self):
        """
        Compute Distance Matrix, Tree, and Assignments

        This function computes the distance matrix based on the original alignment provided using the specified distance
        model (and if specified the Evolutionary Trace distance calculation method). This distance matrix is then used
        to build a phylogenetic tree using the specified tree_building_method and tree_building_options. If specified
        the tree is written to the output directory (in Newick/nhx format). Finally, nodes from the tree are assigned to
        ranks and groups.
        """
        serial_fn = f'{self.query}_{("ET_" if self.et_distance else "")}{self.distance_model}_Dist_'\
                    f'{self.tree_building_method}_Tree.pkl'
        serial_fn = os.path.join(self.out_dir, serial_fn)
        if os.path.isfile(serial_fn):
            with open(serial_fn, 'rb') as handle:
                self.distance_matrix, self.phylo_tree, self.phylo_tree_fn, self.assignments = pickle.load(handle)
            # This check is performed for legacy files where the full path was stored instead of the relative path
            if os.path.isabs(self.phylo_tree_fn):
                self.phylo_tree_fn = os.path.basename(self.phylo_tree_fn)
        else:
            calculator = AlignmentDistanceCalculator(protein=(self.polymer_type == 'Protein'),
                                                     model=self.distance_model, skip_letters=None)
            if self.et_distance:
                _, self.distance_matrix, _, _ = calculator.get_et_distance(self.original_aln.alignment,
                                                                           processes=self.processors)
            else:
                self.distance_matrix = calculator.get_distance(self.original_aln.alignment, processes=self.processors)
            start_tree = time()
            self.phylo_tree = PhylogeneticTree(tree_building_method=self.tree_building_method,
                                               tree_building_args=self.tree_building_options)
            self.phylo_tree.construct_tree(dm=self.distance_matrix)
            self.phylo_tree_fn = f'{self.query}_{("ET_" if self.et_distance else "")}{self.distance_model}_dist_'\
                                 f'{self.tree_building_method}_tree.nhx'
            end_tree = time()
            print('Constructing tree took: {} min'.format((end_tree - start_tree) / 60.0))
            self.assignments = self.phylo_tree.assign_group_rank(ranks=self.ranks)
            with open(serial_fn, 'wb') as handle:
                pickle.dump((self.distance_matrix, self.phylo_tree, self.phylo_tree_fn, self.assignments), handle,
                            pickle.HIGHEST_PROTOCOL)
        if 'tree' in self.output_files:
            self.phylo_tree.write_out_tree(filename=os.path.join(self.out_dir, self.phylo_tree_fn))

    def perform_trace(self):
        """
        Perform Trace

        This method collects all of the data generated thus far and performs a trace over the non-gapped alignment using
        the constructed phylogenetic tree. While characterizing groups the sub-alignment and frequency table can be
        written to file if specified. Groups are then scored using the specified scoring metric and scores are
        combined to compute a final rank score. The rank scores are combined to generate final scores which are then
        ranked and coverage scores are computed.
        """
        serial_fn = '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.pkl'.format(
            self.query, ('ET_' if self.et_distance else ''), self.distance_model, self.tree_building_method,
            ('All_Ranks' if self.ranks is None else 'Custom_Ranks'), self.scoring_metric)
        serial_fn = os.path.join(self.out_dir, serial_fn)
        self.scorer = PositionalScorer(seq_length=self.non_gapped_aln.seq_length,
                                       pos_size=(1 if self.position_type == 'single' else 2),
                                       metric=self.scoring_metric)
        if os.path.isfile(serial_fn):
            with open(serial_fn, 'rb') as handle:
                self.trace, self.rankings, self.scores, self.coverages = pickle.load(handle)
        else:
            self.trace = Trace(alignment=self.non_gapped_aln, phylo_tree=self.phylo_tree,
                               group_assignments=self.assignments,
                               pos_size=(1 if self.position_type == 'single' else 2),
                               match_mismatch=(('match' in self.scoring_metric) or ('mismatch' in self.scoring_metric)),
                               output_dir=self.out_dir, low_memory=self.low_memory)
            self.trace.characterize_rank_groups(processes=self.processors,
                                                write_out_sub_aln='sub-alignments' in self.output_files,
                                                write_out_freq_table='frequency_tables' in self.output_files)
            self.rankings, self.scores, self.coverages = self.trace.trace(scorer=self.scorer, processes=self.processors,
                                                                          gap_correction=self.gap_correction)
            with open(serial_fn, 'wb') as handle:
                pickle.dump((self.trace, self.rankings, self.scores, self.coverages), handle, pickle.HIGHEST_PROTOCOL)
        # Generate descriptive file name
        rank_fn = '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.ranks'.format(
            self.query, ('ET_' if self.et_distance else ''), self.distance_model, self.tree_building_method,
            ('All_Ranks' if self.ranks is None else 'Custom_Ranks'), self.scoring_metric)
        if os.path.isfile(os.path.join(self.out_dir, rank_fn)):
            print('Evolutionary Trace analysis with the same parameters already saved to this location.')
        else:
            reordered_aln = deepcopy(self.non_gapped_aln)
            reorder_df = pd.DataFrame({'IDs': self.non_gapped_aln.seq_order,
                                       'Indexes': list(range(self.non_gapped_aln.size))}).set_index('IDs')

            print(reordered_aln.seq_order == self.trace.assignments[1][1]['terminals'])
            reordered_aln.seq_order = self.trace.assignments[1][1]['terminals']
            reordered_seq = []
            for x in reorder_df.loc[reordered_aln.seq_order, 'Indexes'].values.tolist():
                reordered_seq.append(reordered_aln.alignment[x])
            reordered_aln.alignment = MultipleSeqAlignment(reordered_seq)
            write_out_et_scores(file_name=rank_fn, out_dir=self.out_dir, aln=reordered_aln,
                                pos_size=self.trace.pos_size, match_mismatch=False, ranks=self.rankings,
                                scores=self.scores, coverages=self.coverages, precision=3, processors=self.processors)
            if (self.position_type == 'pair') and ('single_pos_scores' in self.output_files):
                rank_fn = convert_pair_to_single_residue_output(res_fn=os.path.join(self.out_dir, rank_fn), precision=3)
                if 'legacy' in self.output_files:
                    convert_file_to_legacy_format(res_fn=rank_fn, reverse_score=(self.scorer.rank_type == 'max'))
            else:
                if 'legacy' in self.output_files:
                    convert_file_to_legacy_format(res_fn=os.path.join(self.out_dir, rank_fn),
                                                  reverse_score=(self.scorer.rank_type == 'max'))

    def calculate_scores(self):
        """
        Calculate Scores

        This method calls the compute_distance_matrix_tree_and_assignments and the perform_trace to generate predictive
        scores for position importance or pair covariance using the Evolutionary Trace method.
        """
        start = time()
        self.compute_distance_matrix_tree_and_assignments()
        self.perform_trace()
        end = time()
        self.time = end - start
        print(self.time)

    def visualize_trace(self, positions, ranks=None):
        """
        Visualize Trace

        This function is meant to create an image of a position or pair of position (or other subset of positions)
        visualizing the alignment at each level in the trace. The images for each level of the trace will be saved to
        a directory within the out_dir, named by joining each position specified in positions with an '_'.

        Arguments:
            positions (list): The subset of positions in the alignment for which to visualize their progression through
            the EvolutionaryTrace.
            ranks (list): Which ranks to visualize. If None, all ranks which have been calculated will be visualized. If
            a list is provided all elements which overlap with the ranks actually computed will be visualized.
        """
        start = time()
        # Generate an alignment with only the specified positions and ordered such that the sequence order represents
        # the ordering used when constructing the phylogenetic tree.
        sub_aln = self.non_gapped_aln.generate_positional_sub_alignment(positions=positions)
        tree_seq_order = [self.assignments[max(self.assignments.keys())][i]['terminals'][0]
                          for i in sorted(self.assignments[max(self.assignments.keys())].keys())]
        look_up = {seq_id: i for i, seq_id in enumerate(sub_aln.seq_order)}
        sub_aln_to = deepcopy(sub_aln)
        sub_aln_to.seq_order = tree_seq_order
        sub_aln_to.alignment = MultipleSeqAlignment([deepcopy(sub_aln.alignment[look_up[x], :])
                                                     for x in sub_aln_to.seq_order])
        check1 = time()
        print('It took {} sec to generate the sub-alignment for visualization.'.format(check1 - start))
        # Setup the output directory.
        sub_dir = os.path.join(self.out_dir, '_'.join([str(x) for x in positions]))
        os.makedirs(sub_dir, exist_ok=True)
        # Specify the full dimensions to make the plots, based on the sizing expected for each cell and each individual
        # plot.
        cellsize = 0.0001  # inch
        marg_top = 0.
        marg_bottom = 0.0
        marg_left = 0.0
        marg_right = 0.0
        # Determine individual plot sizing
        cells_in_row = len(positions)
        sub_fig_width = (cellsize * cells_in_row) + marg_left + marg_right
        cells_in_column = 1
        sub_fig_height = (cellsize * cells_in_column) + marg_top + marg_bottom
        # Identify which ranks to visualize
        if ranks is None:
            ranks = list(sorted(self.assignments.keys()))
        else:
            ranks = list(sorted(set(ranks).intersection(set(self.assignments.keys()))))
        # Determine the full size of the image to make to hold all of the plots.
        num_rows = sub_aln.size
        num_cols = sub_aln.size
        # num_cols = len(ranks)
        col_spacing = 2.0
        row_spacing = 1.0
        fig_width = (num_cols * sub_fig_width) + ((num_cols - 1) * col_spacing)
        fig_height = (num_rows * sub_fig_height) + ((num_rows - 1) * row_spacing)
        # Create the figure canvas and the subdivisions needed to plot all of the ranks and groups in the trace.
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(nrows=sub_aln.size, ncols=sub_aln.size, wspace=col_spacing, hspace=row_spacing)
        # Create plotting axes and fill with blank figures to establish the size of the plotting area
        plotting_data = {}
        print('Creating plot area')
        for r in ranks:
            plotting_data[r] = {}
            starting_gs_pos = 0
            for g in sorted(self.assignments[r].keys()):
                plotting_data[r][g] = {}
                plotting_data[r][g]['aln'] = sub_aln.generate_sub_alignment(
                    sequence_ids=self.assignments[r][g]['terminals'])
                plotting_data[r][g]['ax'] = fig.add_subplot(
                    gs[starting_gs_pos: starting_gs_pos + plotting_data[r][g]['aln'].size, r - 1])
                hm = heatmap(data=np.zeros((plotting_data[r][g]['aln'].size, len(positions))), cmap=cm.Greys,
                             center=10.0, vmin=0.0, vmax=20.0, cbar=False, square=True,
                             xticklabels=False, yticklabels=False, ax=plotting_data[r][g]['ax'])
                hm.tick_params(left=False, bottom=False)
                hm.set_title('Rank {} Group {}'.format(r, g), color='w')
                starting_gs_pos += plotting_data[r][g]['aln'].size
        check2 = time()
        print('It took {} sec to create the fully sized canvas for visualizing the trace.'.format(check2 - check1))
        # Fill in the figures and save at each rank
        for r in ranks:
            print('Plotting Rank: {}'.format(r))
            for g in sorted(self.assignments[r].keys()):
                plotting_data[r][g]['ax'].clear()
                plotting_data[r][g]['aln'].heatmap_plot(name='Rank {} Group {}'.format(r, g),
                                                        out_dir=sub_dir, save=False,
                                                        ax=plotting_data[r][g]['ax'])
                if r > 1:
                    plotting_data[r][g]['ax'].set_xticklabels([None] * 2)
                    plotting_data[r][g]['ax'].set_yticklabels([None] * plotting_data[r][g]['aln'].size)
            fig.savefig(os.path.join(sub_dir, 'Rank_{}.png'.format(r)), bbox_inches='tight')
        end = time()
        print('Full visualization took {} sec.'.format(end - start))


def init_var_pool(aln, match_mismatch):
    """
    Initialize Variability Pool

    Args:
        aln (SeqAlignment): The root level SeqAlignment (gaps removed for the query sequence) for the trace which is
        being written to file.
        match_mismatch (bool): Whether the analysis being visualized is the result of a match/mismatch analysis or not.
    """
    global var_aln, mm_check
    var_aln = aln
    mm_check = match_mismatch


def get_var_pool(pos):
    """
    Get Variability Pool

    This function retrieves the characters observed at a given position in a query sequence, as well as across all
    sequences in a MultipleSequenceAlignment, counts them, and turns them into a string.

    Args:
        pos (tuple): A tuple of one or two ints specifying the position for which characters should be retrieved.
    Returns:
        tuple: The position(s) characterized by this return.
        tuple: The nucleic/amino acids observed at the specified position(s).
        int: The count of unique characters observed at the specified position in the alignment.
        str: A string listing characters observed at the specified position, separated by commas.
    """
    if len(pos) not in [1, 2]:
        raise ValueError('Only single positions or pairs of positions accepted at this time.')
    pos_i = int(pos[0])
    query_i = var_aln.query_sequence[pos_i]
    col_i = list(var_aln.alignment[:, pos_i])
    if len(pos) == 1:
        pos_final = (pos_i, )
        query_final = (query_i, )
        col_final = col_i
    else:
        pos_j = int(pos[1])
        query_j = var_aln.query_sequence[pos_j]
        col_j = list(var_aln.alignment[:, pos_j])
        pos_final = (pos_i, pos_j)
        query_final = (query_i, query_j)
        col_final = [i + j for i, j in zip(col_i, col_j)]
    if mm_check:
        mm_col = []
        for i in range(len(col_final)):
            curr_col = [col_final[i] + x for x in col_final[i+1:]]
            mm_col += curr_col
        col_final = mm_col
    col_final = set(col_final)
    character_str = ','.join(sorted(col_final))
    character_count = len(col_final)
    return pos_final, query_final, character_str, character_count


def write_out_et_scores(file_name, out_dir, aln, pos_size, match_mismatch, ranks, scores, coverages, precision=3,
                        processors=1):
    """
    Write Out Evolutionary Trace Scores

    This method writes out the results of the Evolutionary Trace analysis.

    Args:
        file_name (str): The name to write the results to.
        out_dir (str): The directory to write the results file to.
        aln (SeqAlignment): The non-gapped sequence alignment used to perform the trace which is being written out.
        pos_size (int): The size of positions analyzed 1 for single, 2 for pair, etc.
        match_mismatch (bool): Whether the analysis being visualized is the result of a match/mismatch analysis or not.
        ranks (np.array): The ranking of each position analyzed in the trace.
        scores (np.array): The score for each position analyzed in the trace.
        coverages (np.array): The coverage for each position analyzed in the trace.
        precision (int): The number of decimal places to write out for floating point values such coverages (and scores
        if a real valued scoring metric was used).
        processors (int): If pairs of residues were scored in the trace being written to file, then this will be the
        size of the multiprocessing pool used to speed up the slowest step (retrieving characters at each position to
        describe its variability).
    """
    full_path = os.path.join(out_dir, file_name)
    if os.path.isfile(full_path):
        print('Evolutionary Trace analysis with the same parameters already saved to this location.')
        return
    start = time()
    if pos_size not in [1, 2]:
        raise ValueError("write_out_et_scores is not implemented to work with scoring for position sizes other than 1 "
                         "or 2.")
    scoring_dict = {}
    columns = []
    # Define indices for writing
    if pos_size == 1:
        indices = np.r_[0:aln.seq_length]
        total = len(indices)
    else:
        indices = np.triu_indices(aln.seq_length, k=1)
        total = len(indices[0])
    # Characterize each positions query sequence and variability.
    var_data = []
    var_pbar = tqdm(total=total, unit='variation')

    def update_variation(return_tuple):
        """
        Update Variation

        This function serves to update the progress bar for query and variation data. It also updates the var_data list
        which will be used to complete the data for file writing.

        Args:
            return_tuple (tuple): A tuple consisting of a tuple defining the position characterized, a tuple containing
            the data for the query position(s), a string of the variation at that position, and the count of that
            variation.
        """
        var_data.append(return_tuple)
        var_pbar.update(1)
        var_pbar.refresh()

    pool = Pool(processes=processors, initializer=init_var_pool, initargs=(aln, match_mismatch))
    if pos_size == 1:
        for x in indices:
            pool.apply_async(get_var_pool, ((int(x),),), callback=update_variation)
    else:
        for x in range(len(indices[0])):
            pool.apply_async(get_var_pool, ((int(indices[0][x]), int(indices[1][x])),), callback=update_variation)
    pool.close()
    pool.join()
    var_pbar.close()
    var_data = sorted(var_data)
    positions, queries, var_strings, var_counts = zip(*var_data)
    # Fill in the data dictionary for writing, starting with Position and Query data which differs for one and two
    # position traces.
    # if frequency_table.position_size == 1:
    if pos_size == 1:
        scoring_dict['Position'] = list(indices + 1)
        scoring_dict['Query'] = [q[0] for q in queries]
        columns += ['Position', 'Query']
    else:
        scoring_dict['Position_i'] = list(indices[0] + 1)
        scoring_dict['Position_j'] = list(indices[1] + 1)
        query_i, query_j = zip(*queries)
        scoring_dict['Query_i'] = list(query_i)
        scoring_dict['Query_j'] = list(query_j)
        columns += ['Position_i', 'Position_j', 'Query_i', 'Query_j']
    scoring_dict['Variability_Characters'] = var_strings
    scoring_dict['Variability_Count'] = var_counts
    scoring_dict['Rank'] = list(ranks[indices])
    scoring_dict['Score'] = list(scores[indices])
    scoring_dict['Coverage'] = list(coverages[indices])
    columns += ['Variability_Count', 'Variability_Characters', 'Rank', 'Score', 'Coverage']
    # Write the data out to file using pandas.
    scoring_df = pd.DataFrame(scoring_dict)
    scoring_df.to_csv(full_path, sep='\t', header=True, index=False, float_format='%.{}f'.format(precision),
                      columns=columns)
    end = time()
    print('Results written to file in {} min'.format((end - start) / 60.0))


def convert_pair_to_single_residue_output(res_fn, precision=3):
    """
    Convert Pair To Single Residue Output

    This function accepts the file name for a pair residue result, reads it in, and converts it to a single residue
    result. This means the first occurrence of each residue by pair rank is recorded and new coverage scores and ranks
    are computed based on the single residue data. The coverage and ranks are computed based on the original ranks since
    these are not impacted by the precision used when writing out the original file, however the original score is
    preserved in the 'Score' column. If many pairs (and therefore residues) were tied in their rank, this is preserved
    in the conversion. The residue variability is parsed from the pair of positions data, by keeping only the first or
    second character from the list of unique characters observed for a pair of positions.

    Args:
        res_fn (str): The path to the file to the pair ranks file to be converted.
        precision (int): The number of decimal places to write out for floating point values such coverages (and scores
        if a real valued scoring metric was used). If a precision greater than that used to write out the original file
        is provided, no additional value will be added to the 'Score' column since this is preserved from the original
        file, however the 'Rank' and 'Coverage' columns will reflect this change.
    Return:
        str: The path to the converted output file.
    """
    res_dir = os.path.dirname(res_fn)
    res_base_name, res_base_extension = os.path.splitext(os.path.basename(res_fn))
    scoring_df = pd.read_csv(res_fn, sep='\t', header=0, index_col=None)
    assert {'Position_i', 'Position_j', 'Query_i', 'Query_j'}.issubset(scoring_df.columns), "Provided file does not "\
                                                                                            "include expected columns,"\
                                                                                            " make sure this is a pair"\
                                                                                            " analysis result!"

    columns = ['Position', 'Query', 'Variability_Count', 'Variability_Characters', 'Rank', 'Score', 'Coverage']
    converted_scoring_data = {x: [] for x in columns + ['Original_Rank']}

    all_res = len(set(scoring_df['Position_i']).union(set(scoring_df['Position_j'])))
    counter = 0
    completed = set()
    rank_groups = scoring_df.groupby('Rank')
    for rank in sorted(rank_groups.groups.keys()):
        if counter == all_res:
            break

        curr_group = rank_groups.get_group(rank)
        for i in curr_group.index:
            if curr_group.loc[i, 'Position_i'] not in completed:
                converted_scoring_data['Position'].append(curr_group.loc[i, 'Position_i'])
                converted_scoring_data['Query'].append(curr_group.loc[i, 'Query_i'])
                converted_scoring_data['Original_Rank'].append(rank)
                converted_scoring_data['Score'].append(curr_group.loc[i, 'Score'])
                var_chars = list(set([x[0] for x in curr_group.loc[i, 'Variability_Characters'].split(',')]))
                converted_scoring_data['Variability_Count'].append(len(var_chars))
                converted_scoring_data['Variability_Characters'].append(','.join(var_chars))
                completed.add(curr_group.loc[i, 'Position_i'])
            if curr_group.loc[i, 'Position_j'] not in completed:
                converted_scoring_data['Position'].append(curr_group.loc[i, 'Position_j'])
                converted_scoring_data['Query'].append(curr_group.loc[i, 'Query_j'])
                converted_scoring_data['Original_Rank'].append(rank)
                converted_scoring_data['Score'].append(curr_group.loc[i, 'Score'])
                var_chars = list(set([x[1] for x in curr_group.loc[i, 'Variability_Characters'].split(',')]))
                converted_scoring_data['Variability_Count'].append(len(var_chars))
                converted_scoring_data['Variability_Characters'].append(','.join(sorted(var_chars)))
                completed.add(curr_group.loc[i, 'Position_j'])
    new_ranks, new_coverage = compute_rank_and_coverage(seq_length=all_res, pos_size=1, rank_type='min',
                                                        scores=np.array(converted_scoring_data['Original_Rank']))
    converted_scoring_data['Rank'] = new_ranks
    converted_scoring_data['Coverage'] = new_coverage
    converted_scoring_df = pd.DataFrame(converted_scoring_data)
    converted_scoring_df.sort_values(by='Position', inplace=True)
    full_path = os.path.join(res_dir, f'{res_base_name}_Converted_To_Single_Pos{res_base_extension}')
    converted_scoring_df.to_csv(full_path, sep='\t', header=True, index=False, float_format='%.{}f'.format(precision),
                                columns=columns)
    return full_path


def convert_file_to_legacy_format(res_fn, reverse_score=False):
    """
    Convert File To Legacy Format

    This function accepts a single residue analysis file (tab separated .ranks file) and convets it to the legacy ETC
    ranks file format. This file format is needed to use previous tools like the PyETViewer.

    Args:
        res_fn (str): The path the the single residue ranks file to be converted.
        reverse_score (bool): If this flag is set to true, recompute scores so that each score is:
        score_new = 1 + |score_i - score_max|
        This is used to correct for metrics (like ET-MIp) where a higher score is better, but which does not interact
        correctly with PyETViewer, which imports rho or the score for residue selection and coloring and recomputes
        coverage on the fly.
    Return:
        str: The path to the converted output file.
    """
    res_dir = os.path.dirname(res_fn)
    res_base_name, res_base_extension = os.path.splitext(os.path.basename(res_fn))
    scoring_df = pd.read_csv(res_fn, sep='\t', header=0, index_col=None)
    assert {'Position', 'Query'}.issubset(scoring_df.columns), "Provided file does not include expected columns, make"\
                                                               " sure this is a single position analysis result!"
    if reverse_score:
        max_score = scoring_df['Score'].max()
        scoring_df['Reversed_Score'] = scoring_df['Score'].apply(lambda x: 1 + abs(x - max_score))
        scoring_df.rename(columns={'Score': 'Original Score'}, inplace=True)
        scoring_df.rename(columns={'Reversed_Score': 'Score'}, inplace=True)
    full_path = os.path.join(res_dir, f'{res_base_name}.legacy.ranks')
    with open(full_path, 'w') as handle:
        handle.write('% Note: in this file % is a comment sign.\n')
        handle.write(f'% This file converted from: {res_fn}\n')
        handle.write('%	 RESIDUE RANKS: \n')
        handle.write('% alignment#  residue#      type      rank              variability           rho     coverage\n')
        for ind in scoring_df.index:
            pos_str = str(scoring_df.loc[ind, 'Position'])
            aln_str = pos_str.rjust(10, ' ')
            res_str = aln_str
            type_str = scoring_df.loc[ind, 'Query'].rjust(10, ' ')
            rank_str = f'{scoring_df.loc[ind, "Score"]:.3f}'.rjust(10, ' ')
            var_count_str = str(scoring_df.loc[ind, 'Variability_Count']).rjust(10, ' ')
            var_char_str = scoring_df.loc[ind, 'Variability_Characters'].replace(',', '').rjust(22, ' ')
            rho_str = rank_str
            cov_str = f'{scoring_df.loc[ind, "Coverage"]:.3f}'.rjust(10, ' ')
            handle.write(aln_str + res_str + type_str + rank_str + var_count_str + var_char_str + rho_str + cov_str +
                         '\n')
    return full_path


def parse_args():
    """
    Parse Arguments

    This method processes the command line arguments for Evolutionary Trace.
    """
    # Create input parser
    parser = argparse.ArgumentParser(description='Process Evolutionary Trace options.')
    # Mandatory arguments (no defaults)
    parser.add_argument('--query', metavar='Q', type=str, nargs='?', required=True,
                        help='The name of the protein being queried in this analysis (should be the sequence identifier'
                             ' in the specified alignment).')
    parser.add_argument('--alignment', metavar='A', type=str, nargs='?', required=True,
                        help='The file path to the alignment to analyze (fasta formatted alignment expected).')
    # Optional argument (has default), which is not part of a preset.
    parser.add_argument('--output_dir', metavar='O', type=str, nargs='?',
                        default=os.getcwd(), help='File path to a directory where the results can be generated.')
    # Check if a preset has been selected.
    parser.add_argument('--preset', metavar='P', type=str, nargs='?', choices=['intET', 'rvET', 'ET-MIp', 'CovET'],
                        help='Specifying a preset will run a previously published Evolutionary Trace algorithm. '
                             'The current options are: "intET", "rvET", or "ET-MIp". Specifying any of these will '
                             'overwrite other options except "query", "alignment", "output_dir", "processes", and '
                             '"low_memory_off".')
    # Optional argument (has default), which is not part of a preset.
    parser.add_argument('--polymer_type', metavar='p', type=str, default='Protein', choices=['Protein', 'DNA'],
                        nargs='?', help='Which kind of sequence is being analyzed, currently accepted values are: '
                                        'Protein and DNA.')
    parser.add_argument('--distance_model', metavar='D', type=str, default='blosum62', nargs='?',
                        choices=['identity'] + DistanceCalculator.protein_models + DistanceCalculator.dna_models,
                        help=f'Which distance model to use, availability is limited depending on polymer type models ' \
                             f'available to: Both polymer types: [{", ".join(["identity"])}];Protein only models: ' \
                             f'[{", ".join(DistanceCalculator.protein_models)}]; DNA only models: ' \
                             f'[{", ".join(DistanceCalculator.dna_models)}]')
    parser.add_argument('--et_distance_off', default=True, action='store_false', dest='et_distance',
                        help='If this flag is provided the Evolutionary Trace distance computation over the specified '
                             'distance model (i.e. similarity using the specified distance model) will not be used.')
    parser.add_argument('--tree_building_method', metavar='T', type=str, default='et', nargs='?',
                        choices=['et', 'upgma', 'agglomerative', 'custom'],
                        help='Which tree building method to use when constructing the tree, options required for each '
                             'tree can be specified with the tree_building_options flag.')
    parser.add_argument('--tree_building_options', metavar='t', default=[], nargs='+',
                        help="Options are dependent on the tree_building_method specified options for each are listed "
                             "below. The name of the option should be provided first and then the desired value (e.g. "
                             "--tree_building_options tree_path /home/user/Documents/tree.nhx):\n\t'et\n\t\tNo "
                             "additional arguments.\n\t 'upgma'\n\t\tNo additional arguments.\n\t'agglomerative'\n\t\t"
                             "'cache_dir' (str): The path to the directory where the agglomerative clustering data can "
                             "be saved and loaded from.\n\t\t'affinity' (str): The affinity/distance calculation method"
                             "to use when operating on the distance values for clustering. Further details can be found"
                             " at: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClust"
                             "ering.html#sklearn.cluster.AgglomerativeClustering. The options are:\n\t\t\teuclidean "
                             "(default)\n\t\t\tl1\n\t\t\tl2\n\t\t\tmanhattan\n\t\t\tcosin\n\t\t\tprecomputed\n\t\t"
                             "'linkage' (str): The linkage algorithm to use when building the agglomerative clustering "
                             "tree structure. Further details can be found at: https://scikit-learn.org/stable/modules/"
                             "generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClust"
                             "ering. The options are:\n\t\t\tward (default)\n\t\t\tcomplete\n\t\t\taverage\n\t\t\t"
                             "single\n\t'custom'\n\t\ttree_path (str/path): The path to a file where the desired tree "
                             "has been written in 'newick' format.")
    parser.add_argument('--ranks', metavar='R', type=int, nargs='+', default=None,
                        help="Which ranks in the phylogenetic tree to use for scoring in the Evolutionary Trace "
                             "algorithm. The default behavior (specified by 0)  is to trace all ranks (levels) in the "
                             "tree. If not all ranks should be used, specify each rank to score (e.g. --ranks 1 2 3 5 7"
                             "10 25). No matter what, the first rank will always be scored, so if '--ranks 2 3 5' is"
                             "provided the actual input will be '--ranks 1 2 3 5'.")
    parser.add_argument('--position_type', metavar='S', type=str, nargs='?', default='single',
                        choices=['single', 'pair'],
                        help="Whether to score individual positions 'single' or pairs of positions 'pair'. This will "
                             "affect which scoring metrics can be selected.")
    parser.add_argument('--scoring_metric', metavar='S', type=str, default='identity', nargs='?',
                        choices=['identity', 'plain_entropy', 'mutual_information', 'normalized_mutual_information',
                                 'average_product_corrected_mutual_information',
                                 'filtered_average_product_corrected_mutual_information', 'match_count',
                                 'mismatch_count', 'match_mismatch_count_ratio', 'match_mismatch_count_angle',
                                 'match_entropy', 'mismatch_entropy', 'match_mismatch_entropy_ratio',
                                 'match_mismatch_entropy_angle', 'match_diversity', 'mismatch_diversity',
                                 'match_mismatch_diversity_ratio', 'match_mismatch_diversity_angle',
                                 'match_diversity_mismatch_entropy_ratio', 'match_diversity_mismatch_entropy_angle'],
                        help="Scoring metric to use when performing trace algorithm, method availability depends on "
                             "the specified position_type: single: [identity, plain_entropy]; pair: [identity, "
                             "plain_entropy, mutual_information, normalized_mutual_information, "
                             "average_product_corrected_mutual_information, "
                             "filtered_average_product_corrected_mutual_information, match_count, match_entropy, "
                             "match_diversity, mismatch_count, mismatch_entropy, mismatch_diversity, "
                             "match_mismatch_count_ratio, match_mismatch_entropy_ratio, match_mismatch_diversity_ratio"
                             "match_mismatch_count_angle, match_mismatch_entropy_angle, match_mismatch_diversity_angle"
                             "match_diversity_mismatch_entropy_ratio, match_diversity_mismatch_entropy_angle].")
    parser.add_argument('--gap_correction', metavar='G', type=float, default=None, nargs='?',
                        help="The fraction of rows in a column which should be a gap for a position to be correct. The"
                             "correction simply leads to that position being assigned the worst value found among all "
                             "positions.")
    parser.add_argument('--output_files', metavar='o', type=str, nargs='+', default='default',
                        choices=['original_aln', 'non-gap_aln', 'tree', 'sub-alignments', 'frequency_tables', 'scores',
                                 'single_pos_ranks', 'legacy', 'default'],
                        help="Which files to write to the provided output_dir. These can be specified one by one (e.g. "
                             "--output_files original_aln tree scores), or the option 'default' can be specified which "
                             "will result in 'original_aln', 'non-gap_aln', 'tree', 'scores', 'single_pos_scores' (for"
                             "position_type pair analyses), and 'legacy' being written to file.")
    parser.add_argument('--low_memory_off', default=True, action='store_false', dest='low_memory',
                        help="If this flag is specified the low memory option, which serializes intermediate data "
                             "(sub-alignment frequency tables, group scores, and rank scores) to file instead of "
                             "holding it in memory, will not be used. The default behavior is to use this option.")
    parser.add_argument('--processors', metavar='C', type=int, nargs='?', default=1,
                        help="The number of CPU cores available to this process while running (will determine the "
                             "number of processes forked by steps which can be completed using multiprocessing pools).")
    # Clean command line input
    arguments = parser.parse_args()
    arguments = vars(arguments)
    # If preset is chosen set all other
    if arguments['preset']:
        arguments['polymer_type'] = 'Protein'
        arguments['distance_model'] = 'blosum62'
        arguments['et_distance'] = True
        arguments['tree_building_method'] = 'et'
        arguments['tree_building_options'] = {}
        arguments['ranks'] = None
        arguments['output_files'] = 'default'
        if arguments['preset'] in ['ET-MIp', 'CovET']:
            arguments['position_type'] = 'pair'
            arguments['gap_correction'] = None
            if arguments['preset'] == 'ET-MIp':
                arguments['scoring_metric'] = 'filtered_average_product_corrected_mutual_information'
            else:
                arguments['scoring_metric'] = 'mismatch_diversity'
        else:
            arguments['position_type'] = 'single'
            if arguments['preset'] == 'intET':
                arguments['scoring_metric'] = 'identity'
                arguments['gap_correction'] = None
            else:
                arguments['scoring_metric'] = 'plain_entropy'
                arguments['gap_correction'] = 0.6
    # Process tree building options so they are formatted correctly.
    if arguments['tree_building_options']:
        arguments['tree_building_options'] = {arguments['tree_building_options'][i]:
                                                  arguments['tree_building_options'][i + 1]
                                              for i in range(0, len(arguments['tree_building_options']), 2)}
    else:
        arguments['tree_building_options'] = {}
    # Process output files:
    if arguments['output_files'] == 'default':
        arguments['output_files'] = {'original_aln', 'non_gap_aln', 'tree', 'scores', 'legacy'}
        if arguments['position_type'] == 'pair':
            arguments['output_files'] |= {'single_pos_scores'}
    else:
        arguments['output_files'] = set(arguments['output_files'])
    return arguments


if __name__ == "__main__":
    # Read input from the command line
    args = parse_args()
    # Initialize EvolutionaryTrace object
    et = EvolutionaryTrace(query=args['query'], polymer_type=args['polymer_type'], aln_file=args['alignment'],
                           et_distance=args['et_distance'], distance_model=args['distance_model'],
                           tree_building_method=args['tree_building_method'],
                           tree_building_options=args['tree_building_options'], ranks=args['ranks'],
                           position_type=args['position_type'], scoring_metric=args['scoring_metric'],
                           gap_correction=args['gap_correction'], out_dir=args['output_dir'],
                           output_files=args['output_files'], processors=args['processors'],
                           low_memory=args['low_memory'])
    # Compute distance matrix, construct tree, perform sequence assignments, trace, and write out final scores
    et.calculate_scores()

