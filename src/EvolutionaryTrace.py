"""
Created on August 10, 2019

@author: Daniel Konecki
"""
import os
import argparse
import numpy as np
import pandas as pd
from time import time
import cPickle as pickle
from Bio.Phylo.TreeConstruction import DistanceCalculator
from SupportingClasses.Trace import Trace, load_freq_table
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.FrequencyTable import FrequencyTable
from SupportingClasses.PhylogeneticTree import PhylogeneticTree
from SupportingClasses.PositionalScorer import PositionalScorer
from SupportingClasses.AlignmentDistanceCalculator import AlignmentDistanceCalculator


class EvolutionaryTrace(object):
    """
    This class draws on all supporting classes and methods to implement the full Evolutionary Trace algorithm.

    Attributes:
        query_id (str): The sequence identifier for the sequence being analyzed.
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
        score replaced with the worse observed score during the trace (down weighting all highly gapped columns). The
        default value for this has traditionally been set to 0.6 for rvET but has not been used for intET or ET-MIp.
        trace (Trace): A Trace object used to organize data required for performing a trace and the methods to do so.
        scorer (PositionalScorer): A PositionalScorer object which uses the specified scoring metric to score groups and
        takes the combined group scores at a given rank to calculate a final rank score.
        ranking (np.array): The rank (lowest being best, highest being worst) of each single or paired position in the
        provided alignment as determined from the calculated scores.
        scores (np.array): The raw scores calculated for each single or paired position in the provided alignment while
        performing the trace. For some method (identity, plain_entropy) the lower the score the better, while for others
        (mutual_information, normalized_mutual_information, average_product_corrected_mutual_information, and
        filtered_average_product_corrected_mutual_information) the higher the score the better.
        coverage (np.array): The percentage of scores at or better than the score for this single or paired position
        (i.e. the percentile rank).
        out_dir (str): The path where results of this analysis should be written to.
        output_files (set): Which files to write out, possible values include: 'original_aln', 'non-gap_aln', 'tree',
        'sub-alignment', 'frequency_tables', and 'scores'.
        processors (int): The number of CPU cores which this analysis can use while running.
        low_memory (bool): Whether or not to serialize files during execution in order to avoid keeping everything in
        memory (this is important for large alignments).
    """

    def __init__(self, query_id, polymer_type, aln_fn, et_distance, distance_model, tree_building_method,
                 tree_building_options, ranks, position_type, scoring_metric, gap_correction, out_dir, output_files,
                 processors, low_memory):
        """
        Initialization for EvolutionaryTrace class:

        Args:
            query_id (str): The sequence identifier for the sequence being analyzed.
            polymer_type (str): What kind of sequence information is being analyzed (.i.e. Protein or DNA).
            aln_fn (str): The path to the alignment to analyze.
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
        self.query_id = query_id
        self.polymer_type = polymer_type
        self.original_aln_fn = aln_fn
        self.original_aln = None
        self.et_distance = et_distance
        self.distance_model = distance_model
        self.distance_matrix = None
        self.tree_building_method = tree_building_method
        self.tree_building_options = tree_building_options
        self.phylo_tree = None
        self.phylo_tree_fn = None
        self.ranks = ranks
        self.assignments = None
        self.non_gapped_aln_fn = None
        self.non_gapped_aln = None
        self.position_type = position_type
        self.scoring_metric = scoring_metric
        self.gap_correction = gap_correction
        self.trace = None
        self.ranking = None
        self.scorer = None
        self.scores = None
        self.coverage = None
        self.out_dir = out_dir
        self.output_files = output_files
        self.processors = processors
        self.low_memory = low_memory

    def import_and_process_aln(self):
        """
        Import and Processes Alignment

        This function checks for the specified alignment then imports it. If specified the original alignment is written
        out the output directory. A second alignment is then created with all columns which are gaps in the query
        sequence removed, and if specified this is written to the output directory as well.
        """
        serial_fn = os.path.join(self.out_dir, '{}_alignments.pkl'.format(self.query_id))
        aln_base_name, _ = os.path.splitext(os.path.basename(self.original_aln_fn))
        if os.path.isfile(serial_fn):
            with open(serial_fn, 'rb') as handle:
                self.original_aln, self.non_gapped_aln_fn, self.non_gapped_aln = pickle.load(handle)
        else:
            if not os.path.isfile(self.original_aln_fn):
                raise ValueError('Provided alignment path is not valid: {}'.format(self.original_aln_fn))
            self.original_aln = SeqAlignment(file_name=self.original_aln_fn, query_id=self.query_id,
                                             polymer_type=self.polymer_type)
            self.original_aln.import_alignment()
            self.non_gapped_aln_fn = os.path.join(self.out_dir, '{}_non_gapped.fa'.format(aln_base_name))
            self.non_gapped_aln = self.original_aln.remove_gaps()
            with open(serial_fn, 'wb') as handle:
                pickle.dump((self.original_aln, self.non_gapped_aln_fn, self.non_gapped_aln), handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        if 'original_aln' in self.output_files:
            self.original_aln.write_out_alignment(file_name=os.path.join(self.out_dir, '{}_original.fa'.format(
                aln_base_name)))
        if 'non_gap_aln' in self.output_files:
            self.non_gapped_aln.write_out_alignment(file_name=self.non_gapped_aln_fn)

    def compute_distance_matrix_tree_and_assignments(self):
        """
        Compute Distance Matrix, Tree, and Assignments

        This function computes the distance matrix based on the original alignment provided using the specified distance
        model (and if specified the Evolutionary Trace distance calculation method). This distance matrix is then used
        to build a phylogenetic tree using the specified tree_building_method and tree_building_options. If specified
        the tree is written to the output directory (in Newick/nhx format). Finally, nodes from the tree are assigned to
        ranks and groups.
        """
        serial_fn = '{}_{}{}_Dist_{}_Tree.pkl'.format(self.query_id, ('ET_' if self.et_distance else ''),
                                                      self.distance_model, self.tree_building_method)
        serial_fn = os.path.join(self.out_dir, serial_fn)
        if os.path.isfile(serial_fn):
            with open(serial_fn, 'rb') as handle:
                self.distance_matrix, self.phylo_tree, self.phylo_tree_fn, self.assignments = pickle.load(handle)
        else:
            calculator = AlignmentDistanceCalculator(protein=(self.polymer_type == 'Protein'), model=self.distance_model,
                                                     skip_letters=None)
            if self.et_distance:
                _, self.distance_matrix, _, _ = calculator.get_et_distance(self.original_aln.alignment)
            else:
                self.distance_matrix = calculator.get_distance(self.original_aln.alignment)
            self.phylo_tree = PhylogeneticTree(tree_building_method=self.tree_building_method,
                                               tree_building_args=self.tree_building_options)
            self.phylo_tree.construct_tree(dm=self.distance_matrix)
            self.phylo_tree.rename_internal_nodes()
            self.phylo_tree_fn = os.path.join(self.out_dir, '{}_{}{}_dist_{}_tree.nhx'.format(
                self.query_id, ('ET_' if self.et_distance else ''), self.distance_model, self.tree_building_method))
            self.assignments = self.phylo_tree.assign_group_rank(ranks=self.ranks)
            with open(serial_fn, 'wb') as handle:
                pickle.dump((self.distance_matrix, self.phylo_tree, self.phylo_tree_fn, self.assignments), handle,
                            pickle.HIGHEST_PROTOCOL)
        if 'tree' in self.output_files:
            self.phylo_tree.write_out_tree(filename=self.phylo_tree_fn)

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
            self.query_id, ('ET_' if self.et_distance else ''), self.distance_model, self.tree_building_method,
            ('All_Ranks' if self.ranks is None else 'Custom_Ranks'), self.scoring_metric)
        serial_fn = os.path.join(self.out_dir, serial_fn)
        if os.path.isfile(serial_fn):
            with open(serial_fn, 'rb') as handle:
                self.trace, self.ranking, self.scores, self.coverage = pickle.load(handle)
        else:
            self.trace = Trace(alignment=self.non_gapped_aln, phylo_tree=self.phylo_tree,
                               group_assignments=self.assignments, position_specific=(self.position_type == 'single'),
                               pair_specific=(self.position_type == 'pair'), output_dir=self.out_dir,
                               low_memory=self.low_memory)
            self.trace.characterize_rank_groups(processes=self.processors,
                                                write_out_sub_aln='sub-alignments' in self.output_files,
                                                write_out_freq_table='frequency_tables' in self.output_files)
            self.scorer = PositionalScorer(seq_length=self.non_gapped_aln.seq_length,
                                           pos_size=(1 if self.position_type == 'single' else 2),
                                           metric=self.scoring_metric)
            self.ranking, self.scores, self.coverage = self.trace.trace(scorer=self.scorer, processes=self.processors,
                                                                        gap_correction=self.gap_correction)
            with open(serial_fn, 'wb') as handle:
                pickle.dump((self.trace, self.ranking, self.scores, self.coverage), handle, pickle.HIGHEST_PROTOCOL)
        root_node_name = self.assignments[1][1]['node'].name
        root_freq_table = self.trace.unique_nodes[root_node_name][self.position_type.lower()]
        root_freq_table = load_freq_table(freq_table=root_freq_table, low_memory=self.low_memory)
        # Generate descriptive file name
        rank_fn = '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.ranks'.format(
            self.query_id, ('ET_' if self.et_distance else ''), self.distance_model, self.tree_building_method,
            ('All_Ranks' if self.ranks is None else 'Custom_Ranks'), self.scoring_metric)
        # rank_fn = '{}_{}'.format(self.query_id, ('ET_' if self.et_distance else ''), )
        # rank_fn += '{}_Dist_{}_Tree_'.format(self.distance_model, self.tree_building_method)
        # if self.ranks is None:
        #     rank_fn += 'All_Ranks_'
        # else:
        #     rank_fn += 'Custom_Ranks_'
        # rank_fn += '{}_Scoring.ranks'.format(self.scoring_metric)
        write_out_et_scores(file_name=rank_fn, out_dir=self.out_dir, aln=self.non_gapped_aln,
                            freq_table=root_freq_table, ranks=self.ranking, scores=self.scores, coverages=self.coverage,
                            precision=3)


def write_out_et_scores(file_name, out_dir, aln, freq_table, ranks, scores, coverages, precision=3):
    """
    Write Out Evolutionary Trace Scores

    This method writes out the results of the Evolutionary Trace analysis.

    Args:
        file_name (str): The name to write the results to.
        out_dir (str): The directory to write the results file to.
        aln (SeqAlignment): The non-gapped sequence alignment used to perform the trace which is being written out.
        freq_table (FrequencyTable): The characterization of the root node of the phylogenetic tree (full alignment).
        ranks (np.array): The ranking of each position analyzed in the trace.
        scores (np.array): The score for each position analyzed in the trace.
        coverages (np.array): The coverage for each position analyzed in the trace.
        precision (int): The number of decimal places to write out for floating point values such coverages (and scores
        if a real valued scoring metric was used).
    """
    full_path = os.path.join(out_dir, file_name)
    if os.path.isfile(full_path):
        print('Evolutionary Trace analysis with the same parameters already saved to this location.')
        return
    start = time()
    scoring_dict = {}
    columns = []
    if freq_table.position_size == 1:
        scoring_dict['Position'] = list(range(1, et.non_gapped_aln.seq_length + 1))
        scoring_dict['Query'] = list(str(et.non_gapped_aln.query_sequence))
        relevant_table = freq_table.get_table()
        positions = relevant_table.nonzero()
        scoring_dict['Variability_Count'] = list(np.array((relevant_table > 0).sum(axis=1)).reshape(-1))
        scoring_dict['Variability_Characters'] = [','.join([freq_table.reverse_mapping[x]
                                                            for x in positions[1][positions[0] == i]])
                                                  for i in range(relevant_table.shape[0])]
        scoring_dict['Rank'] = list(et.ranking)
        scoring_dict['Score'] = list(et.scores)
        scoring_dict['Coverage'] = list(et.coverage)
        columns += ['Position', 'Query']
    elif freq_table.position_size == 2:
        off_diagonal_indices = []
        starting_index = 1
        pos_i, pos_j, query_i, query_j = [], [], [], []
        for x in range(1, et.non_gapped_aln.seq_length + 1):
            off_diagonal_indices += list(range(starting_index, starting_index + (et.non_gapped_aln.seq_length - x)))
            pos_i += [x] * (et.non_gapped_aln.seq_length - x)
            query_i += [et.non_gapped_aln.query_sequence[x - 1]] * (et.non_gapped_aln.seq_length - x)
            pos_j += list(range(x + 1, et.non_gapped_aln.seq_length + 1))
            query_j += list(et.non_gapped_aln.query_sequence[x:])
            starting_index = off_diagonal_indices[-1] + 2
        print(freq_table.get_table()[off_diagonal_indices, :].nonzero())
        scoring_dict['Position_i'] = pos_i
        scoring_dict['Position_j'] = pos_j
        scoring_dict['Query_i'] = query_i
        scoring_dict['Query_j'] = query_j
        relevant_table = freq_table.get_table()[off_diagonal_indices, :]
        positions = relevant_table.nonzero()
        scoring_dict['Variability_Count'] = list(np.array((relevant_table > 0).sum(axis=1)).reshape(-1))
        scoring_dict['Variability_Characters'] = [','.join([freq_table.reverse_mapping[x]
                                                            for x in positions[1][positions[0] == i]])
                                                  for i in range(relevant_table.shape[0])]
        scoring_dict['Rank'] = list(et.ranking[np.triu_indices(et.non_gapped_aln.seq_length, k=1)])
        scoring_dict['Score'] = list(et.scores[np.triu_indices(et.non_gapped_aln.seq_length, k=1)])
        scoring_dict['Coverage'] = list(et.coverage[np.triu_indices(et.non_gapped_aln.seq_length, k=1)])
        columns += ['Position_i', 'Position_j', 'Query_i', 'Query_j']
    else:
        raise ValueError("write_out_et_scores is not implemented to work with scoring for position sizes other than 1 "
                         "or 2.")
    columns += ['Variability_Count', 'Variability_Characters', 'Rank', 'Score', 'Coverage']
    scoring_df = pd.DataFrame(scoring_dict)
    scoring_df.to_csv(full_path, sep='\t', header=True, index=False, float_format='%.{}f'.format(precision),
                      columns=columns)
    end = time()
    print('Results written to file in {} min'.format((end - start) / 60.0))


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
    parser.add_argument('--preset', metavar='P', type=str, nargs='?', choices=['intET', 'rvET', 'ET-MIp'],
                        help='Preset Evolutionary Action methodology which has been used and characterized in the past '
                        'current options are: intET, rvET, or ET-MIp')
    # Optional argument (has default), which is not part of a preset.
    parser.add_argument('--polymer_type', metavar='p', type=str, default='Protein', choices=['Protein', 'DNA'],
                        nargs='?', help='Which kind of sequence is being analyzed, currently accepted values are: '
                                        'Protein and DNA.')
    parser.add_argument('--distance_model', metavar='D', type=str, default='blosum62', nargs='?',
                        choices=['identity'] + DistanceCalculator.protein_models + DistanceCalculator.dna_models,
                        help='Which distance model to use, availability is limited depending on polymer type models '
                             'available to both polymer types:\n{}to protein models:\n{}to DNA models:\n{}'.format(
                            '\n'.join(['identity']), '\n'.join(DistanceCalculator.protein_models),
                            '\n'.join(DistanceCalculator.dna_models)))
    parser.add_argument('--et_distance', default=True, action='store_false',
                        help='Whether or not to use the Evolutionary Trace distance computation over the specified '
                             'distance model (i.e. similarity using the specified distance model).')
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
                                 'filtered_average_product_corrected_mutual_information'],
                        help="Scoring metric to use when performing trace algorithm, method availability depends on "
                             "the specified position_type:\n\tsingle:\n\t\tidentity\n\t\tplain_entropy\n\tpair:\n\t\t"
                             "identity\n\t\tplain_entropy\n\t\tmutual_information\n\t\tnormalized_mutual_information"
                             "\n\t\taverage_product_corrected_mutual_information\n\t\t"
                             "filtered_average_product_corrected_mutual_information")
    parser.add_argument('--gap_correction', metavar='G', type=float, default=None, nargs='?',
                        help="The fraction of rows in a column which should be a gap for a position to be correct. The"
                             "correction simply leads to that position being assigned the worst value found among all "
                             "positions.")
    parser.add_argument('--output_files', metavar='o', type=str, nargs='+', default='default',
                        choices=['original_aln', 'non-gap_aln', 'tree', 'sub-alignments', 'frequency_tables', 'scores',
                                 'default'],
                        help="Which files to write to the provided output_dir. These can be specified one by one (e.g. "
                             "--output_files original_aln tree scores), or the option 'default' can be specified which "
                             "will result in 'original_aln', 'non-gap_aln', 'tree', and 'scores' being written to "
                             "file.")
    parser.add_argument('--low_memory', default=True, action='store_false',
                        help="Whether or not to use the low memory option which will serialize intermediate data "
                             "(sub-alignment frequency tables, group scores, and rank scores) to file instead of "
                             "holding it in memory. The default behavior is to use this option, specifying this flag "
                             "turns it off.")
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
        if arguments['preset'] == 'ET-MIp':
            arguments['position_type'] = 'pair'
            arguments['scoring_metric'] = 'filtered_average_product_corrected_mutual_information'
            arguments['gap_correction'] = None
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
        arguments['output_files'] = {'original_aln', 'non_gap_aln', 'tree', 'scores'}
    else:
        arguments['output_files'] = set(arguments['output_files'])
    return arguments


if __name__ == "__main__":
    # Read input from the command line
    args = parse_args()
    print(args)
    # Initialize EvolutionaryTrace object
    et = EvolutionaryTrace(query_id=args['query'], polymer_type=args['polymer_type'], aln_fn=args['alignment'],
                           et_distance=args['et_distance'], distance_model=args['distance_model'],
                           tree_building_method=args['tree_building_method'],
                           tree_building_options=args['tree_building_options'], ranks=args['ranks'],
                           position_type=args['position_type'], scoring_metric=args['scoring_metric'],
                           gap_correction=args['gap_correction'], out_dir=args['output_dir'],
                           output_files=args['output_files'], processors=args['processors'],
                           low_memory=args['low_memory'])
    # Import and process the alignment provided.
    et.import_and_process_aln()
    # Compute distance matrix, construct tree, and perform sequence assignments
    et.compute_distance_matrix_tree_and_assignments()
    # Perform the trace and write out final scores
    et.perform_trace()

