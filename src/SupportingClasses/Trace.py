"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep, time
from Bio.Alphabet import Gapped
from multiprocessing import Manager, Pool, Lock
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.FrequencyTable import FrequencyTable
from SupportingClasses.PhylogeneticTree import PhylogeneticTree
from SupportingClasses.EvolutionaryTraceAlphabet import MultiPositionAlphabet
from SupportingClasses.utils import gap_characters, build_mapping, compute_rank_and_coverage


class Trace(object):
    """
    This class represents the fundamental behavior of the Evolutionary Trace algorithm developed by Olivier Lichtarge
    and expanded upon by members of his laboratory.

    Attributes:
        aln (SeqAlignment): The alignment for which a trace is being performed
        phylo_tree (PhylogeneticTree): The tree constructed from the alignment over which the trace is performed.
        assignments (dict): The rank and group assignments made based on the tree.
        unique_nodes (dict): A dictionary to track the unique nodes from the tree, this will be used for
        characterization of sub-alignments and group scoring during the trace procedure to reduce the required
        computations.
        rank_scores (dict): A dictionary to track the the scores at each rank in the assignments dictionary.
        match_mismatch (bool): Whether to use the match mismatch characterization (looking at all possible transitions
        in a position/pair between sequences of an alignment, True) or the standard protocol (looking at each position
        only once for each sequence in the alignment, False).
        out_dir (str/path): Where results from a trace should be stored.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
    """

    def __init__(self, alignment, phylo_tree, group_assignments, pos_size, match_mismatch=False, output_dir=None,
                 low_memory=False):
        """
        Initializer for Trace object.

        Args:
            alignment (SeqAlignment): The alignment for which to perform a trace analysis.
            phylo_tree (PhylogeneticTree): The tree based on the alignment to use during the trace analysis.
            group_assignments (dict): The group assignments for nodes in the tree.
            pos_size (int): The size of the the positions being analyzed (expecting 1 single position scores or 2 pair
            position scores).
            match_mismatch (bool): Whether to use the match mismatch characterization (True) or the standard protocol
            (False).
            output_dir (str/path): Where results from a trace should be stored.
            low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
            resources.
        """
        if not isinstance(alignment, SeqAlignment):
            raise ValueError('Provided alignment must be a SeqAlignment object!')
        self.aln = alignment
        # This check is being deprecated because it fails depending on how the PhylogeneticTree class is imported.
        # if not isinstance(phylo_tree, PhylogeneticTree):
        if type(phylo_tree).__name__ != 'PhylogeneticTree':
            print(type(phylo_tree))
            print(type(phylo_tree).__name__)
            raise ValueError('Provided tree must be a PhylogeneticTree object!')
        self.phylo_tree = phylo_tree
        if not isinstance(group_assignments, dict):
            raise ValueError('Provided assignments must be a dictionary produced by '
                             'PhylogeneticTree.assign_group_rank()!')
        self.assignments = group_assignments
        if (not isinstance(match_mismatch, bool)) or (not isinstance(low_memory, bool)):
            raise ValueError('Input of type bool is expected for arguments match_mismatch, and low_memory!')
        if (pos_size < 1) or (pos_size > 2):
            raise ValueError('Currently the only supported positions sizes are 1 (position specific) and 2 (pairs of '
                             'positions)!')
        self.pos_size = pos_size
        self.match_mismatch = match_mismatch
        self.low_memory = low_memory
        if output_dir is None:
            self.out_dir = os.getcwd()
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            self.out_dir = output_dir
        self.unique_nodes = None
        self.rank_scores = None
        self.final_scores = None
        self.final_ranks = None
        self.final_coverage = None

    def characterize_rank_groups_standard(self, unique_dir, alpha_size, alpha_mapping, alpha_reverse,
                                          single_to_pair=None, processes=1, write_out_sub_aln=True,
                                          write_out_freq_table=True, maximum_iterations=100000, single_map=None):
        """
        Characterize Rank Group

        This function iterates over the rank and group assignments and characterizes all positions for each sub
        alignment. Characterization consists of FrequencyTable objects which are added to the group_assignments
        dictionary provided at initialization (FrequencyTables are added under the key 'freq_table').

        Args:
            unique_dir (str/path): The path to the directory where unique node data can be stored.
            alpha_size (int): The size of the alphabet needed for the alignment and position size being characterized
            (including the gap character).
            alpha_mapping (dict): A dictionary mapping the letters of the alphabet to numerical positions.
            alpha_reverse (np.array): An array mapping numerical positions back to the letter alphabet of the alphabet.
            single_to_pair (np.array): A two dimensional array mapping single letter numeric positions to paired letter
            numeric positions. Only needed for pairs of positions.
            processes (int): The maximum number of sequences to use when performing this characterization.
            write_out_sub_aln (bool): Whether or not to write out the sub-alignments for each node while characterizing
            it.
            write_out_freq_table (bool): Whether or not to write out the frequency table for each node while
            characterizing it.
            maximum_iterations (int): The most attempts that can be made to retrieve a single node descendant.
        """
        visited = {}
        components = False
        to_characterize = []
        for r in sorted(self.assignments.keys(), reverse=True):
            for g in self.assignments[r]:
                node = self.assignments[r][g]['node']
                if not components:
                    to_characterize.append((node.name, 'component'))
                elif node.name not in visited:
                    to_characterize.append((node.name, 'inner'))
                else:
                    continue
                visited[node.name] = {'terminals': self.assignments[r][g]['terminals'],
                                      'descendants': self.assignments[r][g]['descendants']}
            if not components:
                components = True
        characterization_pbar = tqdm(total=len(to_characterize), unit='characterizations')

        def update_characterization(return_name):
            """
            Update Characterization

            This function serves to update the progress bar for characterization.

            Args:
                return_name (str): The name of the node which has just finished being characterized.
            """
            characterization_pbar.update(1)
            characterization_pbar.refresh()

        pool_manager = Manager()
        frequency_tables = pool_manager.dict()
        tables_lock = Lock()
        pool = Pool(processes=processes, initializer=init_characterization_pool,
                    initargs=(alpha_size, alpha_mapping, alpha_reverse, single_to_pair, self.aln, self.pos_size,
                              visited, frequency_tables, tables_lock, unique_dir, self.low_memory, write_out_sub_aln,
                              write_out_freq_table, processes, maximum_iterations, single_map))
        for char_node in to_characterize:
            pool.apply_async(func=characterization, args=char_node, callback=update_characterization)
        pool.close()
        pool.join()
        characterization_pbar.close()
        frequency_tables = dict(frequency_tables)
        if len(frequency_tables) != len(to_characterize):
            raise ValueError('Characterization incomplete, please check inputs provided!')
        self.unique_nodes = frequency_tables

    def characterize_rank_groups_match_mismatch(self, unique_dir, single_size, single_mapping, processes=1,
                                                write_out_sub_aln=True, write_out_freq_table=True,
                                                maximum_iterations=100000):
        """
        Characterize Rank Groups Match Mismatch

        This function iterates over the rank and group assignments and characterizes all positions for each sub
        alignment. Characterization consists of FrequencyTable objects which are added to the group_assignments
        dictionary provided at initialization (the match FrequencyTable can be found with the key 'match' and the
        mismatch FrequencyTable with the key 'mismatch').

        Args:
            unique_dir (str/path): The path to the directory where unique node data can be stored.
            single_size (int): The size of the single letter alphabet being used.
            single_mapping (dict): A dictionary from character to integer representation.
            processes (int): The maximum number of sequences to use when performing this characterization.
            write_out_sub_aln (bool): Whether or not to write out the sub-alignments for each node while characterizing
            it.
            write_out_freq_table (bool): Whether or not to write out the frequency table for each node while
            characterizing it.
            maximum_iterations (int): The most attempts that can be made to retrieve a single node descendant.
        """
        pair_alphabet = MultiPositionAlphabet(alphabet=Gapped(self.aln.alphabet), size=2)
        pair_size, _, pair_mapping, pair_revers = build_mapping(alphabet=pair_alphabet)
        single_to_pair = np.zeros((single_size + 1, single_size + 1), dtype=np.int32)
        mismatch_mask = np.zeros(pair_size, dtype=np.bool_)
        for char in pair_mapping:
            single_to_pair[single_mapping[char[0]], single_mapping[char[1]]] = pair_mapping[char]
            if single_mapping[char[0]] != single_mapping[char[1]]:
                mismatch_mask[pair_mapping[char]] = True
        if self.pos_size == 1:
            larger_size, larger_mapping, larger_reverse = pair_size, pair_mapping, pair_revers
            comparison_mapping = single_to_pair
            single_to_pair = None
        elif self.pos_size == 2:
            quad_alphabet = MultiPositionAlphabet(alphabet=Gapped(self.aln.alphabet), size=4)
            larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=quad_alphabet)
            # single_to_pair set above
            comparison_mapping = np.zeros((pair_size + 1, pair_size + 1), dtype=np.int32)
            mismatch_mask = np.zeros(larger_size, dtype=np.bool_)
            for char in larger_mapping:
                comparison_mapping[pair_mapping[char[:2]], pair_mapping[char[2:]]] = larger_mapping[char]
                if ((char[0] != char[2]) and (char[1] == char[3])) or ((char[0] == char[2]) and (char[1] != char[3])):
                    mismatch_mask[larger_mapping[char]] = True
        else:
            raise AttributeError('Match/Mismatch characterization requires that either pos_specific or pair_'
                                 'specific be selected but not both.')
        visited = {}
        components = False
        to_characterize = []
        for r in sorted(self.assignments.keys(), reverse=True):
            for g in self.assignments[r]:
                node = self.assignments[r][g]['node']
                if not components:
                    to_characterize.append((node.name, 'component'))
                elif node.name not in visited:
                    to_characterize.append((node.name, 'inner'))
                else:
                    continue
                visited[node.name] = {'terminals': self.assignments[r][g]['terminals'],
                                      'descendants': self.assignments[r][g]['descendants']}
            if not components:
                components = True
        characterization_pbar = tqdm(total=len(to_characterize), unit='characterizations')

        def update_characterization(return_name):
            """
            Update Characterization

            This function serves to update the progress bar for characterization.

            Args:
                return_name (str): The name of the node which has just finished being characterized.
            """
            characterization_pbar.update(1)
            characterization_pbar.refresh()

        pool_manager = Manager()
        frequency_tables = pool_manager.dict()
        tables_lock = Lock()
        pool = Pool(processes=processes, initializer=init_characterization_mm_pool,
                    initargs=(single_mapping, larger_size, larger_mapping, larger_reverse, single_to_pair,
                              comparison_mapping, mismatch_mask, self.aln, self.pos_size, visited, frequency_tables,
                              tables_lock, unique_dir, self.low_memory, write_out_sub_aln, write_out_freq_table,
                              maximum_iterations))
        for char_node in to_characterize:
            pool.apply_async(func=characterization_mm, args=char_node, callback=update_characterization)
        pool.close()
        pool.join()
        characterization_pbar.close()
        frequency_tables = dict(frequency_tables)
        if len(frequency_tables) != len(to_characterize):
            raise ValueError('Characterization incomplete, please check inputs provided!')
        self.unique_nodes = frequency_tables

    def characterize_rank_groups(self, processes=1, maximum_iterations=100000, write_out_sub_aln=True,
                                 write_out_freq_table=True):
        """
        Characterize Rank Groups:

        This function acts as a switch statement to use the appropriate characterization function (either standard or
        match mismatch).

        Arguments:
            processes (int): The maximum number of sequences to use when performing this characterization.
            maximum_iterations (int): The most attempts that can be made to retrieve a single node descendant.
            write_out_sub_aln (bool): Whether or not to write out the sub-alignments for each node while characterizing
            it.
            write_out_freq_table (bool): Whether or not to write out the frequency table for each node while
            characterizing it.
        """
        unique_dir = os.path.join(self.out_dir, 'unique_node_data')
        if not os.path.isdir(unique_dir):
            os.makedirs(unique_dir)
        single_alphabet = Gapped(self.aln.alphabet)
        single_size, _, single_mapping, single_reverse = build_mapping(alphabet=single_alphabet)
        if self.match_mismatch:
            self.characterize_rank_groups_match_mismatch(unique_dir=unique_dir, single_size=single_size,
                                                         single_mapping=single_mapping,  processes=processes,
                                                         write_out_sub_aln=write_out_sub_aln,
                                                         write_out_freq_table=write_out_freq_table,
                                                         maximum_iterations=maximum_iterations)
        else:
            if self.pos_size == 2:
                pair_alphabet = MultiPositionAlphabet(alphabet=Gapped(self.aln.alphabet), size=2)
                curr_size, _, curr_mapping, curr_reverse = build_mapping(alphabet=pair_alphabet)
                curr_single_to_pair = pro_single_to_pair = np.zeros((max(single_mapping.values()) + 1,
                                                                     max(single_mapping.values()) + 1), dtype=np.int)
                for char in curr_mapping:
                    pro_single_to_pair[single_mapping[char[0]], single_mapping[char[1]]] = curr_mapping[char]
            else:
                curr_size = single_size
                curr_mapping = single_mapping
                curr_reverse = single_reverse
                curr_single_to_pair = None
            self.characterize_rank_groups_standard(unique_dir=unique_dir, alpha_size=curr_size,
                                                   alpha_mapping=curr_mapping, alpha_reverse=curr_reverse,
                                                   single_to_pair=curr_single_to_pair, processes=processes,
                                                   write_out_sub_aln=write_out_sub_aln,
                                                   write_out_freq_table=write_out_freq_table,
                                                   maximum_iterations=maximum_iterations, single_map=single_mapping)

    def trace(self, scorer, gap_correction=0.6, processes=1):
        """
        Trace

        The central part of the Evolutionary Trace algorithm. This function is a multiprocessed version of the trace
        process which scores sub alignments for groups within each rank of a phylogenetic tree and then combines them to
        generate a rank score. Rank scores are combined to calculate a final importance score for each position (or pair
        of positions). The group and rank level scores can be computed based on different metrics which are dictated by
        the provided PositionalScorer object.

        Args:
            scorer (PositionalScorer): The scoring object to use when computing the group and rank scores.
            gap_correction (float): If this value is set then after the trace has been performed positions in the
            alignment where the gap / alignment size ratio is greater than the gap_correction will have their final
            scores set to the highest (least important) value computed up to that point. If you do not want to perform
            this correction please set gap_correction to None.
            processes (int): The maximum number of sequences to use when performing this characterization.
        Returns:
            np.array: A properly dimensioned vector/matrix/tensor of ranks for each position in the alignment being
            traced.
            np.array: A properly dimensioned vector/matrix/tensor of importance scores for each position in the
            alignment being traced.
            np.array: A properly dimensioned vector/matrix/tensor of coverage scores for each position in the
            alignment being traced.
        pseudocode:
            final_scores = np.zeros(scorer.dimensions)
            for rank in sorted(self.assignments.keys()):
                group_scores = []
                for group in sorted(self.assignments[rank].keys()):
                    group_scores.append(scorer.score_group(self.assignments[rank][group][pos_type]))
                group_scores = np.stack(group_scores, axis=0)
                rank_scores = scorer.score_rank(group_scores)
                final_scores += rank_scores
            final_scores += 1
        """
        if scorer.position_size != self.pos_size:
            raise ValueError('Scorer and Trace position size do not agree!')
        unique_dir = os.path.join(self.out_dir, 'unique_node_data')
        # Generate group scores for each of the unique_nodes from the phylo_tree
        group_pbar = tqdm(total=len(self.unique_nodes), unit='group')

        def update_group(return_tuple):
            """
            Update Group

            This function serves to update the progress bar for group scoring. It also updates the unique_nodes
            dictionary with the returned group scores for a given node.

            Args:
                return_tuple (tuple): A tuple consisting of the name of the node (str) which has been scored and a
                dictionary containing the single and pair position scores for that node.
            """
            completed_name, completed_components = return_tuple
            self.unique_nodes[completed_name]['group_scores'] = completed_components
            group_pbar.update(1)
            group_pbar.refresh()

        pool1 = Pool(processes=processes, initializer=init_trace_groups,
                     initargs=(scorer, self.match_mismatch, self.unique_nodes, self.low_memory, unique_dir))
        for node_name in self.unique_nodes:
            pool1.apply_async(trace_groups, (node_name,), callback=update_group)
        pool1.close()
        pool1.join()
        group_pbar.close()
        # For each rank collect all group scores and compute a final rank score
        self.rank_scores = {}
        rank_pbar = tqdm(total=len(self.assignments), unit='rank')

        def update_rank(return_tuple):
            """
            Update Rank

            This function serves to update the progress bar for rank scoring. It also updates the rank_scores
            dictionary with the returned rank scores for a given rank.

            Args:
                return_tuple (tuple): A tuple consisting of the rank (int) which has been scored and a dictionary
                containing the single and pair position scores for that ranks.
            """
            rank, component = return_tuple
            self.rank_scores[rank] = component
            rank_pbar.update(1)
            rank_pbar.refresh()

        pool2 = Pool(processes=processes, initializer=init_trace_ranks,
                     initargs=(scorer, self.assignments, self.unique_nodes, self.low_memory, unique_dir))
        for rank in sorted(self.assignments.keys(), reverse=True):
            pool2.apply_async(trace_ranks, (rank,), callback=update_rank)
        pool2.close()
        pool2.join()
        rank_pbar.close()
        if len(self.rank_scores) < len(self.assignments):
            raise ValueError('Trace incomplete, check initialization and/or input variables.')
        # Combine rank scores to generate a final score for each position
        final_scores = np.zeros(scorer.dimensions)
        for rank in self.rank_scores:
            rank_scores = load_numpy_array(mat=self.rank_scores[rank], low_memory=self.low_memory)
            final_scores += rank_scores
        if scorer.rank_type == 'min':
            if scorer.position_size == 1:
                final_scores += 1
            elif scorer.position_size == 2:
                final_scores += np.triu(np.ones((self.aln.seq_length, self.aln.seq_length)), k=1)
        # Perform gap correction, if a column gap content threshold has been set.
        if gap_correction is not None:
            pos_type = 'single' if scorer.position_size == 1 else 'pair'
            # The FrequencyTable for the root node is used because it characterizes all sequences and its depth is equal
            # to the alignment size. The resulting gap frequencies are equivalent to gap count / alignment size.
            root_node_name = self.phylo_tree.tree.root.name
            if self.match_mismatch:
                freq_table = (load_freq_table(freq_table=self.unique_nodes[root_node_name]['match'],
                                              low_memory=self.low_memory) +
                              load_freq_table(freq_table=self.unique_nodes[root_node_name]['mismatch'],
                                              low_memory=self.low_memory))
            else:
                freq_table = load_freq_table(freq_table=self.unique_nodes[root_node_name]['freq_table'],
                                             low_memory=self.low_memory)
            resized_gap_characters = set([(x * len(freq_table.reverse_mapping[0])) for x in list(gap_characters)])
            gap_chars = list(set(freq_table.reverse_mapping).intersection(resized_gap_characters))
            if len(gap_chars) > 1:
                raise ValueError('More than one gap character present in alignment alphabet! {}'.format(gap_chars))
            gap_char = gap_chars[0]
            if scorer.rank_type == 'min':
                worst_rank_score = np.max(final_scores)
            else:
                worst_rank_score = np.min(final_scores[final_scores != 0])
            for i in freq_table.get_positions():
                if freq_table.get_frequency(pos=i, char=gap_char) > gap_correction:
                    if self.pos_size == 1:
                        final_scores[i] = worst_rank_score
                    elif self.pos_size == 2 and i[0] != i[1]:
                        final_scores[i[0], i[1]] = worst_rank_score
                    else:
                        pass
        self.final_scores = final_scores
        self.final_ranks, self.final_coverage = compute_rank_and_coverage(
            seq_length=self.aln.seq_length, scores=self.final_scores, pos_size=scorer.position_size,
            rank_type=scorer.rank_type)
        return self.final_ranks, self.final_scores, self.final_coverage


def check_freq_table(low_memory, node_name, table_type, out_dir):
    """
    Check Frequency Table

    This function is used to check whether the desired FrequencyTable has already been produced and stored or not.

    Args:
        low_memory (bool): Whether or not low_memory mode is active (should serialized even files exist?).
        node_name (str): A string which will be used in identifying the save file.
        table_type (str): Whether the FrequencyTable being checked for is for single or paired positions.
        out_dir (str): The path to a directory where the FrequencyTable could have been saved.
    Returns:
        bool: Whether or not the desired table has previously been saved (if low_memory is False, False is returned by
        default).
        str/None: If low_memory is True the expected path to the file, whether it is present or not.
    """
    check = False
    fn = None
    if low_memory:
        if (node_name is None) or (table_type is None) or (out_dir is None):
            raise ValueError('All values: node_name, table_type, and out_dir must be provided when low_memory is True.')
        else:
            fn = os.path.join(out_dir, '{}_{}_freq_table.pkl'.format(node_name, table_type))
            if os.path.isfile(fn):
                check = True
    return check, fn


def save_freq_table(freq_table, low_memory, node_name, table_type, out_dir):
    """
    Save Frequency Table

    This function returns either the FrequencyTable that was provided to it or the path where that FrequencyTable has
    been stored if the low memory option has been set. The FrequencyTable will be saved to a file in the out_dir with
    name {}_freq)table.pkl where {} is filled in with the value passed to node_name.

    Args:
        freq_table (FrequencyTable): The FrequencyTable instance to either serialize or pass through.
        low_memory (bool): Whether or not to save the FrequencyTable or just pass it through.
        node_name (str): A string which will be used in naming the save file.
        table_type (str): Whether the FrequencyTable being saved is for single or paired positions.
        out_dir (str): The path to a directory where the FrequencyTable should be saved.
    Returns:
        FrequencyTable/str: Either the provided FrequencyTable (if low_memory is False) or a string providing the path
        to a file where the FrequencyTable has been saved.
    """
    if low_memory:
        check, fn = check_freq_table(low_memory=low_memory, node_name=node_name, table_type=table_type, out_dir=out_dir)
        if not check:
            with open(fn, 'wb') as handle:
                pickle.dump(freq_table, handle, pickle.HIGHEST_PROTOCOL)
        freq_table = fn
    return freq_table


def load_freq_table(freq_table, low_memory):
    """
    Load Frequency Table

    This function returns a FrequencyTable. If a FrequencyTable was provided to it and low_memory is False the instance
    returned is the same as the one passed in. If a string was passed in and low_memory is True a FrequencyTable is
    returned after loading it from the file specified by the string.

    Args:
        freq_table (FrequencyTable/str): A FrequencyTable or the path to one which should be loaded from file.
        low_memory (bool): Whether or not to load the FrequencyTable from file.
    Returns:
        FrequencyTable: Either the FrequencyTable passed in or the one loaded from the file specified by a passed in
        string.
    """
    if low_memory and isinstance(freq_table, FrequencyTable):
        raise ValueError('Low memory setting active but frequency table provided is not a path.')
    elif (not low_memory) and not isinstance(freq_table, FrequencyTable):
        raise ValueError('Low memory setting not active, a frequency table is expected as input.')
    elif low_memory:
        if not os.path.isfile(freq_table):
            raise ValueError('File path is not valid: {}'.format(freq_table))
        with open(freq_table, 'rb') as handle:
            freq_table = pickle.load(handle)
    else:
        pass
    return freq_table


def check_numpy_array(low_memory, node_name, pos_type, score_type, metric, out_dir):
    """
    Check Frequency Table

    This function is used to check wither the desired FrequencyTable has already been produced and stored or not.

    Args:
        low_memory (bool): Whether or not low_memory mode is active (should serialized even files exist?).
        node_name (str): A string which will be used in identifying the save file.
        pos_type (str): Whether the array being saved is for 'single' or 'pair' positions.
        score_type (str): Whether the array being saved contains 'group' or 'rank' scores.
        metric (str): Which method is being used to compute the scores for the group or rank being serialized.
        out_dir (str): The path to a directory where the FrequencyTable could have been saved.
    Returns:
        bool: Whether or not the desired table has previously been saved (if low_memory is False, False is returned by
        default).
        str/None: If low_memory is True the expected path to the file, whether it is present or not.
    """
    check = False
    fn = None
    if low_memory:
        if (node_name is None) or (pos_type is None) or (score_type is None) or (metric is None) or (out_dir is None):
            raise ValueError('All values: node_name, pos_type, score_type, metric, and out_dir must be provided when '
                             'low_memory is True.')
        else:
            fn = os.path.join(out_dir, '{}_{}_{}_{}_score.npz'.format(node_name, pos_type, score_type, metric))
            if os.path.isfile(fn):
                check = True
    return check, fn


def save_numpy_array(mat, out_dir, node_name, pos_type, score_type, metric, low_memory):
    """
    Save Numpy Array

    This function returns either the np.array that was provided to it or the path where that np.array has been stored if
    the low memory option was set. The FrequencyTable will be saved to a file in the out_dir with
    name {}_freq)table.pkl where {} is filled in with the value passed to node_name.

    Args:
        mat (np.array): The np.array instance to either serialize or pass through.
        out_dir (str): The path to a directory where the FrequencyTable could have been saved.
        node_name (str): A string which will be used in identifying the save file.
        pos_type (str): Whether the array being saved is for 'single' or 'pair' positions.
        score_type (str): Whether the array being saved contains 'group' or 'rank' scores.
        metric (str): Which method is being used to compute the scores for the group or rank being serialized.
        low_memory (bool): Whether or not low_memory mode is active (should serialized even files exist?).
    Returns:
        np.array/str: Either the provided np.array (if low_memory is False) or a string providing the path to a file
        where the np.array has been saved.
    """
    if low_memory:
        check, fn = check_numpy_array(low_memory=low_memory, node_name=node_name, pos_type=pos_type,
                                      score_type=score_type, metric=metric, out_dir=out_dir)
        if not check:
            np.savez(fn, mat=mat)
        mat = fn
    return mat


def load_numpy_array(mat, low_memory):
    """
    Load Numpy Array

    This function returns a np.array. If a np.array was provided to it and low_memory is False the instance
    returned is the same as the one passed in. If a string was passed in and low_memory is True a np.array is
    returned after loading it from the file specified by the string.

    Args:
        mat (np.array/str): A np.array or the path to one which should be loaded from file.
        low_memory (bool): Whether or not to load the np.array from file.
    Returns:
        np.array: Either the np.array passed in or the one loaded from the file specified by a passed in string.
    """
    if low_memory and isinstance(mat, np.ndarray):
        raise ValueError('Low memory setting active but matrix provided is not a path.')
    elif (not low_memory) and not isinstance(mat, np.ndarray):
        raise ValueError('Low memory setting not active, a numpy array is expected as input.')
    elif low_memory:
        if not os.path.isfile(mat):
            raise ValueError('File path is not valid: {}'.format(mat))
        load = np.load(mat)
        mat = load['mat']
    else:
        pass
    return mat


def init_characterization_pool(alpha_size, alpha_mapping, alpha_reverse, single_to_pair, alignment, pos_size,
                               components, sharable_dict, sharable_lock, unique_dir, low_memory, write_out_sub_aln,
                               write_out_freq_table, processes, maximum_iterations=10000, single_map=None):
    """
    Init Characterization Pool

    This function initializes a pool of workers with shared resources so that they can quickly characterize the sub
    alignments for each node in the phylogenetic tree.

    Args:
        alpha_size (int): The size of the alphabet needed for the alignment and position size being characterized
        (including the gap character).
        alpha_mapping (dict): A dictionary mapping the letters of the alphabet to numerical positions.
        alpha_reverse (np.array): An array mapping numerical positions back to the letter alphabet of the alphabet.
        single_to_pair (np.array): A two dimensional array mapping single letter numeric positions to paired letter
        numeric positions. Only needed for pairs of positions.
        alignment (SeqAlignment): The alignment for which the trace is being computed, this will be used to generate
        sub alignments for the terminal nodes in the tree.
        pos_size (int): The size of the the positions being analyzed (expecting 1 single position scores or 2 pair
        position scores).
        components (dict): A dictionary mapping a node to its descendants and terminal nodes.
        sharable_dict (multiprocessing.Manager.dict): A thread safe dictionary where the individual processes can
        deposit the characterization of each node and retrieve characterizations of children needed for larger nodes.
        sharable_lock (multiprocessing.Lock): A lock to be used for synchronization of the characterization process.
        unique_dir (str/path): The directory where sub-alignment and frequency table files can be written.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
        write_out_sub_aln (bool): Whether or not to write out the sub-alignment for each node.
        write_out_freq_table (bool): Whether or not to write out the frequency table(s) for each node.
        processes (int): The number of processes being used by the initialized pool.
        maximum_iterations (int): The most attempts that can be made to retrieve a single node descendant.
    """
    global s_to_p, aln, comps, freq_tables, freq_lock, freq_lock, u_dir, low_mem, write_sub_aln, write_freq_table,\
        cpu_count, sleep_time, table_type, single, pair, s_size, s_map, s_rev, p_size, p_map, p_rev, max_iters
    s_to_p = single_to_pair
    aln = alignment
    if pos_size == 1:
        table_type = 'single'
        single = True
        pair = False
        s_size = alpha_size
        s_map = alpha_mapping
        s_rev = alpha_reverse
        p_size = None
        p_map = None
        p_rev = None
    elif pos_size == 2:
        table_type = 'pair'
        single = False
        pair = True
        s_size = None
        s_map = single_map
        s_rev = None
        p_size = alpha_size
        p_map = alpha_mapping
        p_rev = alpha_reverse
    else:
        raise ValueError('pos_size must be defined as 1 or 2.')
    comps = components
    freq_tables = sharable_dict
    freq_lock = sharable_lock
    u_dir = unique_dir
    low_mem = low_memory
    write_sub_aln = write_out_sub_aln
    write_freq_table = write_out_freq_table
    cpu_count = processes
    # Sleep time to use for inner nodes which are not in the lowest rank of the tree being characterized. If a component
    # of the node being characterized is missing this time will be used as the wait time before attempting to get the
    # nodes again. This time is updated from its default during processing, based on the time it takes to characterize a
    # single node.
    sleep_time = 0.1
    max_iters = maximum_iterations


def characterization(node_name, node_type):
    """
    Characterization

    This function accepts a node and its type and characterizes that node. If the node is in the lowest (closest to the
    leaves of the tree) rank being traced, then a sub-alignment is generated from the full alignment (provided by
    init_characterization_pool) and the individual positions (if specified by pos_specific in
    init_characterization_pool) and pairs of positions (if specified by pair_specific in init_characterization) are
    characterized for their nucleic/amino acid content. If the node is non-terminal then the dictionary of
    frequency tables (sharable_dict provided by init_characterization_pool) is accessed to retrieve the characterization
    for the nodes descendants; these are then summed. Each node that is completed by a worker process is
    added to the dictionary provided to init_characterization as sharable_dict, which is where results can be
    found after all processes finish.

    Args:
        node_name (str): The node name to process, this will be used to check for previous characterizations of the node
        if the low memory option is being used. It will also be used to identify which sequences contribute to the node
        (to generate the sub alignment) and to identify its descendants (if it is an inner node). Finally, the node name
        is used to store the characterization so other jobs can access it if necessary (i.e. if an ancestor needs the
        characterization for its own characterization).
        node_type (str): Accepted values are 'component' or 'inner'. A component will be processed from scratch while an
        inner node will be processed by retrieving the frequency tables of previously characterized nodes and adding
        them up.
    Return:
        node_name (str): The node name is returned to keep track of which node has been most recently processed (in the
        multiprocessing context).
    """
    # Since sleep_time may be updated this function must declare that it is referencing the global variable sleep_time
    global sleep_time
    # Check whether the alignment characterization has already been saved to file.
    ft_check, ft_fn = check_freq_table(low_memory=low_mem, node_name=node_name, table_type=table_type,
                                       out_dir=u_dir)
    # If the file was found set the return values for this sub-alignment.
    if low_mem and ft_check:
        freq_table = ft_fn
    else:  # Check what kind of node is being processed
        # Generate the sub alignment for the current node.
        sub_aln = aln.generate_sub_alignment(sequence_ids=comps[node_name]['terminals'])
        # If specified write the alignment to file.
        if write_sub_aln:
            sub_aln.write_out_alignment(file_name=os.path.join(u_dir, '{}.fa'.format(node_name)))
        if node_type == 'component':  # Otherwise generate the sub-alignment and characterize it
            start = time()
            # Characterize the alignment using the most size efficient method.
            if sub_aln.size < 5:
                # If the node is small characterize its sub-alignment by sequence.
                pos_table, pair_table = sub_aln.characterize_positions(
                    single=single, pair=pair, single_size=s_size, single_mapping=s_map, single_reverse=s_rev,
                    pair_size=p_size, pair_mapping=p_map, pair_reverse=p_rev)
            else:
                # If the node is larger characterize the whole alignment, one position at a time.
                pos_table, pair_table = sub_aln.characterize_positions2(
                    single=single, pair=pair, single_size=s_size, single_mapping=s_map, single_reverse=s_rev,
                    pair_size=p_size, pair_mapping=p_map, pair_reverse=p_rev, single_to_pair=s_to_p)
            if table_type == 'single':
                freq_table = pos_table
            elif table_type == 'pair':
                freq_table = pair_table
            else:
                raise ValueError('Unknown table type encountered in characterization!')
            end = time()
            # If characterization has not been timed before record the characterization time (used for sleep time during
            # the next loop, where higher rank nodes are characterized).
            elapsed_time = (end - start) / float(cpu_count)
            freq_lock.acquire()
            if sleep_time > elapsed_time:
                sleep_time = elapsed_time
            freq_lock.release()
        elif node_type == 'inner':
            # Since the node is non-terminal retrieve its descendants' characterizations and merge them to get the
            # parent characterization.
            descendants = set([d.name for d in comps[node_name]['descendants']])
            tries = {}
            components = []
            while len(descendants) > 0:
                descendant = descendants.pop()
                # Attempt to retrieve the current node's descendants' data, sleep and try again if it is not already in
                # the dictionary (i.e. another process is still characterizing that descendant), until all are
                # successfully retrieved.
                try:
                    component = freq_tables[descendant]
                    components.append(component)
                except KeyError:
                    if descendant not in tries:
                        tries[descendant] = 0
                    elif tries[descendant] == max_iters:
                        raise TimeoutError('Too many attempts made to access the descendant data!')
                    else:
                        tries[descendant] += 1
                    descendants.add(descendant)
                    sleep(sleep_time)
            freq_table = None
            # Merge the descendants' FrequencyTable(s) to generate the one for this node.
            for i in range(len(components)):
                curr_freq_table = load_freq_table(components[i]['freq_table'], low_memory=low_mem)
                if freq_table is None:
                    freq_table = curr_freq_table
                else:
                    freq_table += curr_freq_table
        else:
            raise ValueError("node_type must be either 'component' or 'inner'.")
        # Write out the FrequencyTable(s) if specified and serialize it/them if low memory mode is active.
        if write_freq_table:

            freq_table.to_csv(os.path.join(u_dir, f'{node_name}_{table_type}_freq_table.tsv'))
        freq_table = save_freq_table(freq_table=freq_table, low_memory=low_mem, node_name=node_name,
                                     table_type=table_type, out_dir=u_dir)
    # Store the current nodes characterization in the shared dictionary.
    tables = {'freq_table': freq_table}
    freq_lock.acquire()
    freq_tables[node_name] = tables
    freq_lock.release()
    return node_name


def init_characterization_mm_pool(single_mapping, larger_size, larger_mapping, larger_reverse, single_to_pair,
                                  comparison_mapping, mismatch_mask, alignment, position_size, components,
                                  sharable_dict, sharable_lock, unique_dir, low_memory, write_out_sub_aln,
                                  write_out_freq_table, maximum_iterations=10000):
    """
    Init Characterization Match Mismatch Pool

    This function initializes a pool of workers with shared resources so that they can quickly characterize the matches
    and mismatches of sub-alignments for each node in the phylogenetic tree.

    Args:
        single_mapping (dict): A dictionary mapping the single letter alphabet for DNA or Protein to numerical values.
        larger_size (int): The size of the pair, quad, or larger letter alphabet for the alignment being characterized
        (including the gap character).
        larger_mapping (dict): A dictionary mapping the pair, quad, or larger letter alphabet to numerical positions.
        larger_reverse (np.array): An array mapping numerical positions back to the larger letter alphabet.
        single_to_pair (np.array, dtype=np.int32): An array mapping single letter numerical representations
        (axes 0 and 1) to a numerical representations of pairs of residues (value). If pos_size == 1 this argument
        should be set to None (default).
        comparison_mapping (np.array, dtype=np.int32): An array mapping the alphabet used for the positions_size of this
        FrequencyTable, mapped to the alphabet that has characters twice as large (if pos_size == 1, this is the same as
        the description for single_to_pair, if pos_size == 2 this is the mapping from the alphabet of pairs to the
        alphabet of quadruples.
        mismatch_mask (np.array, dtype=np.bool_): An array identifying which positions in the alphabet (the values
        in the provided larger alphabet mapping) correspond to mismatches (variance events). This will be used to
        separate the counts into match and mismatch tables.
        alignment (SeqAlignment): The alignment for which the trace is being computed, this will be used to generate
        sub alignments for the terminal nodes in the tree.
        position_size (int): The size of the positions being considered (1 for single positions, 2 for pairs, etc.).
        components (dict): A dictionary mapping a node to its descendants and terminal nodes.
        sharable_dict (multiprocessing.Manager.dict): A thread safe dictionary where the individual processes can
        deposit the characterization of each node and retrieve characterizations of children needed for larger nodes.
        sharable_lock (multiprocessing.Lock): A lock to be used for synchronization of the characterization process.
        unique_dir (str/path): The directory where sub-alignment and frequency table files can be written.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
        write_out_sub_aln (bool): Whether or not to write out the sub-alignment for each node.
        write_out_freq_table (bool): Whether or not to write out the frequency table(s) for each node.
        maximum_iterations (int): The most attempts that can be made to retrieve a single node descendant.
    """
    global s_map, l_size, l_map, l_rev, s_to_p, comp_map, mis_mask, aln, p_size, t_type, comps, freq_tables, freq_lock,\
        freq_lock, u_dir, low_mem, write_sub_aln, write_freq_table, sleep_time, max_iters
    s_map = single_mapping
    l_size = larger_size
    l_map = larger_mapping
    l_rev = larger_reverse
    s_to_p = single_to_pair
    comp_map = comparison_mapping
    mis_mask = mismatch_mask
    aln = alignment
    p_size = position_size
    if p_size == 1:
        t_type = 'single'
    elif p_size == 2:
        t_type = 'pair'
    else:
        raise ValueError('position_size must be 1 or 2 for charcterization_mm!')
    comps = components
    freq_tables = sharable_dict
    freq_lock = sharable_lock
    u_dir = unique_dir
    low_mem = low_memory
    write_sub_aln = write_out_sub_aln
    write_freq_table = write_out_freq_table
    # Sleep time to use for inner nodes which are not in the lowest rank of the tree being characterized. If a component
    # of the node being characterized is missing this time will be used as the wait time before attempting to get the
    # nodes again. This time is updated from its default during processing, based on the time it takes to characterize a
    # single node.
    sleep_time = 0.1
    max_iters = maximum_iterations


def characterization_mm(node_name, node_type):
    """
    Characterization Match Mismatch

    This function accepts a node characterizes its sub-alignment's matches and mismatches. For each node a sub-alignment
    is generated from the full alignment (provided by init_characterization_pool) and the individual positions (if
    specified by pos_size in init_characterization_pool) and pairs of positions (if specified by pos_size in
    init_characterization) are characterized for the matches and mismatches between nucleic/amino acids in all pairs of
    sequences at that position (in this case a concerted mismatch, .i.e full transition to a different set of
    nucleic/amino acids is considered a match for the purposes of capturing signals like covariation). Each node that is
    completed by a worker process is added to the dictionary provided to init_characterization as sharable_dict, which
    is where results can be found after all processes finish.

    Args:
        node_name (str): The node name to process, this will be used to check for previous characterizations of the node
        if the low memory option is being used. It will also be used to identify which sequences contribute to the node
        (to generate the sub alignment). Finally, the node name is used to store the characterization into a single
        structure which will be returned at the end of processing.
        node_type (str): Accepted values are 'component' or 'inner'. A component will be processed from scratch while an
        inner node will be processed by retrieving the frequency tables of previously characterized nodes and adding
        them up, as well as characterizing the section not previously covered by those nodes.
    Return:
        node_name (str): The node name is returned to keep track of which node has been most recently processed (in the
        multiprocessing context).
    """
    # Check whether the alignment characterization has already been saved to file.
    match_check, match_fn = check_freq_table(low_memory=low_mem, node_name=node_name, table_type=t_type + '_match',
                                             out_dir=u_dir)
    mismatch_check, mismatch_fn = check_freq_table(low_memory=low_mem, node_name=node_name,
                                                   table_type=t_type + '_mismatch', out_dir=u_dir)
    # If the file(s) were found set the return values for this sub-alignment.
    if low_mem and match_check and mismatch_check:
        tables = {'match': match_fn, 'mismatch': mismatch_fn}
    else:  # Check what kind of node is being processed
        # Generate the sub alignment for the current node.
        sub_aln = aln.generate_sub_alignment(sequence_ids=comps[node_name]['terminals'])
        # If specified write the alignment to file.
        if write_sub_aln:
            sub_aln.write_out_alignment(file_name=os.path.join(u_dir, '{}.fa'.format(node_name)))
        curr_num_aln = sub_aln._alignment_to_num(mapping=s_map)
        # Re-order the numeric aln to match the ordering of the sequences in the tree
        reorder_df = pd.DataFrame({'IDs': sub_aln.seq_order, 'Indexes': np.array(range(sub_aln.size))}).set_index('IDs')
        curr_num_aln = curr_num_aln[reorder_df.loc[comps[node_name]['terminals'], 'Indexes'].values, :]
        # Create easy look up for the new indexes
        index_df = pd.DataFrame({'IDs': comps[node_name]['terminals'],
                                 'Indexes': np.array(range(sub_aln.size))}).set_index('IDs')
        index_df = index_df.astype({'Indexes': np.int32})
        if node_type == 'component':
            tables = {'match': FrequencyTable(alphabet_size=l_size, mapping=l_map, reverse_mapping=l_rev,
                                              seq_len=aln.seq_length, pos_size=p_size)}
            tables['mismatch'] = tables['match'].characterize_alignment_mm(num_aln=curr_num_aln, single_to_pair=s_to_p,
                                                                           comparison=comp_map, mismatch_mask=mis_mask)
        elif node_type == 'inner':
            # Since the node is non-terminal characterize the rectangle of sequence comparisons not covered by the
            # descendants, then retrieve its descendants' characterizations and merge all to get the parent
            # characterization.
            descendants = set()
            terminal_indices = []
            tables = {'match': None, 'mismatch': None}
            for d in comps[node_name]['descendants']:
                descendants.add(d.name)
                curr_indices = index_df.loc[comps[d.name]['terminals'], 'Indexes'].values
                for prev_indices in terminal_indices:
                    if np.min(curr_indices) < np.min(prev_indices):
                        ind1 = curr_indices
                        ind2 = prev_indices
                    else:
                        ind1 = prev_indices
                        ind2 = curr_indices
                    curr_match = FrequencyTable(alphabet_size=l_size, mapping=l_map, reverse_mapping=l_rev,
                                                seq_len=aln.seq_length, pos_size=p_size)
                    curr_mismatch = curr_match.characterize_alignment_mm(num_aln=curr_num_aln, single_to_pair=s_to_p,
                                                                         comparison=comp_map, mismatch_mask=mis_mask,
                                                                         indexes1=ind1, indexes2=ind2)
                    if tables['match'] is None:
                        tables['match'] = curr_match
                        tables['mismatch'] = curr_mismatch
                    else:
                        tables['match'] += curr_match
                        tables['mismatch'] += curr_mismatch
                terminal_indices.append(curr_indices)
            components = []
            tries = {}
            while len(descendants) > 0:
                descendant = descendants.pop()
                # Attempt to retrieve the current node's descendants' data, sleep and try again if it is not already in
                # the dictionary (i.e. another process is still characterizing that descendant), until all are
                # successfully retrieved.
                try:
                    component = freq_tables[descendant]
                    components.append(component)
                    # Merge the descendants' FrequencyTable(s) to generate the one for this node.
                    for m in tables:
                        curr_pos_table = load_freq_table(component[m], low_memory=low_mem)
                        tables[m] += curr_pos_table
                except KeyError:
                    if descendant not in tries:
                        tries[descendant] = 0
                    elif tries[descendant] == max_iters:
                        raise TimeoutError('Too many attempts made to access the descendant data!')
                    else:
                        tries[descendant] += 1
                    descendants.add(descendant)
                    sleep(sleep_time)
            for m in tables:
                # Since addition of FrequencyTable objects leads to
                tables[m].set_depth(1.0 if sub_aln.size == 1 else ((sub_aln.size * (sub_aln.size - 1)) / 2.0))
        else:
            raise ValueError("node_type must be either 'component' or 'inner'.")
        # Write out the FrequencyTable(s) if specified and serialize it/them if low memory mode is active.
        for m in tables:
            if write_freq_table:
                tables[m].to_csv(os.path.join(u_dir, '{}_{}_{}_freq_table.tsv'.format(node_name, t_type, m)))
            tables[m] = save_freq_table(freq_table=tables[m], low_memory=low_mem, node_name=node_name,
                                        table_type=t_type + '_' + m, out_dir=u_dir)
    freq_lock.acquire()
    freq_tables[node_name] = tables
    freq_lock.release()
    return node_name


def init_trace_groups(scorer, match_mismatch, u_dict, low_memory, unique_dir):
    """
    Init Trace Pool

    This function initializes a pool of workers with shared resources so that they can quickly perform the group level
    scoring for the trace algorithm.

    Args:
        scorer (PositionalScorer): A scorer used to compute the group and rank scores according to a given metric.
        match_mismatch (bool): Whether or not characterization was of the match_mismatch type (comparison of all
        possible transitions for a given position/pair, or considering each position/pair only once for each sequence
        present).
        u_dict (dict): A dictionary containing the node characterizations.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
        unique_dir (str/path): The directory where group score vectors/matrices can be written.
    """
    global pos_scorer, mm_analysis, unique_nodes, low_mem, u_dir
    pos_scorer = scorer
    mm_analysis = match_mismatch
    unique_nodes = u_dict
    low_mem = low_memory
    u_dir = unique_dir


def trace_groups(node_name):
    """
    Trace Groups

    A function which performs the group scoring part of the trace algorithm. It depends on the init_trace_groups
    function to set up the necessary shared resources. For each node name provided a group score is computed based on
    he FrequencyTable stored in unique_nodes for that node name. This can be performed for any node previously
    characterized and stored in unique_nodes.

    Args:
        node_name (str): The node whose group score to calculate from the characterization in the unique_nodes dict
        made available by init_trace_groups.
    Returns:
        str: The name of the node which has been scored, returned so that the proper position can be updated in
        unique_nodes.
        dict: The single and pair position group scores which will be added to unique_nodes under the name of the node
        which has been scored.
    """
    pos_type = 'single' if pos_scorer.position_size == 1 else 'pair'
    # Check whether the group score has already been saved to file.
    arr_check, arr_fn = check_numpy_array(low_memory=low_mem, node_name=node_name, pos_type=pos_type,
                                          score_type='group', metric=pos_scorer.metric, out_dir=u_dir)
    # If the file(s) were found set the return values for this node.
    if low_mem and arr_check:
        group_score = arr_fn
    else:
        # Using the provided scorer and the characterization for the node found in unique nodes, compute the
        # group score
        if mm_analysis:
            freq_table = {'match': load_freq_table(freq_table=unique_nodes[node_name]['match'], low_memory=low_mem),
                          'mismatch': load_freq_table(freq_table=unique_nodes[node_name]['mismatch'],
                                                      low_memory=low_mem)}
        else:
            freq_table = load_freq_table(freq_table=unique_nodes[node_name]['freq_table'], low_memory=low_mem)
        group_score = pos_scorer.score_group(freq_table)
        group_score = save_numpy_array(mat=group_score, out_dir=u_dir, low_memory=low_mem, node_name=node_name,
                                       pos_type=pos_type, metric=pos_scorer.metric, score_type='group')
    return node_name, group_score


def init_trace_ranks(scorer, a_dict, u_dict, low_memory, unique_dir):
    """
    Init Trace Ranks

    This function initializes a pool of workers with shared resources so that they can quickly perform the rank level
    scoring for the trace algorithm.

    Args:
        scorer (PositionalScorer): A scorer used to compute the group and rank scores according to a given metric.
        a_dict (dict): The rank and group assignments for nodes in the tree.
        u_dict (dict): A dictionary containing the node characterizations and group level scores for those nodes.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
        unique_dir (str/path): The directory where group score vectors/matrices can be loaded from and where rank
        scores can be written.
    """
    global pos_scorer, assignments, unique_nodes, low_mem, u_dir
    pos_scorer = scorer
    assignments = a_dict
    unique_nodes = u_dict
    low_mem = low_memory
    u_dir = unique_dir


def trace_ranks(rank):
    """
    Trace Ranks

    A function which performs the rank scoring part of the trace algorithm. It depends on the init_trace_ranks
    function to set up the necessary shared resources. All group scores for the provided rank are gathered by
    determining the nodes using the assignments dictionary and their group scores using the unique_nodes dictionary.
    These scores are summed (by position) and the rank level score is computed using the provided scorer.

    Args:
        rank (int): Which rank to score from the assignments dictionary based on group scores available in the
        unique_nodes dictionary.
    Returns:
        int: The rank which has been scored (this will be used to update the rank_scores dictionary).
        dict: A dictionary containing the single and pair position rank score for the specified rank.
    """
    pos_type = 'single' if pos_scorer.position_size == 1 else 'pair'
    # Check whether the group score has already been saved to file.
    arr_check, arr_fn = check_numpy_array(low_memory=low_mem, node_name=str(rank), pos_type=pos_type,
                                          score_type='rank', metric=pos_scorer.metric, out_dir=u_dir)
    # If the file(s) were found set the return values for this rank.
    if low_mem and arr_check:
        rank_scores = arr_fn
    else:
        # Retrieve all group scores for this rank
        group_scores = np.zeros(pos_scorer.dimensions)
        # For each group in the rank update the cumulative sum for the rank
        for g in assignments[rank]:
            node_name = assignments[rank][g]['node'].name
            group_score = load_numpy_array(mat=unique_nodes[node_name]['group_scores'], low_memory=low_mem)
            group_scores += group_score
        # Compute the rank score over the cumulative sum of group scores.
        rank_scores = pos_scorer.score_rank(group_scores, rank)
        rank_scores = save_numpy_array(mat=rank_scores, out_dir=u_dir, low_memory=low_mem, node_name=str(rank),
                                       pos_type='single', score_type='rank', metric=pos_scorer.metric)
    return rank, rank_scores
