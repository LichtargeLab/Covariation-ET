"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from time import sleep, time
from Bio.Alphabet import Gapped
from multiprocessing import Manager, Pool, Lock
from FrequencyTable import FrequencyTable
from MatchMismatchTable import MatchMismatchTable
from EvolutionaryTraceAlphabet import MultiPositionAlphabet
from utils import gap_characters, build_mapping, compute_rank_and_coverage


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
        position_specific (bool): Whether this trace will perform position specific analyses or not.
        pair_specific (bool): Whether this trace will perform pair specific analyses or not.
        match_mismatch (bool): Whether to use the match mismatch characterization (looking at all possible transitions
        in a position/pair between sequences of an alignment, True) or the standard protocol (looking at each position
        only once for each sequence in the alignment, False).
        out_dir (str/path): Where results from a trace should be stored.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
    """

    def __init__(self, alignment, phylo_tree, group_assignments, position_specific=True, pair_specific=True,
                 match_mismatch=False, output_dir=None, low_memory=False):
        """
        Initializer for Trace object.

        Args:
            alignment (SeqAlignment): The alignment for which to perform a trace analysis.
            phylo_tree (PhylogeneticTree): The tree based on the alignment to use during the trace analysis.
            group_assignments (dict): The group assignments for nodes in the tree.
            position_specific (bool): Whether or not to perform the trace for specific positions.
            pair_specific (bool): Whether or not to perform the trace for pairs of positions.
            match_mismatch (bool): Whether to use the match mismatch characterization (True) or the standard protocol
            (False).
            output_dir (str/path): Where results from a trace should be stored.
            low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
            resources.
        """
        self.aln = alignment
        self.phylo_tree = phylo_tree
        self.assignments = group_assignments
        self.unique_nodes = None
        self.rank_scores = None
        self.final_scores = None
        self.final_ranks = None
        self.final_coverage = None
        self.pos_specific = position_specific
        self.pair_specific = pair_specific
        self.match_mismatch = match_mismatch
        if output_dir is None:
            self.out_dir = os.getcwd()
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            self.out_dir = output_dir
        self.low_memory = low_memory

    def characterize_rank_groups_standard(self, unique_dir, single_size, single_mapping, single_reverse, processes=1,
                                          write_out_sub_aln=True, write_out_freq_table=True):
        """
        Characterize Rank Group

        This function iterates over the rank and group assignments and characterizes all positions for each sub
        alignment. Characterization consists of FrequencyTable objects which are added to the group_assignments
        dictionary provided at initialization (single position FrequencyTables are added under the key 'single', while
        FrequencyTables for pairs of positions are added under the key 'pair').

        Args:
            unique_dir (str/path): The path to the directory where unique node data can be stored.
            single_size (int): The size of the single letter alphabet being used.
            single_mapping (dict): A dictionary from character to integer representation.
            single_reverse (np.array): An array mapping from an integer (index) representation back to a single letter
            character.
            processes (int): The maximum number of sequences to use when performing this characterization.
            write_out_sub_aln (bool): Whether or not to write out the sub-alignments for each node while characterizing
            it.
            write_out_freq_table (bool): Whether or not to write out the frequency table for each node while
            characterizing it.
        """
        if self.pair_specific:
            pair_alphabet = MultiPositionAlphabet(alphabet=Gapped(self.aln.alphabet), size=2)
            pair_size, _, pair_mapping, pair_reverse = build_mapping(alphabet=pair_alphabet)
            single_to_pair = np.zeros((max(single_mapping.values()) + 1, max(single_mapping.values()) + 1))
            for char in pair_mapping:
                single_to_pair[single_mapping[char[0]], single_mapping[char[1]]] = pair_mapping[char]
        else:
            pair_size = None
            pair_mapping = None
            pair_reverse = None
            single_to_pair = None
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
                    initargs=(single_size, single_mapping, single_reverse, pair_size, pair_mapping, pair_reverse,
                              single_to_pair, self.aln, self.pos_specific, self.pair_specific, visited,
                              frequency_tables, tables_lock, unique_dir, self.low_memory, write_out_sub_aln,
                              write_out_freq_table, processes))
        for char_node in to_characterize:
            pool.apply_async(func=characterization, args=char_node, callback=update_characterization)
        pool.close()
        pool.join()
        characterization_pbar.close()
        frequency_tables = dict(frequency_tables)
        self.unique_nodes = frequency_tables

    # def characterize_rank_groups_match_mismatch(self, unique_dir, single_size, single_mapping, single_reverse,
    #                                             processes=1, write_out_sub_aln=True, write_out_freq_table=True):
    #     if self.pos_specific and not self.pair_specific:
    #         pair_alphabet = MultiPositionAlphabet(alphabet=Gapped(self.aln.alphabet), size=2)
    #         larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=pair_alphabet)
    #         single_to_larger = {(single_mapping[char[0]], single_mapping[char[1]]): larger_mapping[char]
    #                             for char in larger_mapping if larger_mapping[char] < larger_size}
    #         gaps = {'-'} if '-' in pair_alphabet.letters else gap_characters
    #         dummy_dict = {'{0}'.format(next(iter(single_mapping))): 0}
    #         pos_type = 'position'
    #         table_type = 'single'
    #     elif self.pair_specific and not self.pos_specific:
    #         quad_alphabet = MultiPositionAlphabet(alphabet=Gapped(self.aln.alphabet), size=4)
    #         larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=quad_alphabet)
    #         single_to_larger = {(single_mapping[char[0]], single_mapping[char[1]]): larger_mapping[char]
    #                           for char in larger_mapping if larger_mapping[char] < larger_size}
    #         gaps = {'-'} if '-' in quad_alphabet.letters else gap_characters
    #         dummy_dict = {'{0}{0}'.format(next(iter(single_mapping))): 0}
    #         pos_type = 'pair'
    #         table_type = 'pair'
    #     else:
    #         raise AttributeError('Match/Mismatch characterization requires that either pos_specific or pair_'
    #                              'specific be selected but not both.')
    #     # The FrequencyTable checks that the alphabet matches the provided position size by looking whether the
    #     # first element of the mapping dictionary has the same size as the position since we are looking at
    #     # something twice the size of the position (either pairs for single positions or quadruples for pairs of
    #     # positions) this causes an error. Therefore I introduced this hack, to use a dummy dictionary to get
    #     # through proper initialization, which does expose a vulnerability in the FrequencyTable class (the check
    #     # could be more stringent) but it allows for this more flexible behavior.
    #     freq_table = FrequencyTable(alphabet_size=larger_size, mapping=dummy_dict, reverse_mapping=larger_reverse,
    #                                 seq_len=self.aln.seq_length, pos_size=(1 if self.pos_specific else 2))
    #     # Completes the hack just described, providing the correct mapping table to replace the dummy table
    #     # provided.
    #     freq_table.mapping = larger_mapping
    #     to_characterize = freq_table.get_positions()
    #     num_aln = self.aln._alignment_to_num(mapping=single_mapping)
    #     match_mismatch_table = MatchMismatchTable(seq_len=self.aln.seq_length, num_aln=num_aln,
    #                                               single_alphabet_size=single_size, single_mapping=single_mapping,
    #                                               single_reverse_mapping=single_reverse,
    #                                               larger_alphabet_size=larger_size,
    #                                               larger_alphabet_mapping=larger_mapping,
    #                                               larger_alphabet_reverse_mapping=larger_reverse,
    #                                               single_to_larger_mapping=single_to_larger,
    #                                               pos_size=(1 if self.pos_specific else 2))
    #     match_mismatch_table.identify_matches_mismatches()
    #     node_sequences = {}
    #     # pool_manager = Manager()
    #     frequency_tables = {}
    #     completed_tables = {}
    #     for r in self.assignments.keys():
    #         for g in self.assignments[r]:
    #             node = self.assignments[r][g]['node']
    #             if node.name not in node_sequences:
    #                 sub_aln = self.aln.generate_sub_alignment(sequence_ids=self.assignments[r][g]['terminals'])
    #                 node_sequences[node.name] = sorted([self.aln.seq_order.index(s)
    #                                                     for s in sub_aln.seq_order])
    #                 # Check whether the alignment characterization has already been saved to file.
    #                 match_check, match_fn = check_freq_table(low_memory=self.low_memory, node_name=node.name,
    #                                                          table_type=table_type + '_match', out_dir=unique_dir)
    #                 mismatch_check, mismatch_fn = check_freq_table(low_memory=self.low_memory, node_name=node.name,
    #                                                                table_type=table_type + '_mismatch',
    #                                                                out_dir=unique_dir)
    #                 # If the file(s) were found set the return values for this sub-alignment.
    #                 if self.low_memory and match_check and mismatch_check:
    #                     completed_tables[node.name] = {'match': match_fn, 'mismatch': mismatch_fn}
    #                 else:  # Check what kind of node is being processed
    #                     frequency_tables[node.name] = {'match': deepcopy(freq_table), 'mismatch': deepcopy(freq_table)}
    #                     possible_matches_mismatches = ((sub_aln.size ** 2) - sub_aln.size) / 2.0
    #                     frequency_tables[node.name]['match'].set_depth(depth=possible_matches_mismatches)
    #                     frequency_tables[node.name]['mismatch'].set_depth(depth=possible_matches_mismatches)
    #                 if write_out_sub_aln:
    #                     sub_aln.write_out_alignment(file_name=os.path.join(unique_dir, '{}.fa'.format(node.name)))
    #     characterization_pbar = tqdm(total=len(to_characterize), unit='characterizations')
    #
    #     def update_characterization(return_val):
    #         """
    #         Update Characterization
    #
    #         This function serves to update the progress bar for characterization.
    #
    #         Args:
    #             return_pos (str): The name of the position which has just finished being characterized.
    #         """
    #         print(return_val)
    #         pos, char_dict = return_val
    #         for node in char_dict:
    #             for status in char_dict[node]:
    #                 for char in char_dict[node][status]:
    #                     frequency_tables[node][status]._increment_count(pos=pos, char=char, amount=char_dict[node][status][char])
    #         characterization_pbar.update(1)
    #         characterization_pbar.refresh()
    #
    #     # pool = Pool(processes=processes, initializer=init_characterization_mm_pool,
    #     #             initargs=(self.aln.size, node_sequences, match_mismatch_table, processes))
    #     # init_characterization_mm_pool(self.aln.size, node_sequences, match_mismatch_table, table_locks,
    #     #                               frequency_tables, processes)
    #     init_characterization_mm_pool(self.aln.size, node_sequences, match_mismatch_table, processes)
    #     for pos in to_characterize:
    #         ret = characterization_mm(pos)
    #         update_characterization(ret)
    #         # pool.apply_async(func=characterization_mm, args=(pos, ), callback=update_characterization)
    #     # pool.close()
    #     # pool.join()
    #     characterization_pbar.close()
    #     for node_name in frequency_tables:
    #         for m in ['match', 'mismatch']:
    #             frequency_tables[node_name][m].finalize_table()
    #             if write_out_freq_table:
    #                 frequency_tables[node_name][m].to_csv(
    #                     os.path.join(unique_dir, '{}_{}_{}_freq_table.tsv'.format(pos_type, node_name, m)))
    #             frequency_tables[node_name][m] = save_freq_table(freq_table=frequency_tables[node_name][m],
    #                                                              low_memory=self.low_memory, node_name=node_name,
    #                                                              table_type=table_type + '_' + m, out_dir=unique_dir)
    #     self.unique_nodes = frequency_tables
    #     self.unique_nodes.update(completed_tables)

    def characterize_rank_groups_match_mismatch(self, unique_dir, single_size, single_mapping, single_reverse,
                                                processes=1, write_out_sub_aln=True, write_out_freq_table=True):
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
            single_reverse (np.array): An array mapping from an integer (index) representation back to a single letter
            character.
            processes (int): The maximum number of sequences to use when performing this characterization.
            write_out_sub_aln (bool): Whether or not to write out the sub-alignments for each node while characterizing
            it.
            write_out_freq_table (bool): Whether or not to write out the frequency table for each node while
            characterizing it.
        """
        if self.pos_specific and not self.pair_specific:
            pair_alphabet = MultiPositionAlphabet(alphabet=Gapped(self.aln.alphabet), size=2)
            larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=pair_alphabet)
            single_to_larger = {(single_mapping[char[0]], single_mapping[char[1]]): larger_mapping[char]
                                for char in larger_mapping if larger_mapping[char] < larger_size}
            pos_size = 1
            pos_type = 'position'
            table_type = 'single'
        elif self.pair_specific and not self.pos_specific:
            quad_alphabet = MultiPositionAlphabet(alphabet=Gapped(self.aln.alphabet), size=4)
            larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=quad_alphabet)
            single_to_larger = {(single_mapping[char[0]], single_mapping[char[1]], single_mapping[char[2]],
                                 single_mapping[char[3]]): larger_mapping[char]
                                for char in larger_mapping if larger_mapping[char] < larger_size}
            pos_size = 2
            pos_type = 'pair'
            table_type = 'pair'
        else:
            raise AttributeError('Match/Mismatch characterization requires that either pos_specific or pair_'
                                 'specific be selected but not both.')
        num_aln = self.aln._alignment_to_num(mapping=single_mapping)
        match_mismatch_table = MatchMismatchTable(seq_len=self.aln.seq_length, num_aln=num_aln,
                                                  single_alphabet_size=single_size, single_mapping=single_mapping,
                                                  single_reverse_mapping=single_reverse,
                                                  larger_alphabet_size=larger_size,
                                                  larger_alphabet_mapping=larger_mapping,
                                                  larger_alphabet_reverse_mapping=larger_reverse,
                                                  single_to_larger_mapping=single_to_larger,
                                                  pos_size=pos_size)
        match_mismatch_table.identify_matches_mismatches()
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
                    initargs=(single_size, single_mapping, single_reverse, larger_size, larger_mapping, larger_reverse,
                              single_to_larger, match_mismatch_table, self.aln, pos_size, pos_type,
                              table_type, visited, frequency_tables, tables_lock, unique_dir, self.low_memory,
                              write_out_sub_aln, write_out_freq_table))
        for char_node in to_characterize:
            pool.apply_async(func=characterization_mm, args=char_node, callback=update_characterization)
        pool.close()
        pool.join()
        characterization_pbar.close()
        frequency_tables = dict(frequency_tables)
        self.unique_nodes = frequency_tables

    def characterize_rank_groups(self, processes=1, write_out_sub_aln=True, write_out_freq_table=True):
        """
        Characterize Rank Groups:

        This function acts as a switch statement to use the appropriate characterization function (either standard or
        match mismatch).

        Arguments:
            processes (int): The maximum number of sequences to use when performing this characterization.
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
            # self.characterize_rank_groups_match_mismatch(unique_dir=unique_dir, single_size=single_size,
            #                                              single_mapping=single_mapping, single_reverse=single_reverse,
            #                                              processes=processes, write_out_sub_aln=write_out_sub_aln,
            #                                              write_out_freq_table=write_out_freq_table)
            self.characterize_rank_groups_match_mismatch(unique_dir=unique_dir, single_size=single_size,
                                                         single_mapping=single_mapping, single_reverse=single_reverse,
                                                         processes=processes, write_out_sub_aln=write_out_sub_aln,
                                                         write_out_freq_table=write_out_freq_table)
        else:
            self.characterize_rank_groups_standard(unique_dir=unique_dir, single_size=single_size,
                                                   single_mapping=single_mapping, single_reverse=single_reverse,
                                                   processes=processes, write_out_sub_aln=write_out_sub_aln,
                                                   write_out_freq_table=write_out_freq_table)

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
            self.unique_nodes[completed_name].update(completed_components)
            group_pbar.update(1)
            group_pbar.refresh()

        pool1 = Pool(processes=processes, initializer=init_trace_groups,
                     initargs=(scorer, self.pos_specific, self.pair_specific, self.match_mismatch, self.unique_nodes,
                               self.low_memory, unique_dir))
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
            rank, components = return_tuple
            self.rank_scores[rank] = components
            rank_pbar.update(1)
            rank_pbar.refresh()

        pool2 = Pool(processes=processes, initializer=init_trace_ranks,
                     initargs=(scorer, self.pos_specific, self.pair_specific, self.assignments,
                               self.unique_nodes, self.low_memory, unique_dir))
        for rank in sorted(self.assignments.keys(), reverse=True):
            pool2.apply_async(trace_ranks, (rank,), callback=update_rank)
        pool2.close()
        pool2.join()
        rank_pbar.close()
        # Combine rank scores to generate a final score for each position
        final_scores = np.zeros(scorer.dimensions)
        for rank in self.rank_scores:
            if self.pair_specific:
                rank_scores = self.rank_scores[rank]['pair_ranks']
            elif self.pos_specific:
                rank_scores = self.rank_scores[rank]['single_ranks']
            else:
                raise ValueError('pair_specific and pos_specific were not set rank: {} not in rank_dict'.format(rank))
            rank_scores = load_numpy_array(mat=rank_scores, low_memory=self.low_memory)
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
            freq_table = load_freq_table(freq_table=self.unique_nodes[root_node_name][pos_type],
                                         low_memory=self.low_memory)
            gap_chars = list(set(Gapped(self.aln.alphabet).letters).intersection(gap_characters))
            if len(gap_chars) > 1:
                raise ValueError('More than one gap character present in alignment alphabet! {}'.format(gap_chars))
            gap_char = gap_chars[0]
            max_rank_score = np.max(final_scores)
            for i in range(freq_table.sequence_length):
                if freq_table.get_frequency(pos=i, char=gap_char) > gap_correction:
                    final_scores[i] = max_rank_score
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
        fn = os.path.join(out_dir, '{}_{}_freq_table.pkl'.format(node_name, table_type))
        if not os.path.isfile(fn):
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
    elif low_memory:
        if not os.path.isfile(freq_table):
            raise ValueError('File path is not valid: {}'.format(freq_table))
        with open(freq_table, 'rb') as handle:
            freq_table = pickle.load(handle)
    else:
        pass
    return freq_table


def init_characterization_pool(single_size, single_mapping, single_reverse, pair_size, pair_mapping, pair_reverse,
                               single_to_pair, alignment, pos_specific, pair_specific, components, sharable_dict,
                               sharable_lock, unique_dir, low_memory, write_out_sub_aln, write_out_freq_table,
                               processes):
    """
    Init Characterization Pool

    This function initializes a pool of workers with shared resources so that they can quickly characterize the sub
    alignments for each node in the phylogenetic tree.

    Args:
        single_size (int): The size of the single letter alphabet for the alignment being characterized (including the
        gap character).
        single_mapping (dict): A dictionary mapping the single letter alphabet to numerical positions.
        single_reverse (np.array): An array mapping numerical positions back to the single letter alphabet.
        pair_size (int): The size of the paired letter alphabet for the alignment being characterized (including the
        gap character).
        pair_mapping (dict): A dictionary mapping the paired letter alphabet to numerical positions.
        pair_reverse (np.array): An array mapping numerical positions back to the paired letter alphabet.
        single_to_pair (np.array): A two dimensional array mapping single letter numeric positions to paired letter
        numeric positions.
        alignment (SeqAlignment): The alignment for which the trace is being computed, this will be used to generate
        sub alignments for the terminal nodes in the tree.
        pos_specific (bool): Whether or not to characterize single positions.
        pair_specific (bool): Whether or not to characterize pairs of positions.
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
    """
    global s_size, s_map, s_rev, p_size, p_map, p_rev, s_to_p, aln, single, pair, comps, freq_tables, freq_lock,\
        freq_lock, u_dir, low_mem, write_sub_aln, write_freq_table, cpu_count, sleep_time
    s_size = single_size
    s_map = single_mapping
    s_rev = single_reverse
    p_size = pair_size
    p_map = pair_mapping
    p_rev = pair_reverse
    s_to_p = single_to_pair
    aln = alignment
    single = pos_specific
    pair = pair_specific
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
    single_check, single_fn = check_freq_table(low_memory=low_mem, node_name=node_name, table_type='single',
                                               out_dir=u_dir)
    pair_check, pair_fn = check_freq_table(low_memory=low_mem, node_name=node_name, table_type='pair',
                                           out_dir=u_dir)
    # If the file(s) were found set the return values for this sub-alignment.
    if low_mem and (single_check >= single) and (pair_check >= pair):
        pos_table = single_fn if single else None
        pair_table = pair_fn if pair else None
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
                    single=single, pair=pair, single_letter_size=s_size, single_letter_mapping=s_map,
                    single_letter_reverse=s_rev, pair_letter_size=p_size, pair_letter_mapping=p_map,
                    pair_letter_reverse=p_rev, single_to_pair=s_to_p)
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
            # tries = 0
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
                    descendants.add(descendant)
                    sleep(sleep_time)
            pos_table = None
            pair_table = None
            # Merge the descendants' FrequencyTable(s) to generate the one for this node.
            for i in range(len(components)):
                if single:
                    curr_pos_table = load_freq_table(components[i]['single'], low_memory=low_mem)
                    if pos_table is None:
                        pos_table = curr_pos_table
                    else:
                        pos_table += curr_pos_table
                if pair:
                    curr_pair_table = load_freq_table(components[i]['pair'], low_memory=low_mem)
                    if pair_table is None:
                        pair_table = curr_pair_table
                    else:
                        pair_table += curr_pair_table
        else:
            raise ValueError("node_type must be either 'component' or 'inner'.")
        # Write out the FrequencyTable(s) if specified and serialize it/them if low memory mode is active.
        if single:
            if write_freq_table:
                pos_table.to_csv(os.path.join(u_dir, '{}_position_freq_table.tsv'.format(node_name)))
            pos_table = save_freq_table(freq_table=pos_table, low_memory=low_mem,
                                        node_name=node_name, table_type='single'.format(node_name), out_dir=u_dir)
        if pair:
            if write_freq_table:
                pair_table.to_csv(os.path.join(u_dir, '{}_pair_freq_table.tsv'.format(node_name)))
            pair_table = save_freq_table(freq_table=pair_table, low_memory=low_mem,
                                         node_name=node_name, table_type='pair'.format(node_name), out_dir=u_dir)
    # Store the current nodes characterization in the shared dictionary.
    tables = {'single': pos_table, 'pair': pair_table}
    freq_lock.acquire()
    freq_tables[node_name] = tables
    freq_lock.release()
    return node_name


def init_characterization_mm_pool(single_size, single_mapping, single_reverse, larger_size, larger_mapping,
                                  larger_reverse, single_to_larger, match_mismatch_table, alignment,
                                  position_size, position_type, table_type, components, sharable_dict, sharable_lock,
                                  unique_dir, low_memory, write_out_sub_aln, write_out_freq_table):
    """
    Init Characterization Match Mismatch Pool

    This function initializes a pool of workers with shared resources so that they can quickly characterize the matches
    and mismatches of sub-alignments for each node in the phylogenetic tree.

    Args:
        single_size (int): The size of the single letter alphabet for the alignment being characterized (including the
        gap character).
        single_mapping (dict): A dictionary mapping the single letter alphabet to numerical positions.
        single_reverse (np.array): An array mapping numerical positions back to the single letter alphabet.
        larger_size (int): The size of the pair, quad, or larger letter alphabet for the alignment being characterized
        (including the gap character).
        larger_mapping (dict): A dictionary mapping the pair, quad, or larger letter alphabet to numerical positions.
        larger_reverse (np.array): An array mapping numerical positions back to the larger letter alphabet.
        single_to_larger (dict): A dictionary mapping single letter numeric positions to pair, quad, or larger letter
        alphabet numeric positions.
        match_mismatch_table (MatchMismatchTable): A characterization of all the matches and mismatches of single
        positions in the full alignment being characterized, which can be used to check the match/mismatch status and
        character at a given position and pair of sequences.
        alignment (SeqAlignment): The alignment for which the trace is being computed, this will be used to generate
        sub alignments for the terminal nodes in the tree.
        position_size (int): The size of the positions being considered (1 for single positions, 2 for pairs, etc.).
        position_type (str): The type of position being considered ('position' for single positions, and 'pair' for
        pairs of positions.
        table_type (str): The type of table being generated and serialized; if specified ('single' for single positions
        and 'pair' for pairs of positions).
        components (dict): A dictionary mapping a node to its descendants and terminal nodes.
        sharable_dict (multiprocessing.Manager.dict): A thread safe dictionary where the individual processes can
        deposit the characterization of each node and retrieve characterizations of children needed for larger nodes.
        sharable_lock (multiprocessing.Lock): A lock to be used for synchronization of the characterization process.
        unique_dir (str/path): The directory where sub-alignment and frequency table files can be written.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
        write_out_sub_aln (bool): Whether or not to write out the sub-alignment for each node.
        write_out_freq_table (bool): Whether or not to write out the frequency table(s) for each node.
    """
    global s_size, s_map, s_rev, l_size, l_map, l_rev, s_to_l, mm_table, aln, p_size, p_type, t_type, comps,\
        freq_tables, freq_lock, freq_lock, u_dir, low_mem, write_sub_aln, write_freq_table, sleep_time
    s_size = single_size
    s_map = single_mapping
    s_rev = single_reverse
    l_size = larger_size
    l_map = larger_mapping
    l_rev = larger_reverse
    s_to_l = single_to_larger
    mm_table = match_mismatch_table
    aln = alignment
    p_size = position_size
    p_type = position_type
    t_type = table_type
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
        tables = {'match': FrequencyTable(alphabet_size=l_size, mapping=l_map, reverse_mapping=l_rev,
                                          seq_len=aln.seq_length, pos_size=p_size)}
        # The depth for the alignment will not be set since we do not use any of the characterization  functions here,
        # it needs to be the number of possible matches mismatches when comparing all elements of a column or pair of
        # columns to one another (i.e. the upper triangle of the column vs. column matrix). For the smallest
        # sub-alignments the size of the sub-alignment is 1 and therefor there are no values in the upper triangle
        # (because the matrix of sequence comparisons is only one element large, which is also the diagonal of that
        # matrix and therefore not counted). Setting the depth to 0.0 causes divide by zero issues when calculating
        # frequencies so the depth is being arbitrarily set to 1 here. This is incorrect, but all counts should be 0 so
        # the resulting frequencies should be calculated correctly.
        depth = 1.0 if sub_aln.size == 1 else (((sub_aln.size**2) - sub_aln.size) / 2.0)
        tables['match'].set_depth(depth)
        tables['mismatch'] = deepcopy(tables['match'])
        if node_type == 'component':
            for pos in tables['match'].get_positions():
                char_dict = {'match': {}, 'mismatch': {}}
                curr_indices = [aln.seq_order.index(s_id) for s_id in sub_aln.seq_order]
                for i in range(sub_aln.size):
                    s1 = aln.seq_order.index(sub_aln.seq_order[i])
                    for j in range(i+1, sub_aln.size):
                        s2 = aln.seq_order.index(sub_aln.seq_order[j])
                        status, char = mm_table.get_status_and_character(pos=pos, seq_ind1=s1, seq_ind2=s2)
                        if char not in char_dict[status]:
                            char_dict[status][char] = 0
                        char_dict[status][char] += 1
                for m in char_dict:
                    for char in char_dict[m]:
                        tables[m]._increment_count(pos=pos, char=char, amount = char_dict[m][char])
            for m in tables:
                tables[m].finalize_table()
        elif node_type == 'inner':
            # Since the node is non-terminal characterize the rectangle of sequence comparisons not covered by the
            # descendants, then retrieve its descendants' characterizations and merge all to get the parent
            # characterization.
            descendants = set()
            terminal_indices = []
            for d in comps[node_name]['descendants']:
                descendants.add(d.name)
                curr_indices = [aln.seq_order.index(t) for t in comps[d.name]['terminals']]
                for pos in tables['match'].get_positions():
                    char_dict = {'match': {}, 'mismatch': {}}
                    for prev_indices in terminal_indices:
                        for r1 in prev_indices:
                            for r2 in curr_indices:
                                first, second = (r1, r2) if r1 < r2 else (r2, r1)
                                status, char = mm_table.get_status_and_character(pos, seq_ind1=first, seq_ind2=second)
                                if char not in char_dict[status]:
                                    char_dict[status][char] = 0
                                char_dict[status][char] += 1
                    for m in char_dict:
                        for char in char_dict[m]:
                            tables[m]._increment_count(pos=pos, char=char, amount=char_dict[m][char])
                terminal_indices.append(curr_indices)
            for m in tables:
                tables[m].finalize_table()
            # tries = 0
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
                    descendants.add(descendant)
                    sleep(sleep_time)
            # Merge the descendants' FrequencyTable(s) to generate the one for this node.
            for i in range(len(components)):
                for m in tables:
                    curr_pos_table = load_freq_table(components[i][m], low_memory=low_mem)
                    tables[m] += curr_pos_table
            for m in tables:
                # Since addition of FrequencyTable objects leads to
                tables[m].set_depth(1.0 if sub_aln.size == 1 else (((sub_aln.size**2) - sub_aln.size) / 2.0))
        else:
            raise ValueError("node_type must be either 'component' or 'inner'.")
        # Write out the FrequencyTable(s) if specified and serialize it/them if low memory mode is active.
        for m in tables:
            if write_freq_table:
                tables[m].to_csv(os.path.join(u_dir, '{}_{}_{}_freq_table.tsv'.format(node_name, p_type, m)))
            tables[m] = save_freq_table(freq_table=tables[m], low_memory=low_mem, node_name=node_name,
                                        table_type=t_type + '_' + m, out_dir=u_dir)
    freq_lock.acquire()
    freq_tables[node_name] = tables
    freq_lock.release()
    return node_name


# def init_characterization_mm_pool(depth, node_sequences, match_mismatch_table, processes):
#     """
#     Init Characterization Pool
#
#     This function initializes a pool of workers with shared resources so that they can quickly characterize the
#     positions for each node in the phylogenetic tree and each pair of sequences.
#
#     Args:
#         depth (int): The number of sequences in the alignment being characterized.
#         node_sequences (dict): The indices of each of the sequences from the overall alignment present in each specific
#         node.
#         match_mismatch_table (MatchMismatchTable): A table with the match and mismatch status of all pairs of sequences
#         for all positions in the alignment.
#         shareable_locks (dict): A dictionary full of multiprocessing.Lock objects to be used for synchronization of the
#         characterization process over all positions and match/mismatch status.
#         shareable_dict (multiprocessing.Manager.dict): A thread safe dictionary where the individual processes can
#         update FrequencyTables for every node depending on match/mismatch status, position, and character.
#         processes (int): The number of processes being used by the initialized pool.
#     """
#     global aln_depth, node_seq_ind, mm_table, freq_lock, freq_tables, cpu_count
#     aln_depth = depth
#     node_seq_ind = node_sequences
#     mm_table = match_mismatch_table
#     cpu_count = processes
#
#
# def characterization_mm(pos):
#     """
#     Characterization Match Mismatch
#
#     This function accepts a position and characterizes its matches and mismatches for all possible pairs of sequences
#     over all nodes in the phylogenetic tree.
#
#     Args:
#         pos (int/tuple): The alignment position to process, this will be used to check what the match/mismatch status
#         of the alignment as well as the character at every pair of sequence comparisons using the MatchMismatchTable for
#         the full alignment.
#     Return:
#         int/tuple: The position is returned to keep track of which position has been most recently processed (in the
#         multiprocessing context).
#     """
#     print('CHARACTERIZATION MM: {}'.format(pos))
#     char_dict = {}
#     for seq1 in range(aln_depth):
#         for seq2 in range(seq1 + 1, aln_depth):
#             status, char = mm_table.get_status_and_character(pos=pos, seq_ind1=seq1, seq_ind2=seq2)
#             for node in node_seq_ind.keys():
#                 if node not in char_dict:
#                     char_dict[node] = {'match': {}, 'mismatch': {}}
#                 if seq1 in node_seq_ind[node] and seq2 in node_seq_ind[node]:
#                     if char not in char_dict[node][status]:
#                         char_dict[node][status][char] = 0
#                     char_dict[node][status][char] += 1
#     return pos, char_dict


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
        fn = os.path.join(out_dir, '{}_{}_{}_{}_score.npz'.format(node_name, pos_type, score_type, metric))
        if not os.path.isfile(fn):
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
    elif low_memory:
        if not os.path.isfile(mat):
            raise ValueError('File path is not valid: {}'.format(mat))
        load = np.load(mat)
        mat = load['mat']
    else:
        pass
    return mat


def init_trace_groups(scorer, pos_specific, pair_specific, match_mismatch, u_dict, low_memory, unique_dir):
    """
    Init Trace Pool

    This function initializes a pool of workers with shared resources so that they can quickly perform the group level
    scoring for the trace algorithm.

    Args:
        scorer (PositionalScorer): A scorer used to compute the group and rank scores according to a given metric.
        pos_specific (bool): Whether or not to characterize single positions.
        pair_specific (bool): Whether or not to characterize pairs of positions.
        match_mismatch (bool): Whether or not characterization was of the match_mismatch type (comparison of all
        possible transitions for a given position/pair, or considering each position/pair only once for each sequence
        present).
        u_dict (dict): A dictionary containing the node characterizations.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
        unique_dir (str/path): The directory where group score vectors/matrices can be written.
    """
    global pos_scorer, single, pair, mm_analysis, unique_nodes, low_mem, u_dir
    pos_scorer = scorer
    single = pos_specific
    pair = pair_specific
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
    # Check whether the group score has already been saved to file.
    single_check, single_fn = check_numpy_array(low_memory=low_mem, node_name=node_name, pos_type='single',
                                                score_type='group', metric=pos_scorer.metric, out_dir=u_dir)
    pair_check, pair_fn = check_numpy_array(low_memory=low_mem, node_name=node_name, pos_type='pair',
                                            score_type='group', metric=pos_scorer.metric, out_dir=u_dir)
    # If the file(s) were found set the return values for this node.
    if low_mem and (single_check >= single) and (pair_check >= pair):
        single_group_score = single_fn if single else None
        pair_group_score = pair_fn if pair else None
    else:
        # Using the provided scorer and the characterization for the node found in unique nodes, compute the
        # group score
        if single:
            if mm_analysis:
                single_freq_table = {'match': load_freq_table(freq_table=unique_nodes[node_name]['match'],
                                                              low_memory=low_mem),
                                     'mismatch': load_freq_table(freq_table=unique_nodes[node_name]['mismatch'],
                                                                 low_memory=low_mem)}
            else:
                single_freq_table = load_freq_table(freq_table=unique_nodes[node_name]['single'], low_memory=low_mem)
            single_group_score = pos_scorer.score_group(single_freq_table)
            single_group_score = save_numpy_array(mat=single_group_score, out_dir=u_dir, low_memory=low_mem,
                                                  node_name=node_name, pos_type='single',
                                                  metric=pos_scorer.metric, score_type='group')
        else:
            single_group_score = None
        if pair:
            if mm_analysis:
                pair_freq_table = {'match': load_freq_table(freq_table=unique_nodes[node_name]['match'],
                                                            low_memory=low_mem),
                                   'mismatch': load_freq_table(freq_table=unique_nodes[node_name]['mismatch'],
                                                               low_memory=low_mem)}
            else:
                pair_freq_table = load_freq_table(freq_table=unique_nodes[node_name]['pair'], low_memory=low_mem)
            pair_group_score = pos_scorer.score_group(pair_freq_table)
            pair_group_score = save_numpy_array(mat=pair_group_score, out_dir=u_dir, low_memory=low_mem,
                                                node_name=node_name, pos_type='pair',
                                                metric=pos_scorer.metric, score_type='group')
        else:
            pair_group_score = None
    # Store the group scores so they can be retrieved when the pool completes processing
    components = {'single_scores': single_group_score, 'pair_scores': pair_group_score}
    return node_name, components


def init_trace_ranks(scorer, pos_specific, pair_specific, a_dict, u_dict, low_memory,
                     unique_dir):
    """
    Init Trace Ranks

    This function initializes a pool of workers with shared resources so that they can quickly perform the rank level
    scoring for the trace algorithm.

    Args:
        scorer (PositionalScorer): A scorer used to compute the group and rank scores according to a given metric.
        pos_specific (bool): Whether or not to characterize single positions.
        pair_specific (bool): Whether or not to characterize pairs of positions.
        a_dict (dict): The rank and group assignments for nodes in the tree.
        u_dict (dict): A dictionary containing the node characterizations and group level scores for those nodes.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
        unique_dir (str/path): The directory where group score vectors/matrices can be loaded from and where rank
        scores can be written.
    """
    global pos_scorer, single, pair, assignments, unique_nodes, low_mem, u_dir
    pos_scorer = scorer
    single = pos_specific
    pair = pair_specific
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
    # Check whether the group score has already been saved to file.
    single_check, single_fn = check_numpy_array(low_memory=low_mem, node_name=rank, pos_type='single',
                                                score_type='rank', metric=pos_scorer.metric, out_dir=u_dir)
    pair_check, pair_fn = check_numpy_array(low_memory=low_mem, node_name=rank, pos_type='pair',
                                            score_type='rank', metric=pos_scorer.metric, out_dir=u_dir)
    # If the file(s) were found set the return values for this rank.
    if low_mem and (single_check >= single) and (pair_check >= pair):
        single_rank_scores = single_fn if single else None
        pair_rank_scores = pair_fn if pair else None
    else:
        # Retrieve all group scores for this rank
        if single:
            single_group_scores = np.zeros(pos_scorer.dimensions)
        else:
            single_group_scores = None
        if pair:
            pair_group_scores = np.zeros(pos_scorer.dimensions)
        else:
            pair_group_scores = None
        # For each group in the rank update the cumulative sum for the rank
        for g in assignments[rank]:
            node_name = assignments[rank][g]['node'].name
            if single:
                single_group_score = unique_nodes[node_name]['single_scores']
                single_group_score = load_numpy_array(mat=single_group_score, low_memory=low_mem)
                single_group_scores += single_group_score
            if pair:
                pair_group_score = unique_nodes[node_name]['pair_scores']
                pair_group_score = load_numpy_array(mat=pair_group_score, low_memory=low_mem)
                pair_group_scores += pair_group_score
        # Compute the rank score over the cumulative sum of group scores.
        if single:
            single_rank_scores = pos_scorer.score_rank(single_group_scores, rank)
            single_rank_scores = save_numpy_array(mat=single_rank_scores, out_dir=u_dir, low_memory=low_mem,
                                                  node_name=rank, pos_type='single', score_type='rank',
                                                  metric=pos_scorer.metric)
        else:
            single_rank_scores = None
        if pair:
            pair_rank_scores = pos_scorer.score_rank(pair_group_scores, rank)
            pair_rank_scores = save_numpy_array(mat=pair_rank_scores, out_dir=u_dir, low_memory=low_mem,
                                                node_name=rank, pos_type='pair', score_type='rank',
                                                metric=pos_scorer.metric)
        else:
            pair_rank_scores = None
    # Store the rank score so that it can be retrieved once the pool completes
    components = {'single_ranks': single_rank_scores, 'pair_ranks': pair_rank_scores}
    return rank, components
