"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
from queue import Empty
import pickle
from time import sleep, time
from Bio.Alphabet import Gapped
from scipy.stats import rankdata
from multiprocessing import Manager, Pool, Queue, Lock
from FrequencyTable import FrequencyTable
from utils import gap_characters, build_mapping
from EvolutionaryTraceAlphabet import MultiPositionAlphabet


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
        out_dir (str/path): Where results from a trace should be stored.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
    """

    def __init__(self, alignment, phylo_tree, group_assignments, position_specific=True, pair_specific=True,
                 output_dir=None, low_memory=False):
        """
        Initializer for Trace object.

        Args:
            alignment (SeqAlignment): The alignment for which to perform a trace analysis.
            phylo_tree (PhylogeneticTree): The tree based on the alignment to use during the trace analysis.
            group_assignments (dict): The group assignments for nodes in the tree.
            position_specific (bool): Whether or not to perform the trace for specific positions.
            pair_specific (bool): Whether or not to perform the trace for pairs of positions.
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
        if output_dir is None:
            self.out_dir = os.getcwd()
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            self.out_dir = output_dir
        self.low_memory = low_memory

    def characterize_rank_groups(self, processes=1, write_out_sub_aln=True, write_out_freq_table=True):
        """
        Characterize Rank Group

        This function iterates over the rank and group assignments and characterizes all positions for each sub
        alignment. Characterization consists of FrequencyTable objects which are added to the group_assignments
        dictionary provided at initialization (single position FrequencyTables are added under the key 'single', while
        FrequencyTAbles for pairs of positions are added under the key 'pair').

        Args:
            processes (int): The maximum number of sequences to use when performing this characterization.
            write_out_sub_aln (bool): Whether or not to write out the sub-alignments for each node while characterizing
            it.
            write_out_freq_table (bool): Whether or not to write out the frequency table for each node while
            characterizing it.
        """
        start = time()
        unique_dir = os.path.join(self.out_dir, 'unique_node_data')
        if not os.path.isdir(unique_dir):
            os.makedirs(unique_dir)
        single_alphabet = Gapped(self.aln.alphabet)
        single_size, _, single_mapping, single_reverse = build_mapping(alphabet=single_alphabet)
        pair_size = None
        pair_mapping = None
        pair_reverse = None
        single_to_pair = None
        if self.pair_specific:
            pair_alphabet = MultiPositionAlphabet(alphabet=Gapped(self.aln.alphabet), size=2)
            pair_size, _, pair_mapping, pair_reverse = build_mapping(alphabet=pair_alphabet)
            single_to_pair = {(single_mapping[char[0]], single_mapping[char[1]]): pair_mapping[char]
                              for char in pair_mapping if pair_mapping[char] < pair_size}
        visited = {}
        queue_components = Queue(maxsize=self.aln.size + processes)
        queue_inner = Queue(maxsize=self.aln.size - 1 + processes)
        components = False
        for r in sorted(self.assignments.keys(), reverse=True):
            for g in self.assignments[r]:
                node = self.assignments[r][g]['node']
                if not components:
                    queue_components.put_nowait(node.name)
                elif node.name not in visited:
                    queue_inner.put_nowait(node.name)
                else:
                    continue
                visited[node.name] = {'terminals': self.assignments[r][g]['terminals'],
                                      'descendants': self.assignments[r][g]['descendants']}
            if not components:
                components = True
        for p in range(processes):
            queue_components.put_nowait('STOP')
            queue_inner.put_nowait('STOP')
        pool_manager = Manager()
        frequency_tables = pool_manager.dict()
        tables_lock = Lock()
        pool = Pool(processes=processes, initializer=init_characterization_pool,
                    initargs=(single_size, single_mapping, single_reverse, pair_size, pair_mapping, pair_reverse,
                              single_to_pair, self.aln, self.pos_specific, self.pair_specific, queue_components,
                              queue_inner, visited, frequency_tables, tables_lock, unique_dir, self.low_memory,
                              write_out_sub_aln, write_out_freq_table, processes))
        pool.map_async(func=characterization, iterable=list(range(processes)))
        pool.close()
        pool.join()
        frequency_tables = dict(frequency_tables)
        self.unique_nodes = frequency_tables
        end = time()
        print('Characterization took: {} min'.format((end - start) / 60.0))

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
            processes (int): The maximum number of sequences to use when performing this characterization.
            gap_correction (float): If this value is set then after the trace has been performed positions in the
            alignment where the gap / alignment size ratio is greater than the gap_correction will have their final
            scores set to the highest (least important) value computed up to that point. If you do not want to perform
            this correction please set gap_correction to None.
        Returns:
            np.array: A properly dimensioned vector/matrix/tensor of importance scores for each position in the
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
        start = time()
        unique_dir = os.path.join(self.out_dir, 'unique_node_data')
        manager = Manager()
        # Generate group scores for each of the unique_nodes from the phylo_tree
        group_queue = Queue(maxsize=(self.aln.size * 2) - 1 + processes)
        visited = set([])
        for r in sorted(self.assignments.keys(), reverse=True):
            for g in self.assignments[r]:
                node = self.assignments[r][g]['node']
                if node.name not in visited:
                    group_queue.put_nowait(node.name)
                    visited.add(node.name)
        for i in range(processes):
            group_queue.put_nowait('STOP')
        group_dict = manager.dict()
        pool1 = Pool(processes=processes, initializer=init_trace_groups,
                     initargs=(group_queue, scorer, group_dict, self.pos_specific, self.pair_specific,
                               self.unique_nodes, self.low_memory, unique_dir))
        pool1.map_async(trace_groups, range(processes))
        pool1.close()
        pool1.join()
        # Update the unique_nodes dictionary with the group scores
        group_dict = dict(group_dict)
        for node_name in group_dict:
            self.unique_nodes[node_name].update(group_dict[node_name])
        group_time = time()
        print('Group processing completed in {} min'.format((group_time - start) / 60.0))
        # For each rank collect all group scores and compute a final rank score
        rank_queue = Queue(maxsize=self.aln.size + processes)
        for rank in sorted(self.assignments.keys(), reverse=True):
            rank_queue.put_nowait(rank)
        for i in range(processes):
            rank_queue.put_nowait('STOP')
        rank_dict = manager.dict()
        pool2 = Pool(processes=processes, initializer=init_trace_ranks,
                     initargs=(rank_queue, scorer, rank_dict, self.pos_specific, self.pair_specific, self.assignments,
                               self.unique_nodes, self.low_memory, unique_dir))
        pool2.map_async(trace_ranks, list(range(processes)))
        pool2.close()
        pool2.join()
        # Combine rank scores to generate a final score for each position
        self.rank_scores = dict(rank_dict)
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
        rank_time = time()
        print('Rank processing completed in {} min'.format((rank_time - group_time) / 60.0))
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
            gap_time = time()
            print('Gap correction completed in {} min'.format((gap_time - rank_time) / 60.0))
        self.final_scores = final_scores
        self.final_ranks, self.final_coverage = self._compute_rank_and_coverage(
            scores=self.final_scores, pos_size=scorer.position_size, rank_type=scorer.rank_type)
        end = time()
        print('Trace of with {} metric took: {} min'.format(scorer.metric, (end - start) / 60.0))
        return self.final_ranks, self.final_scores, self.final_coverage

    def _compute_rank_and_coverage(self, scores, pos_size, rank_type):
        """
        Compute Rank and Coverage

        This function generates rank and coverage values for a set of scores.

        Args:
            scores (np.array): A set of scores to rank and compute coverage for.
            pos_size (int): The dimensionality of the array (whether single, 1, positions or pair, 2, positions are
            being characterized).
            rank_type (str): Whether the optimal value of a set of scores is its 'max' or its 'min'.
        Returns:
            np.array: An array of ranks for the set of scores.
            np.array: An array of coverage scores (what percentile of values are at or below the given score).
        """
        if rank_type == 'max':
            weight = -1.0
        elif rank_type == 'min':
            weight = 1.0
        else:
            raise ValueError('No support for rank types other than max or min, {} provided'.format(rank_type))
        if pos_size == 1:
            indices = range(self.aln.seq_length)
            normalization = float(self.aln.seq_length)
            to_rank = scores * weight
            ranks = np.zeros(self.aln.seq_length)
            coverages = np.zeros(self.aln.seq_length)
        elif pos_size == 2:
            indices = np.triu_indices(self.aln.seq_length, k=1)
            normalization = float(len(indices[0]))
            to_rank = scores[indices] * weight
            ranks = np.zeros((self.aln.seq_length, self.aln.seq_length))
            coverages = np.zeros((self.aln.seq_length, self.aln.seq_length))
        else:
            raise ValueError('Ranking not supported for position sizes other than 1 or 2, {} provided'.format(pos_size))
        ranks[indices] = rankdata(to_rank, method='dense')
        coverages[indices] = rankdata(to_rank, method='max')
        coverages /= normalization
        return ranks, coverages


def init_characterization_pool(single_size, single_mapping, single_reverse, pair_size, pair_mapping, pair_reverse,
                               single_to_pair, alignment, pos_specific, pair_specific, queue1, queue2, components,
                               sharable_dict, sharable_lock, unique_dir, low_memory, write_out_sub_aln,
                               write_out_freq_table, processes):
    """
    Init Characterization Pool

    This function initializes a pool of workers with shared resources so that they can quickly characterize the sub
    alignments for each node in the phylogenetic tree.

    Args:
        single_size (int): The size of the single letter alphabet for the alignment being characterized (including the
        gap character).
        single_mapping (dict): A dictionary mapping the single letter alphabet to numerical positions.
        single_reverse (dict): A dictionary mapping numerical positions back to the single letter alphabet.
        pair_size (int): The size of the paired letter alphabet for the alignment being characterized (including the
        gap character).
        pair_mapping (dict): A dictionary mapping the paired letter alphabet to numerical positions.
        pair_reverse (dict): A dictionary mapping numerical positions back to the paired letter alphabet.
        single_to_pair (dict): A dictionary mapping single letter numeric positions to paired letter numeric positions.
        alignment (SeqAlignment): The alignment for which the trace is being computed, this will be used to generate
        sub alignments for the terminal nodes in the tree.
        pos_specific (bool): Whether or not to characterize single positions.
        pair_specific (bool): Whether or not to characterize pairs of positions.
        queue1 (multiprocessing.Queue): A queue which contains all nodes in the lowest rank (i.e. most nodes) of the
        phylogenetic tree over which the trace is being performed.
        queue2 (multiprocessing.Queue): A queue which contains all nodes in the phylogenetic tree for higher ranks over
        which the trace is being performed. The nodes should be inserted in an order which makes sense for the efficient
        characterization of nodes (i.e. leaves to root).
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
    global s_size
    s_size = single_size
    global s_map
    s_map = single_mapping
    global s_rev
    s_rev = single_reverse
    global p_size
    p_size = pair_size
    global p_map
    p_map = pair_mapping
    global p_rev
    p_rev = pair_reverse
    global s_to_p
    s_to_p = single_to_pair
    global aln
    aln = alignment
    global single
    single = pos_specific
    global pair
    pair = pair_specific
    global component_queue
    component_queue = queue1
    global node_queue
    node_queue = queue2
    global comps
    comps = components
    global freq_tables
    freq_tables = sharable_dict
    global freq_lock
    freq_lock = sharable_lock
    global u_dir
    u_dir = unique_dir
    global low_mem
    low_mem = low_memory
    global write_sub_aln
    write_sub_aln = write_out_sub_aln
    global write_freq_table
    write_freq_table = write_out_freq_table
    global cpu_count
    cpu_count = processes


def check_freq_table(low_memory, node_name, table_type, out_dir):
    """
    Check Frequency Table

    This function is used to check wither the desired FrequencyTable has already been produced and stored or not.

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


def characterization(processor):
    """
    Characterization

    This function pulls nodes from a queue in order to characterize them. If the node is in the lowest (closest to the
    leaves of the tree) rank being traced, then a sub-alignment is generated from the full alignment (provided
    by init_characterization_pool) and the individual positions (if specified by pos_specific in
    init_characterization_pool) and pairs of positions (if specified by pair_specific in init_characterization)
    are characterized for their nucleic/amino acid content. If the node is non-terminal then the dictionary of
    frequency tables (sharable_dict provided by init_characterization_pool) is accessed to retrieve the characterization
    for the nodes descendants; these are then summed. Each node that is completed by a worker process is
    added to the dictionary provided to init_characterization as sharable_dict, which is where results can be
    found after all processes finish.

    Args:
        processor (int): The processor in the pool which is being called.
    """
    # Track how many nodes this process has completed
    count = 0
    # Track how long a single component takes to process
    time_checked = False
    sleep_time = .1
    while True:
        # Retrieve the next node from the queue
        try:
            node_name = component_queue.get_nowait()
        except Empty:
            # If no node is available the queue is likely empty, repeat the loop checking the queue status.
            sleep(.1)
            continue
        # Check for the sentinel value, if it is reached break out of the process loop.
        if node_name == 'STOP':
            break
        # Check whether the alignment characterization has already been saved to file.
        single_check, single_fn = check_freq_table(low_memory=low_mem, node_name=node_name, table_type='single',
                                                   out_dir=u_dir)
        pair_check, pair_fn = check_freq_table(low_memory=low_mem, node_name=node_name, table_type='pair',
                                               out_dir=u_dir)
        # If the file(s) were found set the return values for this sub-alignment.
        if low_mem and (single_check >= single) and (pair_check >= pair):
            pos_table = single_fn if single else None
            pair_table = pair_fn if pair else None
        else:  # Otherwise generate the sub-alignment and characterize it
            # Generate the sub alignment for the current node.
            sub_aln = aln.generate_sub_alignment(sequence_ids=comps[node_name]['terminals'])
            # If specified write the alignment to file.
            if write_sub_aln:
                sub_aln.write_out_alignment(file_name=os.path.join(u_dir, '{}.fa'.format(node_name)))
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
            if not time_checked:
                sleep_time = (end - start) / float(cpu_count)
                time_checked = True
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
        count += 1
    # Run until no more nodes are available
    while True:
        # Retrieve the next node from the queue
        try:
            node_name = node_queue.get_nowait()
        except Empty:
            # If no node is available the queue is likely empty, repeat the loop checking the queue status.
            sleep(.1)
            continue
        # Check for the sentinel value, if it is reached break out of the process loop.
        if node_name == 'STOP':
            break
        # Check whether the alignment characterization has already been saved to file.
        single_check, single_fn = check_freq_table(low_memory=low_mem, node_name=node_name, table_type='single',
                                                   out_dir=u_dir)
        pair_check, pair_fn = check_freq_table(low_memory=low_mem, node_name=node_name, table_type='pair',
                                               out_dir=u_dir)
        # If the file(s) were found set the return values for this sub-alignment.
        if low_mem and (single_check >= single) and (pair_check >= pair):
            pos_table = single_fn if single else None
            pair_table = pair_fn if pair else None
        else:  # Otherwise generate the sub-alignment and characterize it
            sub_aln = aln.generate_sub_alignment(sequence_ids=comps[node_name]['descendants'])
            # If specified write the alignment to file.
            if write_sub_aln:
                sub_aln.write_out_alignment(file_name=os.path.join(u_dir, '{}.fa'.format(node_name)))
            # Since the node is non-terminal retrieve its descendants' characterizations and merge them to get the
            # parent characterization.
            descendants = set([d.name for d in comps[node_name]['descendants']])
            tries = 0
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
                    tries += 1
                    if (tries % 10) == 0:
                        print('Process {} stuck on {}, {} tries so far'.format(processor, node_name, tries))
                    sleep(sleep_time)
            print('Process {} took {} tries to get {} components'.format(processor, tries, node_name))
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
        count += 1
    print('Processor {} Completed {} node characterizations'.format(processor, count))


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


def init_trace_groups(group_queue, scorer, group_dict, pos_specific, pair_specific, u_dict, low_memory, unique_dir):
    """
    Init Trace Pool

    This function initializes a pool of workers with shared resources so that they can quickly perform the group level
    scoring for the trace algorithm.

    Args:
        group_queue (multiprocessing.Queue): A queue containing the nodes which  still need to be processed. Nodes are
        described by their name which should have been previously characterized.
        scorer (PositionalScorer): A scorer used to compute the group and rank scores according to a given metric.
        group_dict (multiprocessing.Manager.dict): A dictionary to hold the group scores for a node in the tree.
        pos_specific (bool): Whether or not to characterize single positions.
        pair_specific (bool): Whether or not to characterize pairs of positions.
        u_dict (dict): A dictionary containing the node characterizations.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
        unique_dir (str/path): The directory where group score vectors/matrices can be written.
    """
    global g_queue
    g_queue = group_queue
    global pos_scorer
    pos_scorer = scorer
    global g_dict
    g_dict = group_dict
    global single
    single = pos_specific
    global pair
    pair = pair_specific
    global unique_nodes
    unique_nodes = u_dict
    global low_mem
    low_mem = low_memory
    global u_dir
    u_dir = unique_dir


def trace_groups(processor):
    """
    Trace Groups

    A function which performs the group scoring part of the trace algorithm. It depends on the init_trace_groups
    function to set up the necessary shared resources. Node names are pulled from the group_queue and a group score is
    computed based on the FrequencyTable stored in unique_nodes for that node name. This is repeated until the queue is
    empty.

    Args:
        processor (int): Which processor is executing this function.
    """
    start = time()
    time_check = False
    sleep_time = .1
    group_count = 0
    # Continue until the queue is empty.
    while True:
        try:
            # Retrieve the next node
            node_name = g_queue.get_nowait()
        except Empty:
            # If no node is available the queue is likely empty, repeat the loop checking the queue status.
            sleep(sleep_time)
            continue
        # Check for the sentinel value, if it is reached break out of the process loop.
        if node_name == 'STOP':
            break
        inner_start = time()
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
                single_freq_table = load_freq_table(freq_table=unique_nodes[node_name]['single'],
                                                    low_memory=low_mem)
                single_group_score = pos_scorer.score_group(single_freq_table)
                single_group_score = save_numpy_array(mat=single_group_score, out_dir=u_dir, low_memory=low_mem,
                                                      node_name=node_name, pos_type='single',
                                                      metric=pos_scorer.metric, score_type='group')
            else:
                single_group_score = None
            if pair:
                pair_freq_table = load_freq_table(freq_table=unique_nodes[node_name]['pair'], low_memory=low_mem)
                pair_group_score = pos_scorer.score_group(pair_freq_table)
                pair_group_score = save_numpy_array(mat=pair_group_score, out_dir=u_dir, low_memory=low_mem,
                                                    node_name=node_name, pos_type='pair',
                                                    metric=pos_scorer.metric, score_type='group')
            else:
                pair_group_score = None
        # Store the group scores so they can be retrieved when the pool completes processing
        components = {'single_scores': single_group_score, 'pair_scores': pair_group_score}
        g_dict[node_name] = components
        group_count += 1
        inner_end = time()
        if not time_check:
            sleep_time = inner_end - inner_start
    end = time()
    print('Processor {} completed {} groups in {} min'.format(processor, group_count, (end - start) / 60.0))


def init_trace_ranks(rank_queue, scorer, rank_dict, pos_specific, pair_specific, a_dict, u_dict, low_memory,
                     unique_dir):
    """
    Init Trace Ranks

    This function initializes a pool of workers with shared resources so that they can quickly perform the rank level
    scoring for the trace algorithm.

    Args:
        rank_queue (multiprocessing.Queue): A queue containing the ranks to be processed.
        scorer (PositionalScorer): A scorer used to compute the group and rank scores according to a given metric.
        rank_dict (multiprocessing.Manager.dict): A dictionary to hold the scores for a given rank.
        pos_specific (bool): Whether or not to characterize single positions.
        pair_specific (bool): Whether or not to characterize pairs of positions.
        a_dict (dict): The rank and group assignments for nodes in the tree.
        u_dict (dict): A dictionary containing the node characterizations and group level scores for those nodes.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
        unique_dir (str/path): The directory where group score vectors/matrices can be loaded from and where rank
        scores can be written.
    """
    global r_queue
    r_queue = rank_queue
    global pos_scorer
    pos_scorer = scorer
    global r_dict
    r_dict = rank_dict
    global single
    single = pos_specific
    global pair
    pair = pair_specific
    global assignments
    assignments = a_dict
    global unique_nodes
    unique_nodes = u_dict
    global low_mem
    low_mem = low_memory
    global u_dir
    u_dir = unique_dir


def trace_ranks(processor):
    """
    Trace Ranks

    A function which performs the rank scoring part of the trace algorithm. It depends on the init_trace_ranks
    function to set up the necessary shared resources. A rank is pulled from the rank_queue and all group scores for
    that rank are gathered by determining the nodes using the assignments dictionary and their group scores using the
    unique_nodes dictionary. These scores are combined into a vector/matrix and the rank level score is computed using
    the provided scorer.

    Args:
        processor (int): Which processor is executing this function.
    """
    start = time()
    rank_count = 0
    while True:
        try:
            # Retrieve the next rank
            rank = r_queue.get_nowait()
        except Empty:
            # If no node is available the queue is likely empty, repeat the loop checking the queue status.
            sleep(0.1)
            continue
        # Check for the sentinel value, if it is reached break out of the process loop.
        if rank == 'STOP':
            break
        # Check whether the group score has already been saved to file.
        single_fn, single_check = check_numpy_array(low_memory=low_mem, node_name=rank, pos_type='single',
                                                    score_type='rank', metric=pos_scorer.metric, out_dir=u_dir)
        pair_fn, pair_check = check_numpy_array(low_memory=low_mem, node_name=rank, pos_type='pair',
                                                score_type='rank', metric=pos_scorer.metric, out_dir=u_dir)
        # If the file(s) were found set the return values for this rank.
        if low_mem and (single >= single_check) and (pair >= pair_check):
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
        r_dict[rank] = {'single_ranks': single_rank_scores, 'pair_ranks': pair_rank_scores}
        rank_count += 1
    end = time()
    print('Processor {} completed {} ranks in {} min'.format(processor, rank_count, (end - start) / 60.0))
