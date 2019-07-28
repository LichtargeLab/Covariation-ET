"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
from Queue import Empty
import cPickle as pickle
from time import sleep, time
from multiprocessing import Manager, Pool, Queue
from utils import gap_characters


class Trace(object):
    """
    This class represents the fundamental behavior of the Evolutionary Trace algorithm developed by Olivier Lichtarge
    and expanded upon by members of his laboratory.

    Attributes:
        aln (SeqAlignment): The alignment for which a trace is being performed
        phylo_tree (PhylogeneticTree): The tree constructed from the alignment over which the trace is performed.
        assignments (dict): The rank and group assignments made based on the tree.
        unique_nodes (dict): A dictionary to track the unique nodes from the tree, this will be used for
        characterization and tracing to reduce the required computations.
        position_specific (bool): Whether this trace will perform position specific analyses or not.
        pair_specific (bool): Whether this trace will perform pair specific analyses or not.
        out_dir (str/path): Where results from a trace should be stored.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
    """

    def __init__(self, alignment, phylo_tree, group_assignments, position_specific=True, pair_specific=True,
                 output_dir=None, low_mem=False):
        """
        Initializer for Trace object.

        Args:
            alignment (SeqAlignment): The alignment for which to perform a trace analysis.
            phylo_tree (PhylogeneticTree): The tree based on the alignment to use during the trace analysis.
            group_assignments (dict): The group assignments for nodes in the tree.
            position_specific (bool): Whether or not to perform the trace for specific positions.
            pair_specific (bool): Whether or not to perform the trace for pairs of positions.
            output_dir (str/path): Where results from a trace should be stored.
            low_mem (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
            resources.
        """
        self.aln = alignment
        self.phylo_tree = phylo_tree
        self.assignments = group_assignments
        self.unique_nodes = None
        self.pos_specific = position_specific
        self.pair_specific = pair_specific
        if output_dir is None:
            self.out_dir = os.getcwd()
        else:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            self.out_dir = output_dir
        self.low_memory = low_mem

    def characterize_rank_groups(self, processes=1):
        """
        Characterize Rank Group

        This function iterates over the rank and group assignments and characterizes all positions for each sub
        alignment. Characterization consists of FrequencyTable objects which are added to the group_assignments
        dictionary provided at initialization (single position FrequencyTables are added under the key 'single', while
        FrequencyTAbles for pairs of positions are added under the key 'pair').

        Args:
            processes (int): The maximum number of sequences to use when performing this characterization.
        """
        start = time()
        unique_dir = os.path.join(self.out_dir, 'unique_node_data')
        if not os.path.isdir(unique_dir):
            os.makedirs(unique_dir)
        queue = Queue(maxsize=(self.aln.size * 2) - 1)
        for t in self.phylo_tree.tree.get_terminals():
            queue.put_nowait(t)
        for n in self.phylo_tree.traverse_bottom_up():
            if not n.is_terminal():
                queue.put_nowait(n)
        pool_manager = Manager()
        frequency_tables = pool_manager.dict()
        pool = Pool(processes=processes, initializer=init_characterization_pool,
                    initargs=(self.aln, self.pos_specific, self.pair_specific, queue, frequency_tables, unique_dir,
                              self.low_memory))
        pool.map_async(func=characterization, iterable=list(range(processes)))
        pool.close()
        pool.join()
        frequency_tables = dict(frequency_tables)
        self.unique_nodes = frequency_tables
        end = time()
        print('Characterization took: {} min'.format((end - start) / 60.0))

    def trace(self, scorer, gap_correction=0.6, processes=1, del_intermediate=False):
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
            del_intermediate (bool): If low_memory is set this determines whether the serialized data will be deleted
            after the computation is complete.
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
        group_queue = Queue(maxsize=(self.aln.size * 2) - 1)
        for node_name in self.unique_nodes:
            group_queue.put_nowait(node_name)
        group_dict = manager.dict()
        pool1 = Pool(processes=processes, initializer=init_trace_groups,
                     initargs=(group_queue, scorer, group_dict, self.pos_specific, self.pair_specific,
                               self.unique_nodes, unique_dir))
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
        rank_queue = Queue(maxsize=self.aln.size)
        for rank in sorted(self.assignments.keys(), reverse=True):
            rank_queue.put_nowait(rank)
        rank_dict = manager.dict()
        pool2 = Pool(processes=processes, initializer=init_trace_ranks,
                     initargs=(rank_queue, scorer, rank_dict, self.pos_specific, self.pair_specific, self.assignments,
                               self.unique_nodes))
        pool2.map_async(trace_ranks, list(range(processes)))
        pool2.close()
        pool2.join()
        # Combine rank scores to generate a final score for each position
        rank_dict = dict(rank_dict)
        final_scores = np.zeros(scorer.dimensions)
        for rank in rank_dict:
            if self.pair_specific:
                rank_scores = rank_dict[rank]['pair_ranks']
            elif self.pos_specific:
                rank_scores = rank_dict[rank]['single_ranks']
            else:
                pass
            rank_scores = load_numpy_array(mat=rank_scores, low_memory=self.low_memory)
            final_scores += rank_scores
        final_scores += 1
        rank_time = time()
        print('Rank processing completed in {} min'.format((rank_time - group_time) / 60.0))
        # Perform gap correction, if a column gap content threshold has been set.
        if gap_correction is not None:
            pos_type = 'single' if scorer.position_size == 1 else 'pair'
            # The FrequencyTable for the root node is used because it characterizes all sequences and its depth is equal
            # to the alignment size. The resulting gap frequencies are equivalent to gap count / alignment size.
            root_node_name = self.phylo_tree.tree.root.name
            freq_table = self.unique_nodes[root_node_name][pos_type]
            gap_chars = list(set(freq_table.alphabet.letters).intersection(gap_characters))
            if len(gap_chars) > 1:
                raise ValueError('More than one gap character present in alignment alphabet! {}'.format(gap_chars))
            gap_char = gap_chars[0]
            max_rank_score = np.max(final_scores)
            for i in range(freq_table.sequence_length):
                if freq_table.get_frequency(pos=i, char=gap_char) > gap_correction:
                    final_scores[i] = max_rank_score
            gap_time = time()
            print('Gap correction completed in {} min'.format((gap_time - rank_time) / 60.0))
        end = time()
        print('Trace of with {} metric took: {} min'.format(scorer.metric, (end - start) / 60.0))
        return final_scores


def init_characterization_pool(alignment, pos_specific, pair_specific, queue, sharable_dict, unique_dir, low_memory):
    """
    Init Characterization Pool

    This function initializes a pool of workers with shared resources so that they can quickly characterize the sub
    alignments for each node in the phylogenetic tree.

    Args:
        alignment (SeqAlignment): The alignment for which the trace is being computed, this will be used to generate
        sub alignments for the terminal nodes in the tree.
        pos_specific (bool): Whether or not to characterize single positions.
        pair_specific (bool): Whether or not to characterize pairs of positions.
        queue (multiprocessing.Queue): A queue which contains all nodes in the phylogenetic tree over which the trace is
        being performed. The nodes should be inserted in an order which makes sense for the efficient characterization
        of nodes (i.e. leaves to root).
        sharable_dict (multiprocessing.Manager.dict): A thread safe dictionary where the individual processes can
        deposit the characterization of each node and retrieve characterizations of children needed for larger nodes.
        unique_dir (str/path): The directory where sub-alignment and frequency table files can be written.
        low_memory (bool): Whether or not to serialize matrices used during the trace to avoid exceeding memory
        resources.
    """
    global aln
    aln = alignment
    global single
    single = pos_specific
    global pair
    pair = pair_specific
    global node_queue
    node_queue = queue
    global freq_tables
    freq_tables = sharable_dict
    global u_dir
    u_dir = unique_dir
    global low_mem
    low_mem = low_memory


def save_freq_table(freq_table, low_memory, node_name, out_dir):
    """
    Save Frequency Table

    This function returns either the FrequencyTable that was provided to it or the path where that FrequencyTable has
    been stored if the low memory option has been set. The FrequencyTable will be saved to a file in the out_dir with
    name {}_freq)table.pkl where {} is filled in with the value passed to node_name.

    Args:
        freq_table (FrequencyTable): The FrequencyTable instance to either serialize or pass through.
        low_memory (bool): Whether or not to save the FrequencyTable or just pass it through.
        node_name (str): A string which will be used in naming the save file.
        out_dir (str): The path to a directory where the FrequencyTable should be saved.
    Returns:
        FrequencyTable/str: Either the provided FrequencyTable (if low_memory is False) or a string providing the path
        to a file where the FrequencyTable has been saved.
    """
    if low_memory:
        fn = os.path.join(out_dir, '{}_freq_table.pkl'.format(node_name))
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
    if low_memory and not isinstance(freq_table, str):
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

    This function pulls nodes from a queue in order to characterize them. If the node is terminal then a sub-alignment
    is generated from the full alignment (provided by init_characterization_pool) and the individual positions (if
    specified by pos_specific in init_characterization_pool) and pairs of positions (if specified by pair_specific in
    init_characterization) are characterized for their nucleic/amino acid content. If the node is non-terminal then the
    dictionary of frequency tables (sharable_dict provided by init_characterization_pool) is accessed to retrieve the
    characterization for the nodes children; these are then summed. Each node that is completed by a worker process is
    added to the dictionary provided to init_characterization as sharable_dict, which is where results can be found
    after all processes finish.

    Args:
        processor (int): The processor in the pool which is being called.
    """
    # Track how many nodes this process has completed
    count = 0
    # Run until no more nodes are available
    while not node_queue.empty():
        # Retrieve the next node from the queue
        try:
            node = node_queue.get_nowait()
        except Empty:
            # If no node is available the queue is likely empty, repeat the loop checking the queue status.
            continue
        sub_aln = aln.generate_sub_alignment(sequence_ids=[x.name for x in node.get_terminals()])
        sub_aln.write_out_alignment(file_name=os.path.join(u_dir, '{}.fa'.format(node.name)))
        if node.is_terminal():
            # If the node is terminal (a leaf) retrieve its sub_alignment and characterize it.
            # sub_aln = aln.generate_sub_alignment(sequence_ids=[node.name])
            pos_table, pair_table = sub_aln.characterize_positions(single=single, pair=pair)
        else:
            # If a node is non-terminal retrieve its childrens' characterizations and merge them to get the parent
            # characterization.
            child1 = node.clades[0].name
            child2 = node.clades[1].name
            success = False
            component1 = None
            component2 = None
            tries = 0
            while not success:
                # Attempt to retrieve the current node's childrens' data, sleep and try again if it is not already in
                # the dictionary, until success.
                try:
                    component1 = freq_tables[child1]
                    component2 = freq_tables[child2]
                    success = True
                except KeyError:
                    sleep(0.5)
                    tries += 1
                    if tries > 10:
                        print('Process {} stuck on {}, {} tries so far'.format(processor, node.name, tries))
            print('Process {} took {} tries to get {} components'.format(processor, tries, node.name))
            if single:
                pos_table = (load_freq_table(component1['single'], low_memory=low_mem) +
                             load_freq_table(component2['single'], low_memory=low_mem))
            else:
                pos_table = None
            if pair:
                pair_table = (load_freq_table(component1['pair'], low_memory=low_mem) +
                              load_freq_table(component2['pair'], low_memory=low_mem))
            else:
                pair_table = None
        # Compute frequencies for the FrequencyTables generated for the current node
        if single:
            pos_table.to_csv(os.path.join(u_dir, '{}_position_freq_table.tsv'.format(node.name)))
            pos_table = save_freq_table(freq_table=pos_table, low_memory=low_mem,
                                        node_name='{}_single'.format(node.name), out_dir=u_dir)
        if pair:
            pair_table.to_csv(os.path.join(u_dir, '{}_pair_freq_table.tsv'.format(node.name)))
            pair_table = save_freq_table(freq_table=pair_table, low_memory=low_mem,
                                         node_name='{}_pair'.format(node.name), out_dir=u_dir)
        # Store the current nodes characterization in the shared dictionary.
        tables = {'single': pos_table, 'pair': pair_table}
        freq_tables[node.name] = tables
        count += 1
    print('Processor {} Completed {} node characterizations'.format(processor, count))


def save_numpy_array(mat, out_dir, name, low_memory):
    """
    Save Numpy Array

    This function returns either the np.array that was provided to it or the path where that np.array has been stored if
    the low memory option was set. The FrequencyTable will be saved to a file in the out_dir with
    name {}_freq)table.pkl where {} is filled in with the value passed to node_name.

    Args:
        mat (np.array): The np.array instance to either serialize or pass through.
        low_memory (bool): Whether or not to save the FrequencyTable or just pass it through.
        name (str): A string which will be used in naming the save file.
        out_dir (str): The path to a directory where the np.array should be saved.
    Returns:
        np.array/str: Either the provided np.array (if low_memory is False) or a string providing the path to a file
        where the np.array has been saved.
    """
    if low_memory:
        fn = os.path.join(out_dir, '{}.npz'.format(name))
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
    if low_memory and not isinstance(mat, str):
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
    group_count = 0
    while not g_queue.empty():
        try:
            # Retrieve the next node
            node_name = g_queue.get_nowait()
            # Using the provided scorer and the characterization for the node found in unique nodes, compute the group
            # score
            if single:
                single_freq_table = load_freq_table(freq_table=unique_nodes[node_name]['single'], low_memory=low_mem)
                single_group_score = pos_scorer.score_group(single_freq_table)
                single_group_score = save_numpy_array(mat=single_group_score, out_dir=u_dir, low_memory=low_mem,
                                                      name='{}_single_group_score'.format(node_name))
            else:
                single_group_score = None
            if pair:
                pair_freq_table = load_freq_table(freq_table=unique_nodes[node_name]['pair'], low_memory=low_mem)
                pair_group_score = pos_scorer.score_group(pair_freq_table)
                pair_group_score = save_numpy_array(mat=pair_group_score, out_dir=u_dir, low_memory=low_mem,
                                                    name='{}_pair_group_score'.format(node_name))
            else:
                pair_group_score = None
            # Store the group scores so they can be retrieved when the pool completes processing
            components = {'single_scores': single_group_score, 'pair_scores': pair_group_score}
            g_dict[node_name] = components
            group_count += 1
        except Empty:
            continue
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
    unique_nodes dictionary. These scores are combined into a matrix/tensor and the rank level score is computed using
    the provided scorer.

    Args:
        processor (int): Which processor is executing this function.
    """
    start = time()
    group_count = 0
    rank_count = 0
    while not r_queue.empty():
        try:
            # Retrieve the next rank
            rank = r_queue.get_nowait()
            # Retrieve all group scores for that rank
            single_group_scores = []
            pair_group_scores = []
            for g in assignments[rank]:
                node_name = assignments[rank][g]['node'].name
                if single:
                    single_group_score = unique_nodes[node_name]['single_scores']
                    single_group_score = load_numpy_array(mat=single_group_score, low_memory=low_mem)
                    single_group_scores.append(single_group_score)
                if pair:
                    pair_group_score = unique_nodes[node_name]['pair_scores']
                    pair_group_score = load_numpy_array(mat=pair_group_score, low_memory=low_mem)
                    pair_group_scores.append(pair_group_score)
            # Combine all group scores into a matrix/tensor and compute the rank score
            if single:
                single_group_scores = np.stack(single_group_scores, axis=0)
                single_rank_scores = pos_scorer.score_rank(single_group_scores)
                single_rank_scores = save_numpy_array(mat=single_rank_scores, out_dir=u_dir, low_memory=low_mem,
                                                      name='{}_single_rank_score'.format(rank))
            else:
                single_rank_scores = None
            if pair:
                pair_group_scores = np.stack(pair_group_scores, axis=0)
                pair_rank_scores = pos_scorer.score_rank(pair_group_scores)
                pair_rank_scores = save_numpy_array(mat=pair_rank_scores, out_dir=u_dir, low_memory=low_mem,
                                                    name='{}_pair_rank_score'.format(rank))
            else:
                pair_rank_scores = None
            # Store the rank score so that it can be retrieved once the pool completes
            r_dict[rank] = {'single_ranks': single_rank_scores, 'pair_ranks': pair_rank_scores}
            rank_count += 1
        except Empty:
            continue
    end = time()
    print('Processor {} completed {} groups and {} ranks in {} min'.format(processor, group_count, rank_count,
                                                                           (end - start) / 60.0))
