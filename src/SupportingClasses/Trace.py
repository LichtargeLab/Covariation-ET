"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import numpy as np
from Queue import Empty
from time import sleep, time
from multiprocessing import Manager, Pool, Queue


class Trace(object):
    """
    This class represents the fundamental behavior of the Evolutionary Trace algorithm developed by Olivier Lichtarge
    and expanded upon by members of his laboratory.

    Attributes:
        aln (SeqAlignment): The alignment for which a trace is being performed
        phylo_tree (PhylogeneticTree): The tree constructed from the alignment over which the trace is performed.
        assignments (dict): The rank and group assignments made based on the tree.
        position_specific (bool): Whether this trace will perform position specific analyses or not.
        pair_specific (bool): Whether this trace will perform pair specific analyses or not.
    """

    def __init__(self, alignment, phylo_tree, group_assignments, position_specific=True, pair_specific=True):
        """
        Initializer for Trace object.

        Args:
            alignment (SeqAlignment): The alignment for which to perform a trace analysis.
            phylo_tree (PhylogeneticTree): The tree based on the alignment to use during the trace analysis.
            group_assignments (dict): The group assignments for nodes in the tree.
            position_specific (bool): Whether or not to perform the trace for specific positions.
            pair_specific (bool): Whether or not to perform the trace for pairs of positions.
        """
        self.aln = alignment
        self.phylo_tree = phylo_tree
        self.assignments = group_assignments
        self.pos_specific = position_specific
        self.pair_specific = pair_specific

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
        # queue = Queue(maxsize=(self.aln.size * 2) - 1)
        queue = Queue(maxsize=np.sum(range(self.aln.size)))
        for t in self.phylo_tree.tree.get_terminals():
            queue.put_nowait(t)
        for n in self.phylo_tree.traverse_bottom_up():
            if not n.is_terminal():
                queue.put_nowait(n)
        pool_manager = Manager()
        frequency_tables = pool_manager.dict()
        pool = Pool(processes=processes, initializer=init_characterization_pool,
                    initargs=(self.aln, self.pos_specific, self.pair_specific, queue, frequency_tables))
        pool.map_async(func=characterization, iterable=list(range(processes)))
        pool.close()
        pool.join()
        frequency_tables = dict(frequency_tables)
        for rank in self.assignments:
            for group in self.assignments[rank]:
                self.assignments[rank][group].update(frequency_tables[self.assignments[rank][group]['node'].name])
        end = time()
        print('Characterization took: {} min'.format((end - start) / 60.0))

    # def trace(self, scorer, processes=1):
    #     start = time()
    #     # pos_type = None
    #     if scorer.position_size == 1:
    #         pos_type = 'single'
    #     elif scorer.position_size == 2:
    #         pos_type = 'pair'
    #     else:
    #         raise ValueError('Currently only scorers with size 1 (position specific) or 2 (pair specific) are valid.')
    #     final_scores = np.zeros(scorer.dimensions)
    #     for rank in sorted(self.assignments.keys()):
    #         group_scores = []
    #         for group in sorted(self.assignments[rank].keys()):
    #             group_scores.append(scorer.score_group(self.assignments[rank][group][pos_type]))
    #         group_scores = np.stack(group_scores, axis=0)
    #         rank_scores = scorer.score_rank(group_scores)
    #         final_scores += rank_scores
    #     final_scores += 1
    #     end = time()
    #     print('Trace of {} positions with {} metric took: {} min'.format(pos_type, scorer.metric, (end - start) / 60.0))
    #     return final_scores

    def trace(self, scorer, processes=1):
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
        # position_type = None
        if scorer.position_size == 1:
            position_type = 'single'
        elif scorer.position_size == 2:
            position_type = 'pair'
        else:
            raise ValueError('Currently only scorers with size 1 (position specific) or 2 (pair specific) are valid.')
        group_queue = Queue(maxsize=(self.aln.size * 2) - 1)
        rank_queue = Queue(maxsize=self.aln.size)
        manager = Manager()
        group_dict = manager.dict()
        rank_dict = manager.dict()
        for rank in sorted(self.assignments.keys(), reverse=True):
            for group in sorted(self.assignments[rank].keys(), reverse=True):
                group_queue.put_nowait((rank, group))
                group_dict[group] = []
        pool = Pool(processes=processes, initializer=init_trace_pool,
                    initargs=(position_type, group_queue, rank_queue, self.assignments, scorer, group_dict, rank_dict))
        pool.map_async(trace_sub, list(range(processes)))
        pool.close()
        pool.join()
        rank_dict = dict(rank_dict)
        final_scores = np.zeros(scorer.dimensions)
        for rank in rank_dict:
            final_scores += rank_dict[rank]
        final_scores += 1
        end = time()
        print('Trace of {} positions with {} metric took: {} min'.format(pos_type, scorer.metric, (end - start) / 60.0))
        return final_scores


def init_characterization_pool(alignment, pos_specific, pair_specific, queue, sharable_dict):
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
        if node.is_terminal():
            # If the node is terminal (a leaf) retrieve its sub_alignment and characterize it.
            sub_aln = aln.generate_sub_alignment(sequence_ids=[node.name])
            pos_table, pair_table = sub_aln.characterize_positions(single=single, pair=pair)
        else:
            # If a node is non-terminal retrieve its childrens' characterizations and merge them to get the parent
            # characterization.
            child1 = node.clades[0].name
            child2 = node.clades[1].name
            success = False
            component1 = None
            component2 = None
            while not success:
                # Attempt to retrieve the current node's childrens' data, sleep and try again if it is not already in
                # the dictionary, until success.
                try:
                    component1 = freq_tables[child1]
                    component2 = freq_tables[child2]
                    success = True
                except KeyError:
                    sleep(0.5)
            if single:
                pos_table = component1['single'] + component2['single']
            else:
                pos_table = None
            if pair:
                pair_table = component1['pair'] + component2['pair']
            else:
                pair_table = None
        # Store the current nodes characterization in the shared dictionary.
        tables = {'single': pos_table, 'pair': pair_table}
        freq_tables[node.name] = tables
        count += 1
    print('Processor {} Completed {} node characterizations'.format(processor, count))


def init_trace_pool(position_type, group_queue, rank_queue, a_dict, scorer, group_dict, rank_dict):
    """
    Init Trace Pool

    This function initializes a pool of workers with shared resources so that they can quickly perform the trace
    algorithm.

    Args:
        position_type (str): 'single' or 'pair' describing what kinds of positions are being traced.
        group_queue (multiprocessing.Queue): A queue containing the groups which still need to be processed. Groups are
        described as a tuple where the first element is the rank and the second element is the group.
        rank_queue (multiprocessing.Queue): A queue containing the ranks to be processed (the queue should begin empty
        and will be added to as groups are processed).
        a_dict (dict): The group assignments for nodes in the tree.
        scorer (PositionalScorer): A scorer used to compute the group and rank scores according to a given metric.
        group_dict (multiprocessing.Manager.dict): A dictionary to hold the group scores for a given rank.
        rank_dict (multiprocessing.Manager.dict): A dictionary to hold the scores for a given rank.
    """
    global pos_type
    pos_type = position_type
    global g_queue
    g_queue = group_queue
    global r_queue
    r_queue = rank_queue
    global assignments
    assignments = a_dict
    global pos_scorer
    pos_scorer = scorer
    global g_dict
    g_dict = group_dict
    global r_dict
    r_dict = rank_dict


def trace_sub(processor):
    """
    Trace Sub

    A function which performs the group and rank scoring part of the trace algorithm. It depends on the init_trace_pool
    function to set up the necessary shared resources. If all groups have been processed for a given rank then that
    rank is pulled from the rank_queue and scored. These scores are recorded in the rank_dict which is where they can be
    retrieved after the pool completes. If no ranks are available for processing then a group is pulled from the
    group_queue and scored.

    Args:
        processor (int): Which processor is executing this function.
    """
    start = time()
    group_count = 0
    rank_count = 0
    while (not g_queue.empty()) or (not r_queue.empty()):
        try:
            rank = r_queue.get_nowait()

            group_scores = np.stack(g_dict[rank], axis=0)
            rank_scores = pos_scorer.score_rank(group_scores)
            r_dict[rank] = rank_scores
            del g_dict[rank]
        except Empty:
            try:
                rank, group = g_queue.get_nowait()
            except Empty:
                continue
            group_score = pos_scorer.score_group(assignments[rank][group][pos_type])
            g_dict[rank].append(group_score)
            if rank == len(g_dict[rank]):
                r_queue.put_nowait(rank)
    end = time()
    print('Processor {} completed {} groups and {} ranks in {} min'.format(processor, group_count, rank_count,
                                                                           (end - start) / 60.0))
