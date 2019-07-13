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
        start = time()
        queue = Queue(maxsize=(self.aln.size * 2) - 1)
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

    def trace(self, scorer, processes=1):
        start = time()
        # pos_type = None
        if scorer.position_size == 1:
            pos_type = 'single'
        elif scorer.position_size == 2:
            pos_type = 'pair'
        else:
            raise ValueError('Currently only scorers with size 1 (position specific) or 2 (pair specific) are valid.')
        final_scores = np.zeros(scorer.dimensions)
        for rank in sorted(self.assignments.keys()):
            group_scores = []
            for group in sorted(self.assignments[rank].keys()):
                group_scores.append(scorer.score_group(self.assignments[rank][group][pos_type]))
            group_scores = np.stack(group_scores, axis=0)
            rank_scores = scorer.score_rank(group_scores)
            final_scores += rank_scores
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
