"""
Created on May 23, 2019

@author: Daniel Konecki
"""
from time import sleep
from Queue import Queue


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

    def characterize_rank_groups(self):
        from time import time
        start = time()
        queue = Queue(maxsize=(self.aln.size * 2) - 1)
        for t in self.phylo_tree.tree.get_terminals():
            queue.put_nowait(t)
        for n in self.phylo_tree.traverse_bottom_up():
            if not n.is_terminal():
                queue.put_nowait(n)
        frequency_tables = {}
        queue_time = time()
        print('Queue initialized in: {} min'.format((queue_time - start) / 60.0))
        while queue.qsize() > 0:
            while1_start = time()
            node = queue.get_nowait()
            print('Processing node: {} min'.format(node.name))
            if node.is_terminal():
                print('Is Terminal')
                sub_aln = self.aln.generate_sub_alignment(sequence_ids=[node.name])
                pos_table, pair_table = sub_aln.characterize_positions(single=self.pos_specific,
                                                                       pair=self.pair_specific)
            else:
                print('Is Non-terminal')
                child1 = node.clades[0].name
                child2 = node.clades[1].name
                print('Children: {}, {}'.format(child1, child2))
                success = False
                component1 = None
                component2 = None
                while not success:
                    while2_start = time()
                    try:
                        component1 = frequency_tables[child1]
                        component2 = frequency_tables[child2]
                        success = True
                    except KeyError:
                        sleep(0.5)
                        exit()
                    while2_end = time()
                    print('Inner while took: {} min'.format((while2_end - while2_start) / 60.0))
                if self.pos_specific:
                    pos_table = component1['single'] + component2['single']
                else:
                    pos_table = None
                if self.pair_specific:
                    pair_table = component1['pair'] + component2['pair']
                else:
                    pair_table = None
            component = {'single': pos_table, 'pair': pair_table}
            frequency_tables[node.name] = component
            while1_end = time()
            print('Outer while loop took: {} min'.format((while1_end - while1_start) / 60.0))
        for rank in self.assignments:
            for group in self.assignments[rank]:
                self.assignments[rank][group].update(frequency_tables[self.assignments[rank][group]['node'].name])

    # def single_position_trace(self):
    #
    # def pair_position_trace(self):