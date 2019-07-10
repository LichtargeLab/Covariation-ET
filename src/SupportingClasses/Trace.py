"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os


class Trace(object):

    def __init__(self, aln, phylo_tree, position_specific=True, pair_specific=True, output_dir=None):
        """

        :param aln:
        :param phylo_tree:
        :param position_specific:
        :param pair_specific:
        """
        self.aln = aln
        self.tree = phylo_tree
        self.pos_specific = position_specific
        self.pair_specific = pair_specific
        self.unique_nodes = {}
        self.groups = {}
        self.final = {}
        if output_dir is None:
            self.out_dir = os.getcwd()
        else:
            self.out_dir = os.path.abspath(output_dir)