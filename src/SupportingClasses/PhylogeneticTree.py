"""
Created on June 3, 2019

@author: Daniel Konecki
"""
import os
import heapq
import numpy as np
import cPickle as pickle
from Bio.Phylo import read, write
from sklearn.cluster import AgglomerativeClustering
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from AlignmentDistanceCalculator import AlignmentDistanceCalculator


class PhylogeneticTree(object):
    """

    """

    def __init__(self, alignment, model='identity', protein=True, skip_letters=None, et_distance_model=False,
                 tree_building_method='upgma', tree_building_args={}):
        self.original_alignment = alignment
        # self.non_gap_alignment = alignment.remove_gaps()
        self.distance_model = model
        self.protein_aln = protein
        self.skip_letters = skip_letters
        self.et_distance = et_distance_model
        self.distance_matrix = None
        self.tree_method = tree_building_method
        self.tree_args = tree_building_args
        self.tree = None
        self.node_data = {}

    def _calculate_distance_matrix(self):
        """
        """
        calculator = AlignmentDistanceCalculator(protein=self.protein_aln, model=self.distance_model,
                                                 skip_letters=self.skip_letters)
        if self.et_distance:
            _, distance_matrix, _, _ = calculator.get_et_distance(self.original_alignment.alignment)
        else:
            distance_matrix = calculator.get_distance(self.original_alignment.alignment)
        self.distance_matrix = distance_matrix

    def __custom_tree(self, tree_path):
        """
        """
        custom_tree = read(file=tree_path, format='newick')
        return custom_tree

    def __upgma_tree(self):
        """
        """
        constructor = DistanceTreeConstructor()
        upgma_tree = constructor.upgma(distance_matrix=self.distance_matrix)
        return upgma_tree

    def __agglomerative_clustering(self, cache_dir=None, affinity='euclidean', linkage='ward'):
        """
        References:
            The solution for converting an agglomerative clustering tree from sklearn into a Newick formatted tree was
            taken from the following StackOverflow discussion, the solution used was provided by user: lucianopaz
            https://stackoverflow.com/questions/29127013/plot-dendrogram-using-sklearn-agglomerativeclustering
        """
        ml_model = AgglomerativeClustering(affinity=affinity, linkage=linkage, n_clusters=self.original_alignment.size,
                                           memory=cache_dir, compute_full_tree=True)
        ml_model.fit(np.array(self.distance_matrix))
        newick_tree_string = convert_agglomerative_clustering_to_newick_tree(
            clusterer=ml_model, labels=self.original_alignment.seq_order, distance_matrix=self.distance_matrix)
        newick_fn = os.path.join(cache_dir, 'joblib', 'agg_clustering_{}_{}.newick'.format(affinity, linkage))
        with open(newick_fn, 'wb') as newick_handle:
            newick_handle.write(newick_tree_string)
        agg_clustering_tree = read(file=newick_fn, format='newick')
        return agg_clustering_tree

    def _construct_tree(self):
        method_dict = {'agglomerative': self.__agglomerative_clustering, 'upgma': self.__upgma_tree,
                       'custom': self.__custom_tree}
        self.tree = method_dict[self.tree_method](**self.tree_args)

    def traverse_top_down(self):
        """
                Traverse Base Tree

                This method generates a dictionary assigning sequences to clusters at each branching level (from 1 the root, to
                n the number of sequences) of the provided tree. This is useful for methods where a rooted Bio.Phylo.BaseTree
                object is used.

                Args:
                    tree (Bio.Phylo.BaseTree): The tree to traverse and represent as a dictionary.
                Returns:
                    dict: A nested dictionray where the first layer has keys corresponding to branching level and the second
                    level has keys corresponding to specific branches within that level and the values are a list of the
                    sequence identifiers in that branch.
                """

        def node_cmp(x, y):
            """
            Node Comparison

            This method is provided such that lists of nodes are sorted properly for the intended behavior of the UPGMA
            tree traversal.

            Args:
                x (Bio.Phylo.BaseTree.Clade): Left object in comparison.
                y (Bio.Phylo.BaseTree.Clade): Right object for comparison.
            Returns:
                int: Old comparator behavior, i.e. 1 x comes before y and -1 if y comes before x. If the two Clades are
                equal according to this comparison then 0 is returned.
            """
            if x.is_terminal() and not y.is_terminal():
                return -1
            elif not x.is_terminal() and y.is_terminal():
                return 1
            else:
                if x.total_branch_length() > y.total_branch_length():
                    return -1
                elif x.total_branch_length() < y.total_branch_length():
                    return 1
                else:
                    return 0

        # Traverse the tree
        # yield self.tree.root
        max_path_length = self.tree.root.total_branch_length()
        internal_nodes_to_process = []
        # leaf_nodes_to_process = []
        heapq.heappush(internal_nodes_to_process, (self.tree.root.branch_length, self.tree.root))
        while len(internal_nodes_to_process) > 0:
            # print(internal_nodes_to_process)
            curr_branch_len, curr_node = heapq.heappop(internal_nodes_to_process)
            # if curr_node.is_terminal():
            print(curr_node.name, curr_branch_len)
            yield curr_node
            # else:
            for node in sorted(curr_node.clades, cmp=node_cmp):
                if node.is_terminal():
                    path_length = max_path_length + curr_branch_len + node.branch_length
                    heapq.heappush(internal_nodes_to_process, (path_length, node))
                else:
                    # yield node
                    heapq.heappush(internal_nodes_to_process, (curr_branch_len + node.branch_length, node))
            # for node in nodes_to_process:
            #     current_cluster.append(node)
            # current_cluster.sort(cmp=node_cmp)
            # print([(x.name, x.total_branch_length()) for x in current_cluster])
            # nearest_node = current_cluster.pop()
            # if not nearest_node.is_terminal():
            #     nodes_to_process = nearest_node.clades
            #     print([(x.name, x.total_branch_length()) for x in nodes_to_process])
            # else:
            #     nodes_to_process = []
            # yield nearest_node

    def traverse_bottom_up(self):
        """
                Traverse Base Tree

                This method generates a dictionary assigning sequences to clusters at each branching level (from 1 the root, to
                n the number of sequences) of the provided tree. This is useful for methods where a rooted Bio.Phylo.BaseTree
                object is used.

                Args:
                    tree (Bio.Phylo.BaseTree): The tree to traverse and represent as a dictionary.
                Returns:
                    dict: A nested dictionray where the first layer has keys corresponding to branching level and the second
                    level has keys corresponding to specific branches within that level and the values are a list of the
                    sequence identifiers in that branch.
                """

        def node_cmp(x, y):
            """
            Node Comparison

            This method is provided such that lists of nodes are sorted properly for the intended behavior of the UPGMA
            tree traversal.

            Args:
                x (Bio.Phylo.BaseTree.Clade): Left object in comparison.
                y (Bio.Phylo.BaseTree.Clade): Right object for comparison.
            Returns:
                int: Old comparator behavior, i.e. 1 x comes before y and -1 if y comes before x. If the two Clades are
                equal according to this comparison then 0 is returned.
            """
            if x.is_terminal() and not y.is_terminal():
                return 1
            elif not x.is_terminal() and y.is_terminal():
                return -1
            else:
                if x.total_branch_length > y.total_branch_length:
                    return -1
                elif x.total_branch_length < y.total_branch_length:
                    return 1
                else:
                    return 0

        # Traverse the tree
        nodes_to_process = [self.tree.root.get_terminals()]
        current_cluster = []
        while len(nodes_to_process) > 0 or len(current_cluster) > 0:
            for node in nodes_to_process:
                current_cluster.append(node)
            current_cluster.sort(cmp=node_cmp)
            nearest_node = current_cluster.pop()
            if nearest_node.root != nearest_node:
                node_path = self.tree.get_path(nearest_node)
                nodes_to_process = [node_path[-2]]
            else:
                nodes_to_process = []
            yield nearest_node


def go_down_tree(children, n_leaves, x, leaf_labels, nodename, spanner):
    """
    go_down_tree(children,n_leaves,X,leaf_labels,nodename,spanner)

    Iterative function that traverses the subtree that descends from
    nodename and returns the Newick representation of the subtree.

    Input:
        children: AgglomerativeClustering.children_
        n_leaves: AgglomerativeClustering.n_leaves_
        x: parameters supplied to AgglomerativeClustering.fit
        leaf_labels: The label of each parameter array in X
        nodename: An int that is the intermediate node name whos
            children are located in children[nodename-n_leaves].
        spanner: Callable that computes the dendrite's span

    Output:
        ntree: A str with the Newick tree representation

    """
    nodeindex = nodename-n_leaves
    if nodename < n_leaves:
        return leaf_labels[nodeindex], np.array([x[nodeindex]])
    else:
        node_children = children[nodeindex]
        branch0, branch0samples = go_down_tree(children, n_leaves, x, leaf_labels, node_children[0], spanner)
        print('Branch0: {}'.format(branch0))
        branch1, branch1samples = go_down_tree(children, n_leaves, x, leaf_labels, node_children[1], spanner)
        print('Branch1: {}'.format(branch1))
        node = np.vstack((branch0samples, branch1samples))
        branch0span = spanner(branch0samples)
        branch1span = spanner(branch1samples)
        nodespan = spanner(node)
        branch0distance = nodespan-branch0span
        branch1distance = nodespan-branch1span
        nodename = '({branch0}:{branch0distance},{branch1}:{branch1distance})'.format(
            branch0=branch0, branch0distance=branch0distance, branch1=branch1, branch1distance=branch1distance)
        return nodename, node


def build_newick_tree(children, n_leaves, x, leaf_labels, spanner):
    """
    build_Newick_tree(children,n_leaves,X,leaf_labels,spanner)

    Get a string representation (Newick tree) from the sklearn
    AgglomerativeClustering.fit output.

    Input:
        children: AgglomerativeClustering.children_
        n_leaves: AgglomerativeClustering.n_leaves_
        x: parameters supplied to AgglomerativeClustering.fit
        leaf_labels: The label of each parameter array in X
        spanner: Callable that computes the dendrite's span

    Output:
        ntree: A str with the Newick tree representation

    """
    return go_down_tree(children, n_leaves, x, leaf_labels, len(children)+n_leaves-1, spanner)[0] + ';'


def get_cluster_spanner(agg_clusterer):
    """
    spanner = get_cluster_spanner(agg_clusterer)

    Input:
        agg_clusterer: sklearn.cluster.AgglomerativeClustering instance

    Get a callable that computes a given cluster's span. To compute
    a cluster's span, call spanner(cluster)

    The cluster must be a 2D numpy array, where the axis=0 holds
    separate cluster members and the axis=1 holds the different
    variables.

    """
    spanner = None
    if agg_clusterer.linkage == 'ward':
        if agg_clusterer.affinity == 'euclidean':
            spanner = lambda x: np.sum((x - agg_clusterer.pooling_func(x, axis=0)) ** 2)
    elif agg_clusterer.linkage == 'complete':
        if agg_clusterer.affinity == 'euclidean':
            spanner = lambda x: np.max(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
        elif agg_clusterer.affinity == 'l1' or agg_clusterer.affinity == 'manhattan':
            spanner = lambda x: np.max(np.sum(np.abs(x[:, None, :] - x[None, :, :]), axis=2))
        elif agg_clusterer.affinity == 'l2':
            spanner = lambda x: np.max(np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)))
        elif agg_clusterer.affinity == 'cosine':
            spanner = lambda x: (np.max(np.sum((x[:, None, :] * x[None, :, :])) /
                                        (np.sqrt(np.sum(x[:, None, :] * x[:, None, :], axis=2, keepdims=True)) *
                                         np.sqrt(np.sum(x[None, :, :] * x[None, :, :], axis=2, keepdims=True)))))
        else:
            raise AttributeError('Unknown affinity attribute value {0}.'.format(agg_clusterer.affinity))
    elif agg_clusterer.linkage == 'average':
        if agg_clusterer.affinity == 'euclidean':
            spanner = lambda x: np.mean(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
        elif agg_clusterer.affinity == 'l1' or agg_clusterer.affinity == 'manhattan':
            spanner = lambda x: np.mean(np.sum(np.abs(x[:, None, :] - x[None, :, :]), axis=2))
        elif agg_clusterer.affinity == 'l2':
            spanner = lambda x: np.mean(np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)))
        elif agg_clusterer.affinity == 'cosine':
            spanner = lambda x: (np.mean(np.sum((x[:, None, :] * x[None, :, :])) /
                                         (np.sqrt(np.sum(x[:, None, :] * x[:, None, :], axis=2, keepdims=True)) *
                                          np.sqrt(np.sum(x[None, :, :] * x[None, :, :], axis=2, keepdims=True)))))
        else:
            raise AttributeError('Unknown affinity attribute value {0}.'.format(agg_clusterer.affinity))
    else:
        raise AttributeError('Unknown linkage attribute value {0}.'.format(agg_clusterer.linkage))
    return spanner


def convert_agglomerative_clustering_to_newick_tree(clusterer, labels, distance_matrix):
    spanner = get_cluster_spanner(clusterer)
    # leaf_labels is a list of labels for each entry in X
    newick_tree = build_newick_tree(clusterer.children_, len(labels), distance_matrix, labels, spanner)
    return newick_tree
