"""
Created on Sep 15, 2017

@author: daniel
"""
from SupportingClasses.ContactScorer import ContactScorer
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.ETMIPC import ETMIPC
from PerformAnalysis import analyze_alignment
from Bio.Phylo.TreeConstruction import DistanceCalculator
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool
import cPickle as pickle
from time import time
import numpy as np
import datetime
import argparse
import os
import re
import sys


def parse_arguments():
    """
    parse arguments

    This method provides a nice interface for parsing command line arguments
    and includes help functionality.

    Returns:
        dict. A dictionary containing the arguments parsed from the command line and their arguments.
    """
    # Create input parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    # Set up all variables to be parsed from the command line (no defaults)
    parser.add_argument('--output', metavar='O', type=str, nargs='?',
                        default='./', help='File path to a directory where the results can be generated.')
    parser.add_argument('--input', metavar='I', type=str, nargs='?',
                        default='./', help='File path to a directory where the inputs can be found.')
    # Set up all optional variables to be parsed from the command line (defaults)
    parser.add_argument('--threshold', metavar='T', type=float, nargs='?', default=8.0,
                        help='The distance within the molecular structure at which two residues are considered '
                             'interacting.')
    parser.add_argument('--treeDepth', metavar='K', type=int, nargs='+', default=[2, 3, 5, 7, 10, 25],
                        help='''The levels of the phylogenetic tree to consider when analyzing this alignment, which
                        determines the attributes sequence_assignments and tree_ordering. The following options are
                        available:
                            0 : Entering 0 means all branches from the top of the tree (1) to the leaves (size of the
                            provided alignment) will be analyzed.
                            x, y: If two integers are provided separated by a comma, this will be interpreted as a tuple
                            which will be taken as a range, the top of the tree (1), and all branches between the first
                            and second (non-inclusive) integer will be analyzed.
                            x, y, z, etc.: If one integer (that is not 0) or more than two integers are entered, this
                            will be interpreted as a list. All branches in the list will be analyzed, as well as the top
                            of the tree (1) even if not listed.''')
    parser.add_argument('--combineBranches', metavar='C', type=str, default='sum', choices=['sum', 'average'],
                        nargs='?', help='The method to use when combining across the specified clustering constants.')
    parser.add_argument('--combineClusters', metavar='c', type=str, nargs='?', default='sum',
                        choices=['sum', 'average', 'size_weighted', 'evidence_weighted', 'evidence_vs_size'],
                        help='How information should be integrated across clusters resulting from the same clustering '
                             'constant.')
    parser.add_argument('--ignoreAlignmentSize', default=False, action='store_true',
                        help='Whether or not to allow alignments with fewer than 125 sequences as suggested by '
                             'PMID:16159918.')
    parser.add_argument('--lowMemoryMode', default=False, action='store_true',
                        help='Whether to use low memory mode or not. If low memory mode is engaged intermediate values '
                             'in the ETMIPC class will be written to file instead of stored in memory. This will reduce'
                             ' the memory footprint but may increase the time to run. Only recommended for very large '
                             'analyses.')
    parser.add_argument('--removeIntermediates', default=False, action='store_true',
                        help='Whether to remove the intermediate files generated if the lowMemoryMode option was set.')
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1, nargs='?', choices=[1, 2, 3, 4, 5],
                        help='Which output to generate. 1 writes scores for all tested clustering constants, 2 tests '
                             'the clustering Z-score of the predictions and writes them to file as well as plotting ' 
                             'Z-Scores against residue count, 3 tests the AUROC of contact prediction at different '
                             'levels of sequence separation and plots the resulting curves to file, 4 tests the '
                             'precision of  contact prediction at different levels of sequence separation and list '
                             'lengths (L, L/2 ... L/10), 5 produces heatmaps and surface plots of scores. In all cases '
                             'a file is written out with the final evaluation of the scores, if no PDB is provided, '
                             'this means only times will be recorded.')
    # Clean command line input
    arguments = parser.parse_args()
    arguments = vars(arguments)
    arguments['treeDepth'] = sorted(arguments['treeDepth'])
    if len(arguments['treeDepth']) == 1 and arguments['treeDepth'][0] == 0:
        arguments['treeDepth'] = None
    elif len(arguments['treeDepth']) == 2:
        arguments['treeDepth'] = tuple(arguments['treeDepth'])
    processor_count = cpu_count()
    if arguments['processes'] > processor_count:
        arguments['processes'] = processor_count
    return arguments


def parse_id(fa_file):
    for line in open(fa_file, 'rb'):
        id_check = re.match(r'^>query_(.*)\s?$', line)
        if id_check:
            return id_check.group(1)
        else:
            continue


# def get_alignment_stats(file_path, query_id, models, cache_dir, jobs):
def get_alignment_stats(file_path, query_id):
    curr_aln = SeqAlignment(file_path, query_id)
    curr_aln.import_alignment()
    curr_aln.remove_gaps()
    # for model in models:
    #     jobs[0].append((curr_aln, model, cache_dir))
    # jobs[1].append((curr_aln, query_id, cache_dir))
    return curr_aln.seq_length, curr_aln.size, int(np.ceil(0.1 * curr_aln.size))


def compute_single_dist(in_tup):
    # print('Performing {} - {} distance computation'.format(in_tup[0].query_id, in_tup[1]))
    in_tup[0].compute_distance_matrix(model=in_tup[1], save_dir=in_tup[2])
    # print('Completing {} - {} distance computation'.format(in_tup[0].query_id, in_tup[1]))


def compute_single_effective_aln_size(in_tup):
    eff_size = in_tup[0].compute_effective_alignment_size(save_dir=in_tup[2])
    return (in_tup[1], eff_size)


if __name__ == '__main__':
    # Set up requirements for experiments
    today = str(datetime.date.today())
    models = ['identity', 'custom'] # 'blosum62'
    # tree_building = {'random': ('random', {}),
    #                  'upgma': ('upgma', {}),
    #                  'custom': ('custom',),
    #                  'agg_pre_comp': ('agglomerative', {'affinity': 'precomputed', 'linkage': 'complete'}),
    #                  'agg_pre_avg': ('agglomerative', {'affinity': 'precomputed', 'linkage': 'average'}),
    #                  'agg_euc_ward': ('agglomerative', {'affinity': 'euclidean', 'linkage': 'ward'}),
    #                  'agg_euc_comp': ('agglomerative', {'affinity': 'euclidean', 'linkage': 'complete'}),
    #                  'agg_euc_avg': ('agglomerative', {'affinity': 'euclidean', 'linkage': 'average'})}
    tree_building = {'upgma': ('upgma', {}),
                     'custom': ('custom',),
                     'agg_euc_ward': ('agglomerative', {'affinity': 'euclidean', 'linkage': 'ward'})}
    # combine_clusters = ['sum'] # ['evidence_vs_size', 'evidence_weighted', 'size_weighted', 'average']
    # combine_branches = ['sum'] # ['average']
    # Parse arguments
    args = parse_arguments()
    # Read in input files
    input_files = []
    files = os.listdir(args['input'])
    input_files += map(lambda file_name: os.path.join(args['input'], file_name), files)
    input_files.sort(key=lambda f: os.path.splitext(f)[1])
    input_dict = {}
    jobs = ([], [])
    for f in input_files:
        check = re.search(r'(\d[\d|A-Z|a-z]{3}[A-Z]?)', f.split('/')[-1])
        if not check:
            continue
        query = check.group(1).lower()
        print(query)
        if query not in input_dict:
            input_dict[query] = {}
        if f.endswith('fa'):
            input_dict[query]['Query_ID'] = parse_id(f)
            input_dict[query]['Alignment'] = f
            # query_distance_dir = os.path.join(args['output'], 'Distances', input_dict[query]['Query_ID'])
            # if not os.path.isdir(query_distance_dir):
            #     os.makedirs(query_distance_dir)
            # aln_stats = get_alignment_stats(f, input_dict[query]['Query_ID'], models, query_distance_dir, jobs)
            aln_stats = get_alignment_stats(f, input_dict[query]['Query_ID'])
            input_dict[query]['Seq_Length'] = aln_stats[0]
            input_dict[query]['Aln_Size'] = aln_stats[1]
            input_dict[query]['Max_Depth'] = aln_stats[2] if aln_stats[2] > 10 else 10
        elif f.endswith('pdb'):
            input_dict[query]['Structure'] = f
        elif f.endswith('nhx'):
            input_dict[query]['Custom_Tree'] = f
        else:
            pass
    # Start Parameter Grid Search
    stats = []
    query_order = sorted(input_dict, key=lambda k: input_dict[k]['Aln_Size'])
    for query in query_order:
        print(query)
        # Evaluate each of the distance metrics
        for dist in models:
            print(dist)
            dist_dir = os.path.join(args['output'], dist)
            # Evaluate each of the tree building methods
            for tb in tree_building:
                if ((dist == 'custom' and tb not in ['custom', 'random']) or
                        (dist != 'custom' and tb in ['custom', 'random'])):
                    continue
                print(tb)
                method_dir = os.path.join(dist_dir, tb)
                method = tree_building[tb][0]
                if tb == 'custom':
                    method_args = {'tree_path': input_dict[query]['Custom_Tree'], 'tree_name': 'ET-Tree'}
                else:
                    method_args = tree_building[tb][1]
                if dist == 'custom':
                    curr_dist = 'identity'
                else:
                    curr_dist = dist
                curr_args = {'alignment': [input_dict[query]['Alignment']], 'pdb': input_dict[query]['Structure'],
                             'query': [input_dict[query]['Query_ID']], 'distanceModel': curr_dist,
                             'treeConstruction': method, 'treeConstructionArgs': method_args}
                curr_args.update(args)
                if not os.path.isdir(method_dir):
                    os.makedirs(method_dir)
                curr_args['output'] = method_dir
                analyze_alignment(args=curr_args)
                df_fn = 'Score_Evaluation_Dist-{}.txt'
                query_df_any = pd.read_csv(os.path.join(method_dir, input_dict[query]['Query_ID'], df_fn.format('Any')),
                                           header=0, index_col=False, sep='\t')
                query_df_cb = pd.read_csv(os.path.join(method_dir, input_dict[query]['Query_ID'], df_fn.format('CB')),
                                          header=0, index_col=False, sep='\t')
                stats += [query_df_any, query_df_cb]
    overall_df = pd.concat(stats, ignore_index=True)
    overall_df.to_csv(os.path.join(args['output'], 'Evaluation_Statistics.csv'), sep='\t', header=True, index=False)
