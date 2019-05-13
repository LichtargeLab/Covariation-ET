"""
Created on Sep 15, 2017

@author: daniel
"""
from SupportingClasses.ContactScorer import ContactScorer
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.ETMIPC import ETMIPC
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pandas as pd
import datetime
import argparse
import sys
import os
import re
from IPython import embed


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
    # Set up all optional variables to be parsed from the command line (defaults)
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--lowMemoryMode', metavar='l', type=bool, nargs='?',
                        default=False, help='Whether to use low memory mode or not. If low memory mode is engaged '
                                            'intermediate values in the ETMIPC class will be written to file instead of'
                                            ' stored in memory. This will reduce the memory footprint but may increase '
                                            'the time to run. Only recommended for very large analyses.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1, nargs='?', choices=[1, 2, 3, 4, 5],
                        help='What level of output to produce: 1. writes scores for all tested clustering constants 2. '
                             'tests the clustering Z-score of the predictions and writes them to file as well as '
                             'plotting Z-Scores against resiude count 3. tests the AUROC of contact prediction at '
                             'different levels of sequence separation and plots the resulting curves to file 4. tests '
                             'the precision of  contact prediction at different levels of sequence separation and list '
                             'lengths (L, L/2 ... L/10). 5. produces heatmaps and surface plots of scores. In all '
                             'cases a file is written out with the final evaluation of the scores, if no PDB is '
                             'provided, this means only times will be recorded.')
    # Clean command line input
    args = parser.parse_args()
    args = vars(args)
    # Scribbed from python dotenv package source code
    frame = sys._getframe()
    frame_filename = frame.f_code.co_filename
    path = os.path.dirname(os.path.abspath(frame_filename))
    # End scrib

    args['inputDir'] = os.path.abspath(os.path.join(path, '..', 'Input/23TestGenes/'))
    args['output'] = os.path.abspath(os.path.join(path, '..', 'Output/23TestProteinInitialEvaluation/'))
    args['clusters'] = [2, 3, 5, 7, 10, 25]
    args['threshold'] = 8.0
    args['combineKs'] = 'sum'
    args['combineClusters'] = 'sum'
    args['ignoreAlignmentSize'] = True
    args['skipPreAnalyzed'] = True
    processor_count = cpu_count()
    if args['processes'] > processor_count:
        args['processes'] = processor_count
    return args


def parse_id(fa_file):
    for line in open(fa_file, 'rb'):
        id_check = re.match(r'^>query_(.*)\s?$', line)
        if id_check:
            return id_check.group(1)
        else:
            continue


if __name__ == '__main__':
    start_load = time()
    today = str(datetime.date.today())
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    arguments = parse_arguments()
    input_files = []
    files = os.listdir(arguments['inputDir'])
    input_files += map(lambda file_name: os.path.abspath(arguments['inputDir']) + '/' + file_name, files)
    input_files.sort(key=lambda f: os.path.splitext(f)[1])
    input_dict = {}
    for f in input_files:
        print f
        check = re.search(r'(\d[\d|A-Za-z]{3}[A-Z]?)', f.split('/')[-1])
        if not check:
            continue
        query = check.group(1).lower()
        if query not in input_dict:
            input_dict[query] = [None, None, None]
        if f.endswith('fa'):
            input_dict[query][0] = parse_id(f)
            input_dict[query][1] = f
        elif f.endswith('pdb'):
            input_dict[query][2] = f
        else:
            pass
    end_load = time()
    print('Finding input data took: {} sec'.format(end_load - start_load))
    start_eval = time()
    eval_stats_filename = os.path.join(arguments['output'], 'Evaluation_Statistics.csv')
    eval_times_filename = os.path.join(arguments['output'], 'Evaluation_Timing.csv')
    method_arguments = {'today': today, 'query': None, 'clusters': [2, 3, 5, 7, 10, 25], 'aa_dict': aa_dict,
                        'combine_clusters': 'sum', 'combine_ks': 'sum', 'processes': arguments['processes'],
                        'low_memory_mode': arguments['lowMemoryMode'], 'ignore_alignment_size': True}
    if not os.path.isdir(arguments['output']):
        os.mkdir(arguments['output'])
    for query in sorted(input_dict.keys()):
        if query != '1h1va':
        # if query != '3q05a':
        # if query != '7hvpa':
        # if query != '3tnua':
        # if query != '2zxea':
        #     print(query)
        # embed()
            continue
        query_aln = SeqAlignment(input_dict[query][1], input_dict[query][0])
        query_aln.import_alignment()
        query_aln.remove_gaps()
        query_aln.compute_distance_matrix()
        query_aln.set_tree_ordering()
        # print(query_aln.tree_order)
        # print(query_aln.sequence_assignemnts)
        from Bio import AlignIO
        out_dir = '/home/daniel/Desktop/Test/'
        positional_alns = {c: {k: {} for k in range(c)} for c in [1, 2, 3, 5, 7, 10, 25]}
        scoring = {'Branch': [], 'Cluster': [], 'Cluster Score': [], 'Branch Score': [], 'Combined Score': []}
        i = 93 - 1
        # i = 7 - 1
        # i = 226 - 1
        j = 94 - 1
        # j = 62 - 1
        # j = 368 - 1
        for k in [1, 2, 3, 5, 7, 10, 25]:
            for c in range(k):
                curr_aln = query_aln.get_branch_cluster(k, c)
                positional_alns[k][c]['aln'] = curr_aln.generate_positional_sub_alignment(i, j)
                AlignIO.write([positional_alns[k][c]['aln'].alignment], os.path.join(out_dir, '{}_C{}_K{}.aln'.format(
                    curr_aln.query_id.split('_')[1], k, c)), 'clustal')
                positional_alns[1][0]['plot'] = positional_alns[k][c]['aln'].heatmap_plot('{}_C={} K={}'.format(
                    curr_aln.query_id.split('_')[1], k, c), aa_dict=aa_dict, save=True, out_dir=out_dir)
        # method_dir = os.path.join(arguments['output'], 'cET-MIp')
        # if not os.path.isdir(method_dir):
        #     os.mkdir(method_dir)
        # protein_dir = os.path.join(method_dir, query)
        # if not os.path.isdir(protein_dir):
        #     os.mkdir(protein_dir)
        # predictor = ETMIPC(query_aln)
        # print(protein_dir, 'cET-MIp_predictions.tsv')
        # method_arguments['query'] = input_dict[query][0]
        # curr_time = predictor.calculate_scores(out_dir=protein_dir, **method_arguments)
        # ####################################################################################################
        # predictor.explore_positions(1, 2, aa_dict)
        # exit()
        # ####################################################################################################
