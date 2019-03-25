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
    # Set up all optional variables to be parsed from the command line (defaults)
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--lowMemoryMode', default=False, action='store_true',
                        help='Whether to use low memory mode or not. If low memory mode is engaged intermediate values '
                             'in the ETMIPC class will be written to file instead of stored in memory. This will reduce'
                             ' the memory footprint but may increase the time to run. Only recommended for very large '
                             'analyses.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1, nargs='?', choices=[1, 2, 3, 4, 5],
                        help='Which output to generate. 1 writes scores for all tested clustering constants, 2 tests '
                             'the clustering Z-score of the predictions and writes them to file as well as plotting ' 
                             'Z-Scores against resiude count, 3 tests the AUROC of contact prediction at different '
                             'levels of sequence separation and plots the resulting curves to file, 4 tests the '
                             'precision of  contact prediction at different levels of sequence separation and list '
                             'lengths (L, L/2 ... L/10), 5 produces heatmaps and surface plots of scores. In all cases '
                             'a file is written out with the final evaluation of the scores, if no PDB is provided, '
                             'this means only times will be recorded.')
    # Clean command line input
    arguments = parser.parse_args()
    arguments = vars(arguments)
    # Scribbed from python dotenv package source code
    frame = sys._getframe()
    frame_filename = frame.f_code.co_filename
    path = os.path.dirname(os.path.abspath(frame_filename))
    # End scrib

    arguments['inputDir'] = os.path.abspath('../Input/23TestGenes/')
    arguments['output'] = os.path.abspath('../Output/Parameterization/')
    arguments['threshold'] = 8.0
    arguments['ignoreAlignmentSize'] = True
    arguments['skipPreAnalyzed'] = True
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


def get_alignment_stats(file_path, query_id, models, cache_dir, jobs):
    curr_aln = SeqAlignment(file_path, query_id)
    curr_aln.import_alignment()
    curr_aln.remove_gaps()
    for model in models:
        jobs[0].append((curr_aln, model, cache_dir))
    jobs[1].append((curr_aln, query_id, cache_dir))
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
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    models = ['identity', 'blosum62', 'custom']
    tree_building = {'random': ('random', {}),
                     'upgma': ('upgma', {}),
                     'custom': ('custom',),
                     'agg_pre_comp': ('agglomerative', {'affinity': 'precomputed', 'linkage': 'complete'}),
                     'agg_pre_avg': ('agglomerative', {'affinity': 'precomputed', 'linkage': 'average'}),
                     'agg_euc_ward': ('agglomerative', {'affinity': 'euclidean', 'linkage': 'ward'}),
                     'agg_euc_comp': ('agglomerative', {'affinity': 'euclidean', 'linkage': 'complete'}),
                     'agg_euc_avg': ('agglomerative', {'affinity': 'euclidean', 'linkage': 'average'})}
    combine_clusters = ['evidence_vs_size', 'evidence_weighted', 'size_weighted', 'average', 'sum']
    combine_branches = ['sum', 'average']
    # Parse arguments
    args = parse_arguments()
    # Read in input files
    input_files = []
    files = os.listdir(args['inputDir'])
    input_files += map(lambda file_name: os.path.join(args['inputDir'], file_name), files)
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
            query_distance_dir = os.path.join(args['output'], 'Distances', input_dict[query]['Query_ID'])
            if not os.path.isdir(query_distance_dir):
                os.makedirs(query_distance_dir)
            aln_stats = get_alignment_stats(f, input_dict[query]['Query_ID'], models, query_distance_dir, jobs)
            input_dict[query]['Seq_Length'] = aln_stats[0]
            input_dict[query]['Aln_Size'] = aln_stats[1]
            input_dict[query]['Max_Depth'] = aln_stats[2] if aln_stats[2] > 10 else 10
        elif f.endswith('pdb'):
            input_dict[query]['Structure'] = f
        elif f.endswith('nhx'):
            input_dict[query]['Custom_Tree'] = f
        else:
            pass
    # Compute distances in parallel
    pool = Pool(args['processes'])
    pool.map_async(compute_single_dist, jobs[0])
    pool.close()
    pool.join()
    # Compute effective alignment sizes
    pool = Pool(args['processes'])
    effective_sizes = pool.map_async(compute_single_effective_aln_size, jobs[1])
    pool.close()
    pool.join()
    for effective_size in effective_sizes.get():
        input_dict[effective_size[0].lower()]['Effective_Aln_Size'] = effective_size[1]
    # Start Parameter Grid Search
    stats = []
    query_order = sorted(input_dict, key=lambda k: input_dict[k]['Aln_Size'])
    for query in query_order:
        print(query)
        query_structure = PDBReference(input_dict[query]['Structure'])
        query_structure.import_pdb(structure_id=input_dict[query]['Query_ID'])
        # Evaluate each of the distance metrics
        for dist in models:
            print(dist)
            dist_dir = os.path.join(args['output'], dist)
            # Evaluate each of the tree building methods
            for tb in tree_building:
                if (dist == 'custom' and tb != 'custom') or (dist != 'custom' and tb == 'custom'):
                    continue
                print(tb)
                method_dir = os.path.join(dist_dir, tb)
                method = tree_building[tb][0]
                if tb == 'custom':
                    method_args = {'tree_path': input_dict[query]['Custom_Tree'], 'tree_name': 'ET-Tree'}
                else:
                    method_args = tree_building[tb][1]

                query_dir = os.path.join(method_dir, input_dict[query]['Query_ID'])
                try:
                    os.makedirs(query_dir)
                except OSError:
                    pass
                print(query_dir)

                predictor = ETMIPC(input_dict[query]['Alignment'])
                time1 = None

                # Evaluate each method for cluster combination
                for cc in combine_clusters:
                    print(cc)
                    cc_dir = os.path.join(query_dir, cc)
                    print(cc_dir)
                    # Evaluate each method for branch combination
                    for cb in combine_branches:
                        print(cb)
                        cb_dir = os.path.join(cc_dir, cb)
                        print(cb_dir)
                        stats_fn = os.path.join(cb_dir, '{}_Stats.csv'.format(input_dict[query]['Query_ID']))
                        if os.path.isfile(stats_fn):
                            query_df = pd.read_csv(stats_fn, sep='\t', header=0, index_col=None)
                            stats.append(query_df)
                            continue
                        dist_fn = '{}.pkl'.format(dist)
                        try:
                            os.symlink(os.path.join(args['output'], 'Distances', input_dict[query]['Query_ID'],
                                                    dist_fn), os.path.join(query_dir, dist_fn))
                        except OSError:
                            pass

                        if dist == 'custom':
                            curr_dist = 'identity'
                        else:
                            curr_dist = dist
                        ################################################################################################
                        if time1 is None:
                            start1 = time()
                            predictor.output_dir = query_dir
                            predictor.processes = args['processes']
                            predictor.low_mem = args['lowMemoryMode']
                            predictor.tree_depth = (2, input_dict[query]['Max_Depth'] + 1)
                            predictor.import_alignment(query=input_dict[query]['Query_ID'],
                                                       ignore_alignment_size=args['ignoreAlignmentSize'],
                                                       clustering=method, clustering_args=method_args, model=curr_dist)
                            predictor.calculate_cluster_scores(evidence=('evidence' in cc), aa_mapping=aa_dict)
                            save_file = os.path.join(predictor.output_dir,
                                                     '{}_cET-MIp.pkl'.format(input_dict[query]['Query_ID']))
                            pickle.dump((predictor.tree_depth, predictor.low_mem, predictor.unique_clusters,
                                         predictor.cluster_mapping, predictor.time), open(save_file, 'wb'),
                                        protocol=pickle.HIGHEST_PROTOCOL)
                            end1 = time()
                            time1 = end1 - start1
                        start2 = time()
                        predictor.cluster_scores = None
                        predictor.branch_scores = None
                        predictor.scores = None
                        predictor.coverage = None
                        predictor.output_dir = cb_dir
                        for k in predictor.tree_depth:
                            curr_dir = os.path.join(cb_dir, str(k))
                            if not os.path.isdir(curr_dir):
                                os.makedirs(curr_dir)
                        predictor.calculate_branch_scores(combine_clusters=cc)
                        predictor.calculate_final_scores(combine_branches=cb)
                        predictor.write_out_scores(curr_date=today)
                        end2 = time()
                        predictor.time['Total'] = time1 + (end2 - start2)
                        curr_time = time1 + (end2 - start2)
                        serialized_path = os.path.join(predictor.output_dir,
                                                       '{}_cET-MIp.npz'.format(input_dict[query]['Query_ID']))
                        np.savez(serialized_path, scores=predictor.scores, coverage=predictor.coverage,
                                 branches=predictor.branch_scores, clusters=predictor.cluster_scores,
                                 nongap_counts=predictor.nongap_counts)
                        ################################################################################################
                        contact_any = ContactScorer(seq_alignment=predictor.alignment, pdb_reference=query_structure,
                                                    cutoff=8.0, query=input_dict[query]['Query_ID'])
                        any_biased_w2_ave = None
                        any_unbiased_w2_ave = None
                        any_score_df, any_coverage_df, any_b_w2_ave, any_u_w2_ave = contact_any.evaluate_predictor(
                            predictor=predictor, verbosity=5, out_dir=cb_dir, dist='Any', today=today,
                            biased_w2_ave=any_biased_w2_ave, unbiased_w2_ave=any_unbiased_w2_ave)
                        if (any_biased_w2_ave is None) and (any_b_w2_ave is not None):
                            any_biased_w2_ave = any_b_w2_ave
                        if (any_unbiased_w2_ave is None) and (any_u_w2_ave is not None):
                            any_unbiased_w2_ave = any_u_w2_ave
                        contact_beta = ContactScorer(seq_alignment=predictor.alignment, pdb_reference=query_structure,
                                                     cutoff=8.0, query=input_dict[query]['Query_ID'])
                        beta_biased_w2_ave = None
                        beta_unbiased_w2_ave = None
                        beta_score_df, beta_coverage_df, beta_b_w2_ave, beta_u_w2_ave = contact_beta.evaluate_predictor(
                            predictor=predictor, verbosity=5, out_dir=cb_dir, dist='CB', today=today,
                            biased_w2_ave=beta_biased_w2_ave, unbiased_w2_ave=beta_unbiased_w2_ave)
                        if (beta_biased_w2_ave is None) and (beta_b_w2_ave is not None):
                            beta_biased_w2_ave = beta_b_w2_ave
                        if (beta_unbiased_w2_ave is None) and (beta_u_w2_ave is not None):
                            beta_unbiased_w2_ave = beta_u_w2_ave
                        query_df = pd.concat([any_score_df, beta_score_df], ignore_index=True)
                        query_df['Query'] = input_dict[query]['Query_ID']
                        query_df['Query_Length'] = input_dict[query]['Seq_Length']
                        query_df['Alignment_Size'] = input_dict[query]['Aln_Size']
                        query_df['Effective_Alignment_Size'] = input_dict[query]['Effective_Aln_Size']
                        query_df['Distance_Model'] = dist
                        query_df['Tree_Building'] = tb
                        query_df['Cluster_Combination'] = cc
                        query_df['Branch_Combination'] = cb
                        query_df['Total_Time'] = curr_time
                        query_df.to_csv(stats_fn, sep='\t', header=True, index=False)
                        stats.append(query_df)
    overall_df = pd.concat(stats, ignore_index=True)
    overall_df.to_csv(os.path.join(args['output'], 'Evaluation_Statistics.csv'), sep='\t', header=True, index=False)

    ####################################################################################################################
    # total_aln_size_df = overall_df.drop(columns=['Effective_Alignment_Size'])
    # total_aln_size_df['Type'] = 'Total'
    # effective_aln_size_df = overall_df.drop(columns=['Alignment_Size'])
    # effective_aln_size_df.rename(columns={'Effective_Alignment_Size': 'Alignment_Size'}, inplace=True)
    # effective_aln_size_df['Type'] = 'Effective'
    # plotting_df = pd.concat([total_aln_size_df, effective_aln_size_df], ignore_index=True)
    # groupby = plotting_df.groupby(['Distance', 'Query'])
    # max_auc = groupby['AUROC'].max()
    # max_auc_ri = max_auc.reset_index(level=0)
    # max_auc_merged = pd.merge(plotting_df, max_auc_ri, how='inner', left_on=['Query', 'Distance', 'AUROC'],
    #                           right_on=['Query', 'Distance', 'AUROC'])
    # sns.lmplot(x='Alignment_Size', y='K', hue='Type', row='Distance', row_order=['Any', 'CB'],
    #            data=max_auc_merged)
    # plt.savefig(os.path.join(arguments['output'], 'Optimal_K_vs_Aln_Size_AUC.eps'))
    # for dist in overall_df['Distance'].unique():
    #     # dist_sub_df = overall_df.loc[overall_df['Distance'] == dist, :]
    #     # query_agg = dist_sub_df.groupby(['Query'])
    #     # max_auc = query_agg['AUC'].max().reset_index(level=0)
    #     # sns.regplot(y='AUC', x='Alignment_Size', )
    #     # sns.regplot(y='AUC', x='Effective_Alignment_Size')
    #     precisions = [x for x in overall_df.columns if 'Precision' in x]
    #     for pre in precisions:
    #         # max_auc = groupby['AUROC'].max()
    #         # max_auc_ri = max_auc.reset_index(level=0)
    #         # max_auc_merged = pd.merge(plotting_df, max_auc_ri, how='inner', left_on=['Query', 'Distance', 'AUROC'],
    #         #                           right_on=['Query', 'Distance', 'AUROC'])
    #         # sns.lmplot(x='Alignment_Size', y='Optimal_K', hue='Type', row='Distance', row_order=['Any', 'CB'],
    #         #            data=max_auc_merged)
    #         # max_precision = query_agg[pre].max()
    #         max_pre = groupby[pre].max()
    #         max_pre_ri = max_pre.reset_index(level=0)
    #         max_pre_merged = pd.merge(plotting_df, max_pre_ri, how='inner', left_on=['Query', 'Distance', pre],
    #                                   right_on=['Query', 'Distance', pre])
    #         sns.lmplot(x='Alignment_Size', y='K', hue='Type', row='Distance', row_order=['Any', 'CB'],
    #                    data=max_pre_merged)
    #         plt.savefig(os.path.join(arguments['output'], 'Optimal_K_vs_Aln_Size_{}.eps'.format(pre)))
    ####################################################################################################################
