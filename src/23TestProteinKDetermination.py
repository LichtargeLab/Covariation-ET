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
    parser.add_argument('--lowMemoryMode', metavar='l', type=bool, nargs='?',
                        default=False, help='Whether to use low memory mode or not. If low memory mode is engaged '
                                            'intermediate values in the ETMIPC class will be written to file instead of'
                                            ' stored in memory. This will reduce the memory footprint but may increase '
                                            'the time to run. Only recommended for very large analyses.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1,
                        nargs='?', choices=[1, 2, 3, 4], help='How many figures to produce.\n1 = ROC Curves, ETMIP '
                                                              'Coverage file, and final AUC and Timing file\n2 = files '
                                                              'with all scores at each clustering\n3 = sub-alignment '
                                                              'files and plots\n4 = surface plots and heatmaps of ETMIP'
                                                              ' raw and coverage scores.')
    # Clean command line input
    args = parser.parse_args()
    args = vars(args)
    # Scribbed from python dotenv package source code
    frame = sys._getframe()
    frame_filename = frame.f_code.co_filename
    path = os.path.dirname(os.path.abspath(frame_filename))
    # End scrib

    args['inputDir'] = os.path.abspath(os.path.join(path, '..', 'Input/23TestGenes/'))
    args['output'] = None
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


def get_alignment_stats(file_path, query_id):
    print(file_path)
    print(query_id)
    curr_aln = SeqAlignment(file_path, query_id)
    curr_aln.remove_gaps()
    curr_aln.importAlignment()
    effective_alignment_size = curr_aln.compute_effective_alignment_size()
    return curr_aln.seqLength, curr_aln.size, effective_alignment_size, int(np.floor(curr_aln.size / 3.0))

if __name__ == '__main__':
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
        check = re.search(r'(\d[\d|A-Z|a-z]{3}[A-Z]?)', f.split('/')[-1])
        if not check:
            continue
        query = check.group(1).lower()
        if query not in input_dict:
            input_dict[query] = [None, None, None, None, None, None, None]
        if f.endswith('fa'):
            input_dict[query][0] = parse_id(f)
            input_dict[query][1] = f
            aln_stats = get_alignment_stats(f, input_dict[query][0])
            input_dict[query][3] = aln_stats[0]
            input_dict[query][4] = aln_stats[1]
            input_dict[query][5] = aln_stats[2]
            input_dict[query][6] = aln_stats[3]
        elif f.endswith('pdb'):
            input_dict[query][2] = f
        else:
            pass
    ####################################################################################################################
    ####################################################################################################################
    if not os.path.isdir(arguments['output']):
        os.mkdir(arguments['output'])
    ####################################################################################################################
    ####################################################################################################################
    stats = []
    query_order = sorted(input_dict, key=lambda k: input_dict[k][4])
    for query in query_order:
        query_aln = SeqAlignment(input_dict[query][1], input_dict[query][0])
        query_aln.import_alignment()
        query_aln.remove_gaps()
        new_aln_fn = os.path.join(arguments['output'], '{}_ungapped.fa'.format(input_dict[query][0]))
        query_aln.write_out_alignment(new_aln_fn)
        query_aln.file_name = new_aln_fn
        query_structure = PDBReference(input_dict[query][2])
        query_structure.import_pdb(structure_id=input_dict[query][0])
        contact_any = ContactScorer(seq_alignment=query_aln, pdb_reference=query_structure, cutoff=8.0)
        any_biased_w2_ave = None
        any_unbiased_w2_ave = None
        contact_beta = ContactScorer(seq_alignment=query_aln, pdb_reference=query_structure, cutoff=8.0)
        beta_biased_w2_ave = None
        beta_unbiased_w2_ave = None
        protein_dir = os.path.join(arguments['output'], query)
        if not os.path.isdir(protein_dir):
            os.mkdir(protein_dir)
        predictor = ETMIPC(query_aln)
        curr_time = predictor.calculate_scores(out_dir=protein_dir, today=today, query=input_dict[query][0],
                                               clusters=range(2, input_dict[query][5] + 1), aa_dict=aa_dict,
                                               combine_clusters=arguments['combineClusters'],
                                               combine_ks=arguments['combineKs'], processes=arguments['processes'],
                                               low_memory_mode=arguments['lowMemoryMode'],
                                               ignore_alignment_size=arguments['ignoreAlignmentSize'],
                                               del_intermediate=False)
        any_score_df, any_coverage_df, any_b_w2_ave, any_u_w2_ave = contact_any.evaluate_predictor(
            query=input_dict[query][0], predictor=predictor, verbosity=4, out_dir=protein_dir, dist='Any',
            biased_w2_ave=any_biased_w2_ave, unbiased_w2_ave=any_unbiased_w2_ave)
        if (any_biased_w2_ave is None) and (any_b_w2_ave is not None):
            any_biased_w2_ave = any_b_w2_ave
        if (any_unbiased_w2_ave is None) and (any_u_w2_ave is not None):
            any_unbiased_w2_ave = any_u_w2_ave
        beta_score_df, beta_coverage_df, beta_b_w2_ave, beta_u_w2_ave = contact_beta.evaluate_predictor(
            query=input_dict[query][0], predictor=predictor, verbosity=4, out_dir=protein_dir, dist='CB',
            biased_w2_ave=beta_biased_w2_ave, unbiased_w2_ave=beta_unbiased_w2_ave)
        if (beta_biased_w2_ave is None) and (beta_b_w2_ave is not None):
            beta_biased_w2_ave = beta_b_w2_ave
        if (beta_unbiased_w2_ave is None) and (beta_u_w2_ave is not None):
            beta_unbiased_w2_ave = beta_u_w2_ave
        query_df = pd.concat([any_score_df, beta_score_df], ignore_index=True)
        query_df['Query'] = input_dict[query][0]
        query_df['Query_Length'] = input_dict[query][3]
        query_df['Alignment_Size'] = input_dict[query][4]
        query_df['Effective_Alignment_Size'] = input_dict[query][5]
        stats.append(query_df)
    ####################################################################################################################
    separation_order = ['Any', 'Neighbors', 'Short', 'Medium', 'Long']
    overall_df = pd.concat(stats, ignore_index=True)
    from IPython import embed
    embed()
    overall_df.to_csv(os.path.join(arguments['output'], 'Evaluation_Statistics.csv'), sep='\t', header=True,
                      index=False)
    ####################################################################################################################
    total_aln_size_df = overall_df.drop(columns=['Effective_Alignment_Size'])
    total_aln_size_df['Type'] = 'Total'
    effective_aln_size_df = overall_df.drop(columns=['Alignment_Size'])
    effective_aln_size_df.rename(columns={'Effective_Alignment_Size': 'Alignment_Size'}, inplace=True)
    effective_aln_size_df['Type'] = 'Effective'
    plotting_df = pd.concat([total_aln_size_df, effective_aln_size_df], ignore_index=True)
    groupby = plotting_df.groupby(['Distance', 'Query'])
    max_auc = groupby['AUROC'].max()
    max_auc_ri = max_auc.reset_index(level=0)
    max_auc_merged = pd.merge(plotting_df, max_auc_ri, how='inner', left_on=['Query', 'Distance', 'AUROC'],
                              right_on=['Query', 'Distance', 'AUROC'])
    sns.lmplot(x='Alignment_Size', y='K', hue='Type', row='Distance', row_order=['Any', 'CB'],
               data=max_auc_merged)
    plt.savefig(os.path.join(arguments['output'], 'Optimal_K_vs_Aln_Size_AUC.eps'))
    for dist in overall_df['Distance'].unique():
        # dist_sub_df = overall_df.loc[overall_df['Distance'] == dist, :]
        # query_agg = dist_sub_df.groupby(['Query'])
        # max_auc = query_agg['AUC'].max().reset_index(level=0)
        # sns.regplot(y='AUC', x='Alignment_Size', )
        # sns.regplot(y='AUC', x='Effective_Alignment_Size')
        precisions = [x for x in overall_df.columns if 'Precision' in x]
        for pre in precisions:
            # max_auc = groupby['AUROC'].max()
            # max_auc_ri = max_auc.reset_index(level=0)
            # max_auc_merged = pd.merge(plotting_df, max_auc_ri, how='inner', left_on=['Query', 'Distance', 'AUROC'],
            #                           right_on=['Query', 'Distance', 'AUROC'])
            # sns.lmplot(x='Alignment_Size', y='Optimal_K', hue='Type', row='Distance', row_order=['Any', 'CB'],
            #            data=max_auc_merged)
            # max_precision = query_agg[pre].max()
            max_pre = groupby[pre].max()
            max_pre_ri = max_pre.reset_index(level=0)
            max_pre_merged = pd.merge(plotting_df, max_pre_ri, how='inner', left_on=['Query', 'Distance', pre],
                                      right_on=['Query', 'Distance', pre])
            sns.lmplot(x='Alignment_Size', y='K', hue='Type', row='Distance', row_order=['Any', 'CB'],
                       data=max_pre_merged)
            plt.savefig(os.path.join(arguments['output'], 'Optimal_K_vs_Aln_Size_{}.eps'.format(pre)))
    ####################################################################################################################
    embed()
    exit()
