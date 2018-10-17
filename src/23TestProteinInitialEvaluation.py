"""
Created on Sep 15, 2017

@author: daniel
"""
from SupportingClasses.ContactScorer import ContactScorer
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.ETMIPWrapper import ETMIPWrapper
from SupportingClasses.DCAWrapper import DCAWrapper
from SupportingClasses.ETMIPC import ETMIPC
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import argparse
import sys
import os
import re


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
    ####################################################################################################################
    ####################################################################################################################
    methods = {'DCA': {'class': DCAWrapper, 'args': {'delete_file': False}},
               'ET-MIp': {'class': ETMIPWrapper, 'args': {'delete_files': False}},
               'cET-MIp': {'class': ETMIPC, 'args': {'today': today, 'query': None, 'clusters': [2, 3, 5, 7, 10, 25],
                                                     'aa_dict': aa_dict, 'combine_clusters': 'sum', 'combine_ks': 'sum',
                                                     'processes': arguments['processes'],
                                                     'low_memory_mode': arguments['lowMemoryMode'],
                                                     'ignore_alignment_size': True}}}
    if not os.path.isdir(arguments['output']):
        os.mkdir(arguments['output'])
    ####################################################################################################################
    ####################################################################################################################
    times = {'Query': [], 'Method': [], 'Time(s)': []}
    stats = []
    for query in sorted(input_dict.keys()):
        # if query != '7hvpa':
        #     print(query)
        #     continue
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
        for method in sorted(methods.keys()):
            if 'dir' not in methods[method]:
                method_dir = os.path.join(arguments['output'], method)
                methods[method]['dir'] = method_dir
                if not os.path.isdir(method_dir):
                    os.mkdir(method_dir)
            else:
                method_dir = methods[method]['dir']
            protein_dir = os.path.join(method_dir, query)
            if not os.path.isdir(protein_dir):
                os.mkdir(protein_dir)
            predictor = methods[method]['class'](query_aln)
            print(protein_dir, '{}_predictions.tsv'.format(method))
            if method == 'cET-MIp':
                methods[method]['args']['query'] = input_dict[query][0]
            curr_time = predictor.calculate_scores(out_dir=protein_dir, **methods[method]['args'])
            times['Query'].append(query)
            times['Method'].append(method)
            times['Time(s)'].append(curr_time)
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
            method_df = pd.concat([any_score_df, beta_score_df], ignore_index=True)
            if 'K' in method_df.columns:
                method_df['Method'] = method_df['K'].apply(lambda k: '{}_{}'.format(method, k))
                method_df.drop(columns='K', inplace=True)
            else:
                method_df['Method'] = method
            method_df['Query'] = input_dict[query][0]
            method_df['Query_Length'] = query_aln.seq_length
            method_df['Alignment_Length'] = query_aln.size
            stats.append(method_df)
    ####################################################################################################################
    method_order = ['DCA', 'ET-MIp', 'cET-MIp_2', 'cET-MIp_3', 'cET-MIp_5', 'cET-MIp_7', 'cET-MIp_10', 'cET-MIp_25']
    separation_order = ['Any', 'Neighbors', 'Short', 'Medium', 'Long']
    query_order = ['3q05A', '2b59A', '7hvpA', '1c17A', '206lA', '1bolA', '2z0eA', '1axbA', '135lA', '2rh1A', '4lliA',
                   '1a26A', '1c0kA', '2zxeA', '1jwlA', '1hckA', '1h1vA', '2ysdA', '2iopA', '3b6vA', '4ycuA', '2werA',
                   '3tnuA']
    overall_df = pd.concat(stats, ignore_index=True)
    query_order = sorted(overall_df['Alignment_Length'].unique())
    from IPython import embed
    embed()
    overall_df.to_csv(os.path.join(arguments['output'], 'Evaluation_Statistics.csv'), sep='\t', header=True,
                      index=False)
    auc_any = sns.catplot(data=overall_df.loc[(overall_df['Distance'] == 'Any'), :], x='Sequence_Separation', y='AUROC',
                          hue='Method', col='Query', order=separation_order, hue_order=method_order,
                          col_order=query_order, col_wrap=4, kind='bar', legend_out=True, sharex=True, sharey=True)
    auc_any.savefig(os.path.join(arguments['output'], 'AUROC_Method_Comparison_DistAny.tiff'))
    auc_cb = sns.catplot(data=overall_df.loc[(overall_df['Distance'] == 'CB'), :], x='Sequence_Separation', y='AUROC',
                          hue='Method', col='Query', order=separation_order, hue_order=method_order,
                          col_order=query_order, col_wrap=4, kind='bar', legend_out=True, sharex=True, sharey=True)
    auc_cb.savefig(os.path.join(arguments['output'], 'AUROC_Method_Comparison_DistCB.tiff'))
    times_df = pd.DataFrame(times)
    times_df.to_csv(os.path.join(arguments['output'], 'Evaluation_Timing.csv'), sep='\t', header=True, index=False)
    sns.barplot(x='Query', y='Time(s)', hue='Method', order=query_order, hue_order=method_order)
    plt.savefig(os.path.join(arguments['output'], 'Time_Comparison_DistAny.tiff'))
    plt.clf()
    embed()
    exit()
