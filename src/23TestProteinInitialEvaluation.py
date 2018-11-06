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


def parse_time_strings(str_in):
    if not isinstance(str_in, str):
        return str_in
    try:
        recorded_time = float(str_in)
        return recorded_time
    except ValueError:
        time_dict = {}
        str_in = str_in.lstrip('{').rstrip('}')
        for pair in str_in.split(', '):
            cat, recorded_time = pair.split(': ')
            try:
                time_dict[int(cat)] = float(recorded_time)
            except ValueError:
                time_dict[cat.strip("'")] = float(recorded_time)
        return time_dict


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
    if os.path.isfile(eval_stats_filename) and os.path.isfile(eval_times_filename):
        overall_df = pd.read_csv(eval_stats_filename, sep='\t', header=0, index_col=False)
        times_df = pd.read_csv(eval_times_filename, sep='\t', header=0, index_col=False)
    else:
        methods = {'DCA': {'class': DCAWrapper, 'args': {'delete_file': False}},
                   'ET-MIp': {'class': ETMIPWrapper, 'args': {'delete_files': False}},
                   'cET-MIp': {'class': ETMIPC, 'args': {'today': today, 'query': None, 'clusters': [2, 3, 5, 7, 10, 25],
                                                         'aa_dict': aa_dict, 'combine_clusters': 'sum', 'combine_ks': 'sum',
                                                         'processes': arguments['processes'],
                                                         'low_memory_mode': arguments['lowMemoryMode'],
                                                         'ignore_alignment_size': True}}}
        if not os.path.isdir(arguments['output']):
            os.mkdir(arguments['output'])
        times = {'Query': [], 'Method': [], 'Time(s)': []}
        stats = []
        for query in sorted(input_dict.keys()):
            # if query != '7hvpa':
            # if query != '3tnua':
            # if query != '2zxea':
            #     print(query)
                # embed()
                # continue
            query_aln = SeqAlignment(input_dict[query][1], input_dict[query][0])
            query_aln.import_alignment()
            query_aln.remove_gaps()
            new_aln_fn = os.path.join(arguments['output'], '{}_ungapped.fa'.format(input_dict[query][0]))
            query_aln.write_out_alignment(new_aln_fn)
            query_aln.file_name = new_aln_fn
            query_structure = PDBReference(input_dict[query][2])
            query_structure.import_pdb(structure_id=input_dict[query][0])
            contact_scorer = ContactScorer(seq_alignment=query_aln, pdb_reference=query_structure, cutoff=8.0)
            biased_w2_ave = None
            unbiased_w2_ave = None
            for dist in ['Any', 'CB']:
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
                    times['Query'].append(input_dict[query][0])
                    times['Method'].append(method)
                    times['Time(s)'].append(curr_time)
                    # any_score_df, any_coverage_df, any_b_w2_ave, any_u_w2_ave = contact_any.evaluate_predictor(
                    score_df, coverage_df, b_w2_ave, u_w2_ave = contact_scorer.evaluate_predictor(
                        # query=input_dict[query][0], predictor=predictor, verbosity=4, out_dir=protein_dir, dist='Any',
                        query=input_dict[query][0], predictor=predictor, verbosity=4, out_dir=protein_dir, dist=dist,
                        biased_w2_ave=biased_w2_ave, unbiased_w2_ave=unbiased_w2_ave)
                    if (biased_w2_ave is None) and (b_w2_ave is not None):
                        biased_w2_ave = b_w2_ave
                    if (unbiased_w2_ave is None) and (u_w2_ave is not None):
                        unbiased_w2_ave = u_w2_ave
                    if 'K' in score_df.columns:
                        score_df['Method'] = score_df['K'].apply(lambda k: '{}_{}'.format(method, k))
                        score_df.drop(columns='K', inplace=True)
                    else:
                        score_df['Method'] = method
                    score_df['Query'] = input_dict[query][0]
                    score_df['Query_Length'] = query_aln.seq_length
                    score_df['Alignment_Length'] = query_aln.size
                    stats.append(score_df)
        overall_df = pd.concat(stats, ignore_index=True)
        overall_df.to_csv(eval_stats_filename, sep='\t', header=True, index=False)
        times_df = pd.DataFrame(times)
        times_df.to_csv(eval_times_filename, sep='\t', header=True, index=False)
    times_df['Time(s)'] = times_df['Time(s)'].apply(parse_time_strings)
    times_df['Total Time(s)'] = times_df['Time(s)'].apply(lambda x: x['Total'] if isinstance(x, dict) else x)
    end_eval = time()
    print('Evaluating methods for test set data took: {} sec'.format(end_eval - start_eval))
    ####################################################################################################################
    #
    method_order = ['DCA', 'ET-MIp', 'cET-MIp_2', 'cET-MIp_3', 'cET-MIp_5', 'cET-MIp_7', 'cET-MIp_10', 'cET-MIp_25']
    separation_order = ['Any', 'Neighbors', 'Short', 'Medium', 'Long']
    query_order = list(zip(*sorted(zip(overall_df['Query'].unique(), overall_df['Alignment_Length'].unique()),
                                   key=lambda k: k[1]))[0])
    auc_any_path = os.path.join(arguments['output'], 'AUROC_Method_Comparison_DistAny.eps')
    auc_any_point_path = os.path.join(arguments['output'], 'AUROC_Method_Comparison_DistAny_PointSummary.eps')
    auc_any_box_path = os.path.join(arguments['output'], 'AUROC_Method_Comparison_DistAny_BoxSummary.eps')
    if (not os.path.isfile(auc_any_path)) or (not os.path.isfile(auc_any_point_path)) or \
            (not os.path.isfile(auc_any_box_path)):
        auc_any = sns.catplot(data=overall_df.loc[(overall_df['Distance'] == 'Any'), :], x='Sequence_Separation',
                              y='AUROC', hue='Method', col='Query', order=separation_order, hue_order=method_order,
                              col_order=query_order, col_wrap=4, kind='bar', legend_out=True, sharex=True, sharey=True)
        auc_any.savefig(auc_any_path)
        plt.close()
        auc_any = sns.catplot(data=overall_df.loc[(overall_df['Distance'] == 'Any'), :], x='Sequence_Separation',
                              y='AUROC', hue='Method', order=separation_order, hue_order=method_order, kind='point',
                              legend_out=True, sharex=True, sharey=True, ci=None)
        auc_any.savefig(auc_any_point_path)
        plt.close()
        auc_any = sns.catplot(data=overall_df.loc[(overall_df['Distance'] == 'Any'), :], x='Sequence_Separation',
                              y='AUROC', hue='Method', order=separation_order, hue_order=method_order, kind='box',
                              legend_out=True, sharex=True, sharey=True)
        auc_any.savefig(auc_any_box_path)
        plt.close()
    auc_cb_path = os.path.join(arguments['output'], 'AUROC_Method_Comparison_DistCB.eps')
    auc_cb_point_path = os.path.join(arguments['output'], 'AUROC_Method_Comparison_DistCB_PointSummary.eps')
    auc_cb_box_path = os.path.join(arguments['output'], 'AUROC_Method_Comparison_DistCB_BoxSummary.eps')
    if (not os.path.isfile(auc_cb_path)) or (not os.path.isfile(auc_cb_point_path)) or \
            (not os.path.isfile(auc_cb_box_path)):
        auc_cb = sns.catplot(data=overall_df.loc[(overall_df['Distance'] == 'CB'), :], x='Sequence_Separation',
                             y='AUROC', hue='Method', col='Query', order=separation_order, hue_order=method_order,
                             col_order=query_order, col_wrap=4, kind='bar', legend_out=True, sharex=True, sharey=True)
        auc_cb.savefig(auc_cb_path)
        plt.close()
        auc_cb = sns.catplot(data=overall_df.loc[(overall_df['Distance'] == 'CB'), :], x='Sequence_Separation',
                             y='AUROC', hue='Method', order=separation_order, hue_order=method_order, kind='point',
                             legend_out=True, sharex=True, sharey=True, ci=None)
        auc_cb.savefig(auc_cb_point_path)
        plt.close()
        auc_cb = sns.catplot(data=overall_df.loc[(overall_df['Distance'] == 'CB'), :], x='Sequence_Separation',
                             y='AUROC', hue='Method', order=separation_order, hue_order=method_order, kind='box',
                             legend_out=True, sharex=True, sharey=True)
        auc_cb.savefig(auc_cb_box_path)
        plt.close()
    precision_order = ['Precision (L)', 'Precision (L/2)', 'Precision (L/3)', 'Precision (L/4)', 'Precision (L/5)',
                       'Precision (L/6)', 'Precision (L/7)', 'Precision (L/8)', 'Precision (L/9)', 'Precision (L/10)']
    precision_df = pd.melt(overall_df, id_vars=['AUROC', 'Distance', 'Sequence_Separation', 'Time', 'Method', 'Query',
                                                'Query_Length', 'Alignment_Length'], value_vars=precision_order,
                           var_name='Precision_Type', value_name='Precision_Score')
    precision_df['Precision_Type'] = precision_df['Precision_Type'].apply(lambda x: x.split(' ')[1].lstrip('(').rstrip(')'))
    precision_order = ['L', 'L/2', 'L/3', 'L/4', 'L/5', 'L/6', 'L/7', 'L/8', 'L/9', 'L/10']
    pre_any_path = os.path.join(arguments['output'], 'Precision_Method_Comparison_DistAny.eps')
    pre_any_summary_path = os.path.join(arguments['output'], 'Precision_Method_Comparison_DistAny_Summary.eps')
    if (not os.path.isfile(pre_any_path)) or (not os.path.isfile(pre_any_summary_path)):
        pre_any = sns.catplot(data=precision_df.loc[precision_df['Distance'] == 'Any', :], x='Precision_Type',
                              y='Precision_Score', hue='Method', col='Sequence_Separation', row='Query', kind='bar',
                              legend_out=True, sharex=True, sharey=True, order=precision_order, hue_order=method_order,
                              col_order=separation_order, row_order=query_order)
        pre_any.savefig(pre_any_path)
        plt.close()
        pre_any_summary = sns.catplot(data=precision_df.loc[precision_df['Distance'] == 'Any', :], x='Precision_Type',
                                      y='Precision_Score', hue='Method', col='Sequence_Separation', kind='point',
                                      legend_out=True, sharex=True, sharey=True, order=precision_order,
                                      hue_order=method_order, col_order=separation_order, ci=None)
        pre_any_summary.savefig(pre_any_summary_path)
        plt.close()
    pre_cb_path = os.path.join(arguments['output'], 'Precision_Method_Comparison_DistCB.eps')
    pre_cb_summary_path = os.path.join(arguments['output'], 'Precision_Method_Comparison_DistCB_Summary.eps')
    if (not os.path.isfile(pre_cb_path)) or (not os.path.isfile(pre_cb_summary_path)):
        pre_cb = sns.catplot(data=precision_df.loc[precision_df['Distance'] == 'CB', :], x='Precision_Type',
                             y='Precision_Score', hue='Method', col='Sequence_Separation', row='Query', kind='bar',
                             legend_out=True, sharex=True, sharey=True, order=precision_order, hue_order=method_order,
                             col_order=separation_order, row_order=query_order)
        pre_cb.savefig(pre_cb_path)
        plt.close()
        pre_cb_summary = sns.catplot(data=precision_df.loc[precision_df['Distance'] == 'CB', :], x='Precision_Type',
                                     y='Precision_Score', hue='Method', col='Sequence_Separation', kind='point',
                                     legend_out=True, sharex=True, sharey=True, order=precision_order,
                                     hue_order=method_order, col_order=separation_order, ci=None)
        pre_cb_summary.savefig(pre_cb_summary_path)
        plt.close()
    method_order2 = ['ET-MIp', 'cET-MIp']
    g = sns.barplot(x='Query', y='Total Time(s)', hue='Method', order=query_order, hue_order=method_order2,
                    data=times_df.loc[times_df['Method'].isin(method_order2), :])
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.legend(frameon=False)
    plt.savefig(os.path.join(arguments['output'], 'Time_Comparison_DistAny.eps'))
    g.set_yscale('log')
    plt.savefig(os.path.join(arguments['output'], 'Time_Comparison_LogScale.eps'))
    plt.close()
