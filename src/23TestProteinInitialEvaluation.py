"""
Created on Sep 15, 2017

@author: daniel
"""
from PerformAnalysis import analyze_alignment
from SupportingClasses.ContactScorer import ContactScorer
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.ETMIPWrapper import ETMIPWrapper
from SupportingClasses.DCAWrapper import DCAWrapper
from multiprocessing import cpu_count
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
    --------
    dict:
        A dictionary containing the arguments parsed from the command line and
        their arguments.
    """
    # Create input parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    # Set up all optional variables to be parsed from the command line
    # (defaults)
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
               'ET-MIp': {'class': ETMIPWrapper, 'args': {'delete_files': False}}} #, 'cET-MIp': {}}
    if not os.path.isdir(arguments['output']):
        os.mkdir(arguments['output'])
    ####################################################################################################################
    ####################################################################################################################
    times = {'query': [], 'method': [], 'time(s)': []}
    aucs = {'query': [], 'method': [], 'score': [], 'distance': [], 'sequence_separation': []}
    precisions = {'query': [], 'method': [], 'score': [], 'distance': [], 'sequence_separation': [], 'k': []}
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
        contact_any.fit()
        contact_any.measure_distance(method='Any')
        contact_beta = ContactScorer(seq_alignment=query_aln, pdb_reference=query_structure, cutoff=8.0)
        contact_beta.fit()
        contact_beta.measure_distance(method='CB')
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
            curr_time = predictor.calculate_scores(out_dir=protein_dir, **methods[method]['args'])
            times['query'].append(query)
            times['method'].append(method)
            times['time(s)'].append(curr_time)
            # Score Prediction Clustering
            z_score_any_biased = contact_any.score_clustering_of_contact_predictions(
                predictor.scores, bias=True, cutoff=4.0, file_path=os.path.join(protein_dir,
                                                                                          'DistAny_Biased_ZScores.tsv'))
                                                                                     # bias=True, cutoff=8.0)
            contact_any.plot_z_scores(z_score_any_biased, os.path.join(protein_dir, 'DistAny_Biased_ZScores.eps'))
            z_score_any_unbiased = contact_any.score_clustering_of_contact_predictions(
                predictor.scores, bias=False, cutoff=4.0, file_path=os.path.join(protein_dir,
                                                                                          'DistAny_Unbiased_ZScores.tsv'))
                                                                                       # bias=False, cutoff=8.0)
            contact_any.plot_z_scores(z_score_any_unbiased, os.path.join(protein_dir, 'DistAny_Unbiased_ZScores.eps'))
            z_score_beta_biased = contact_beta.score_clustering_of_contact_predictions(
                predictor.scores, bias=True, cutoff=4.0, file_path=os.path.join(protein_dir,
                                                                                          'DistBeta_Biased_ZScores.tsv'))
                                                                                       # bias=True, cutoff=8.0)
            contact_beta.plot_z_scores(z_score_beta_biased, os.path.join(protein_dir, 'DistBeta_Biased_ZScores.eps'))
            z_score_beta_unbiased = contact_beta.score_clustering_of_contact_predictions(
                predictor.scores, bias=False, cutoff=4.0, file_path=os.path.join(protein_dir,
                                                                                          'DistBeta_Unbiased_ZScores.tsv'))
                                                                                         # bias=False, cutoff=8.0)
            contact_beta.plot_z_scores(z_score_beta_biased, os.path.join(protein_dir, 'DistBeta_Unbiased_ZScores.eps'))
            # Evaluating scores
            for separation in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
                # AUC Evaluation
                auc_roc_any = contact_any.score_auc(predictor.scores, category=separation)
                aucs['query'].append(query)
                aucs['method'].append(method)
                aucs['score'].append(auc_roc_any[2])
                aucs['distance'].append('Any')
                aucs['sequence_separation'].append(separation)
                contact_any.plot_auc(query_name=query, auc_data=auc_roc_any, title='AUROC Evaluation',
                                     file_name='AUROC Evaluation_{}_{}'.format('Any', separation),
                                     output_dir=protein_dir)
                auc_roc_beta = contact_beta.score_auc(predictor.scores, category=separation)
                aucs['query'].append(query)
                aucs['method'].append(method)
                aucs['score'].append(auc_roc_beta[2])
                aucs['distance'].append('Beta')
                aucs['sequence_separation'].append(separation)
                contact_any.plot_auc(query_name=query, auc_data=auc_roc_any, title='AUROC Evaluation',
                                     file_name='AUROC Evaluation_{}_{}'.format('Beta', separation),
                                     output_dir=protein_dir)
                # Precision Evaluation
                for k in range(1, 11):
                    precision_any = contact_any.score_precision(predictions=predictor.scores, k=k,
                                                                category=separation)
                    precisions['query'].append(query)
                    precisions['method'].append(method)
                    precisions['score'].append(precision_any)
                    precisions['distance'].append('Any')
                    precisions['sequence_separation'].append(separation)
                    precisions['k'].append(k)
                    precision_beta = contact_beta.score_precision(predictions=predictor.scores, k=k,
                                                                category=separation)
                    precisions['query'].append(query)
                    precisions['method'].append(method)
                    precisions['score'].append(precision_beta)
                    precisions['distance'].append('Beta')
                    precisions['sequence_separation'].append(separation)
                    precisions['k'].append(k)
