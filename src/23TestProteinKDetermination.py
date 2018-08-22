"""
Created on Sep 15, 2017

@author: daniel
"""
from SupportingClasses.SeqAlignment import SeqAlignment
from PerformAnalysis import AnalyzeAlignment
from multiprocessing import cpu_count
import datetime
import argparse
import numpy as np
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
                        help='The number of processes to spawn when'
                        ' multiprocessing this analysis.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1,
                        nargs='?', choices=[1, 2, 3, 4],
                        help='How many figures to produce.\n1 = ROC Curves, '
                        'ETMIP Coverage file, and final AUC and Timing file\n2 '
                        '= files with all scores at each clustering\n3 = sub-'
                        'alignment files and plots\n4 = surface plots and '
                        'heatmaps of ETMIP raw and coverage scores.')
    # Clean command line input
    args = parser.parse_args()
    args = vars(args)
    # Scribbed from python dotenv package source code
    frame = sys._getframe()
    frame_filename = frame.f_code.co_filename
    path = os.path.dirname(os.path.abspath(frame_filename))
    # End scrib

    args['inputDir'] = os.path.abspath(os.path.join(path, '..',
                                                    'Input/23TestGenes/'))
    args['output'] = os.path.abspath(
        os.path.join(path, '..', 'Output/23TestProteinKDetermination/'))
    args['threshold'] = 8.0
    args['combineKs'] = 'sum'
    args['combineClusters'] = 'sum'
    args['ignoreAlignmentSize'] = True
    args['skipPreAnalyzed'] = True
    args['lowMemoryMode'] = True
    processor_count = cpu_count()
    if(args['processes'] > processor_count):
        args['processes'] = processor_count
    return args


def parse_id(fa_file):
    for line in open(fa_file, 'rb'):
        id_check = re.match(r'^>query_(.*)\s?$', line)
        if(id_check):
            return id_check.group(1)
        else:
            continue


def get_alignment_stats(file_path, query_id):
    print(file_path)
    print(query_id)
    curr_aln = SeqAlignment(file_path, query_id)
    curr_aln.importAlignment()
    return curr_aln.seqLength, curr_aln.size, int(np.floor(curr_aln.size / 3.0))


if __name__ == '__main__':
    today = str(datetime.date.today())
    arguments = parse_arguments()
    input_files = []
    files = os.listdir(arguments['inputDir'])
    input_files += map(lambda file_name: os.path.abspath(os.path.join(
        arguments['inputDir'], file_name)), files)
    input_files.sort(key=lambda f: os.path.splitext(f)[1])
    input_dict = {}
    for f in input_files:
        print f
        check = re.search(r'(\d[\d|A-Z|a-z]{3}[A-Z]?)', f.split('/')[-1])
        if not check:
            continue
        query = check.group(1).lower()
        if(query not in input_dict):
            input_dict[query] = [None, None, None, None, None, None]
        if(f.endswith('fa')):
            input_dict[query][0] = parse_id(f)
            input_dict[query][1] = f
            aln_stats = get_alignment_stats(f, input_dict[query][0])
            input_dict[query][3] = aln_stats[0]
            input_dict[query][4] = aln_stats[1]
            input_dict[query][5] = aln_stats[2]
        elif(f.endswith('pdb')):
            input_dict[query][2] = f
        else:
            pass
    counter = 0
    incomplete = []
    for query in sorted(input_dict, key=lambda k: input_dict[k][4]):
        if input_dict[query][0] is not None:
            counter += 1
            create_folder = (arguments['output'] + str(today) + "/" +
                             input_dict[query][0])
            if(os.path.isdir(os.path.abspath(create_folder)) and
               arguments['skipPreAnalyzed']):
                continue
            print('Performing analysis for: {} with query length {} size {} and'
                  ' max k {}'.format(query, input_dict[query][3],
                                     input_dict[query][4],
                                     input_dict[query][5]))
            arguments['query'] = [input_dict[query][0]]
            arguments['alignment'] = [input_dict[query][1]]
            arguments['pdb'] = input_dict[query][2]
            arguments['clusters'] = range(2, input_dict[query][5] + 1)
            try:
                AnalyzeAlignment(arguments)
                print('Completed successfully: {}'.format(query))
            except(ValueError):
                print('Analysis for {} incomplete!'.format(query))
                incomplete.append(query)
    print('{} analyses performed'.format(counter))
    print('Incomplete analyses for:\n' + '\n'.join(incomplete))
