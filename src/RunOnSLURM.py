"""
Created on Sep 15, 2017

@author: daniel
"""
from multiprocessing import cpu_count
from subprocess import call
import argparse
import os
import re


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
    # Set up all variables to be parsed from the command line (no defaults)
    parser.add_argument('--inputDir', metavar='I', type=str, nargs='+',
                        help='Directory containing pdb and fa files for running ETMIP analysis.')
    parser.add_argument('--output', metavar='O', type=str, nargs='?',
                        default='./', help='File path to a directory where the results can be generated.')
    # Set up all optional variables to be parsed from the command line
    # (defaults)
    parser.add_argument('--threshold', metavar='T', type=float, nargs='?',
                        default=8.0,
                        help='The distance within the molecular structure at which two residues are considered '
                             'interacting.')
    parser.add_argument('--clusters', metavar='K', type=int, nargs='+',
                        default=[2, 3, 5, 7, 10, 25],
                        help='The clustering constants to use when performing this analysis.')
    parser.add_argument('--combineKs', metavar='C', type=str, nargs='?',
                        default='sum', choices=['sum', 'average'],
                        help='')
    parser.add_argument('--combineClusters', metavar='c', type=str, nargs='?',
                        default='sum',
                        choices=['sum', 'average', 'size_weighted',
                                 'evidence_weighted', 'evidence_vs_size'],
                        help='How information should be integrated across clusters resulting from the same clustering '
                             'constant.')
    parser.add_argument('--ignoreAlignmentSize', metavar='i', type=bool, nargs='?',
                        default=False,
                        help='Whether or not to allow alignments with fewer than 125 sequences as suggested by '
                             'PMID:16159918.')
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1,
                        nargs='?', choices=[1, 2, 3, 4], help='How many figures to produce.\n1 = ROC Curves, ETMIP '
                                                              'Coverage file, and final AUC and Timing file\n2 = '
                                                              'files with all scores at each clustering\n3 = '
                                                              'sub-alignment files and plots\n4 = surface plots and '
                                                              'heatmaps of ETMIP raw and coverage scores.')
    # Clean command line input
    arguments = parser.parse_args()
    arguments = vars(arguments)
    arguments['clusters'] = sorted(arguments['clusters'])
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


def write_out_sbatch_script(top_dir, arguments):
    sb_fn = top_dir + 'ETMIP_{}.sh'.format(arguments['query'][0])
    file_handle = open(sb_fn, 'wb')
    file_handle.write("#!/usr/bin/bash\n")
    file_handle.write("#SBATCH --output={}{}.out\n".format(
        arguments['output'], arguments['query'][0]))
    file_handle.write("#SBATCH -e {}{}.err\n".format(arguments['output'],
                                                     arguments['query'][0]))
    file_handle.write("#SBATCH --mem=40960\n")
    file_handle.write("#SBATCH -c 12\n")
    file_handle.write("#SBATCH -n 1\n")
    file_handle.write("#SBATCH -N 1\n")
    file_handle.write("#SBATCH --job-name={}_ETMIP\n".format(arguments['query'][0]))
    file_handle.write("\n")
    file_handle.write("echo 'Switching to file directory'\n")
    file_handle.write("\n")
    file_handle.write(
        "cd /storage/lichtarge/home/konecki/GIT/ETMIP/ETMIP/ClassBasedCode/\n")
    file_handle.write("\n")
    file_handle.write("echo 'Activating Python Environment'\n")
    file_handle.write("\n")
    file_handle.write("source activate pyETMIPC\n")
    file_handle.write("\n")
    file_handle.write("Starting {} ETMIPC Analysis\n".format(arguments['query'][0]))
    file_handle.write("\n")
    call_string = "python PerformAnalysis.py"
    for key in arguments:
        call_string += " --{} ".format(key)
        if key == 'query':
            call_string += "'{}'".format(arguments[key][0])
        else:
            if type(arguments[key]) == list:
                call_string += " ".join(map(str, arguments[key]))
            else:
                call_string += str(arguments[key])
#     fileHandle.write(
#         "python PerformAnalysis.py --verbosity 4 --processes 11 --alignment {} --pdb {} --query {} --output {}")
    call_string += '\n'
    file_handle.write(call_string)
    file_handle.write("\n")
    file_handle.write("echo '{} Analysis Complete\n".format(arguments['query'][0]))
    file_handle.write("\n")
    file_handle.write("echo 'Deactivating Python Environment\n")
    file_handle.write("\n")
    file_handle.write("source deactivate\n")
    file_handle.write("\n")
    file_handle.write("Job Completed")
    file_handle.close()
    return sb_fn


if __name__ == '__main__':
    args = parse_arguments()
    input_files = []
    for in_dir in args['inputDir']:
        files = os.listdir(in_dir)
        input_files += map(lambda file_name: in_dir + file_name, files)
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
    del(args['inputDir'])
    output_dir = os.path.abspath(args['output'])
    if not output_dir.endswith('/'):
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    counter = 0
    for query in input_dict:
        if input_dict[query][0] is not None:
            counter += 1
            args['query'] = [input_dict[query][0]]
            args['alignment'] = [input_dict[query][1]]
            args['pdb'] = input_dict[query][2]
            args['output'] = output_dir + '{}/'.format(query)
            if not os.path.exists(args['output']):
                os.mkdir(args['output'])
            curr_fn = write_out_sbatch_script(output_dir, args)
            print curr_fn
            status = call(['sbatch', curr_fn])
            print('{} return status: {}'.format(query, status))
    print('{} analyses submitted'.format(counter))
