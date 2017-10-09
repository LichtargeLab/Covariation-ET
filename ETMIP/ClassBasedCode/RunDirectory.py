'''
Created on Sep 15, 2017

@author: daniel
'''
from PerformAnalysis import AnalyzeAlignment
from multiprocessing import cpu_count
import argparse
import os
import re


def parseArguments():
    '''
    parse arguments

    This method provides a nice interface for parsing command line arguments
    and includes help functionality.

    Returns:
    --------
    dict:
        A dictionary containing the arguments parsed from the command line and
        their arguments.
    '''
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
                        help='The distance within the molecular structure at which two residues are considered interacting.')
    parser.add_argument('--clusters', metavar='K', type=int, nargs='+',
                        default=[2, 3, 5, 7, 10, 25],
                        help='The clustering constants to use when performing this analysis.')
    parser.add_argument('--combineKs', metavar='C', type=str, nargs='?',
                        default='average', choices=['sum', 'average'],
                        help='')
    parser.add_argument('--combineClusters', metavar='c', type=str, nargs='?',
                        default='evidence_vs_size',
                        choices=['sum', 'average', 'size_weighted',
                                 'evidence_weighted', 'evidence_vs_size'],
                        help='How information should be integrated across clusters resulting from the same clustering constant.')
    parser.add_argument('--alterInput', metavar='a', type=bool, nargs='?',
                        default=False,
                        help='If the input to the MI calculation should be altered to only those sequences in which both residues are not gaps.')
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1,
                        nargs='?', choices=[1, 2, 3, 4], help='How many figures to produce.\n1 = ROC Curves, ETMIP Coverage file, and final AUC and Timing file\n2 = files with all scores at each clustering\n3 = sub-alignment files and plots\n4 = surface plots and heatmaps of ETMIP raw and coverage scores.')
    # Clean command line input
    args = parser.parse_args()
    args = vars(args)
    args['clusters'] = sorted(args['clusters'])
    pCount = cpu_count()
    if(args['processes'] > pCount):
        args['processes'] = pCount
    return args


def parseID(faFile):
    for line in open(faFile, 'rb'):
        check = re.match(r'^>query_(.*)\s?$', line)
        if(check):
            return check.group(1)
        else:
            continue


if __name__ == '__main__':
    args = parseArguments()
    inputFiles = []
    for inDir in args['inputDir']:
        files = os.listdir(inDir)
        inputFiles += map(lambda fileName: inDir + fileName, files)
    inputFiles.sort(key=lambda f: os.path.splitext(f)[1])
    inputDict = {}
    for f in inputFiles:
        print f
        check = re.search(r'(\d[\d|A-Z|a-z]{3}[A-Z]?)', f.split('/')[-1])
        if not check:
            continue
        query = check.group(1).lower()
        if(query not in inputDict):
            inputDict[query] = [None, None, None]
        if(f.endswith('fa')):
            inputDict[query][0] = parseID(f)
            inputDict[query][1] = f
        elif(f.endswith('pdb')):
            inputDict[query][2] = f
        else:
            pass
    counter = 0
    for query in inputDict:
        if inputDict[query][0] is not None:
            counter += 1
            args['query'] = [inputDict[query][0]]
            args['alignment'] = [inputDict[query][1]]
            args['pdb'] = inputDict[query][2]
            AnalyzeAlignment(args)
            print('Completed successfully: {}'.format(query))
    print('{} analyses performed'.format(counter))