'''
Created on Sep 15, 2017

@author: daniel
'''
from PerformAnalysis import AnalyzeAlignment
from multiprocessing import cpu_count
import datetime
import argparse
import os
import re
import sys


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
    # Set up all optional variables to be parsed from the command line
    # (defaults)
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--lowMemoryMode', metavar='l', type=bool, nargs='?',
                        default=False, help='Whether to use low memory mode or not. If low memory mode is engaged intermediate values in the ETMIPC class will be written to file instead of stored in memory. This will reduce the memory footprint but may increase the time to run. Only recommended for very large analyses.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1,
                        nargs='?', choices=[1, 2, 3, 4], help='How many figures to produce.\n1 = ROC Curves, ETMIP Coverage file, and final AUC and Timing file\n2 = files with all scores at each clustering\n3 = sub-alignment files and plots\n4 = surface plots and heatmaps of ETMIP raw and coverage scores.')
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
        os.path.join(path, '..', 'Output/23TestProteinInitialEvaluation/'))
    args['clusters'] = [2, 3, 5, 7, 10, 25]
    args['threshold'] = 8.0
    args['combineKs'] = 'sum'
    args['combineClusters'] = 'sum'
    args['ignoreAlignmentSize'] = True
    args['skipPreAnalyzed'] = True
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
    today = str(datetime.date.today())
    args = parseArguments()
    inputFiles = []
    files = os.listdir(args['inputDir'])
    inputFiles += map(lambda fileName: os.path.abspath(args['inputDir']) + '/' +
                      fileName, files)
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
    incomplete = []
    for query in inputDict:
        if inputDict[query][0] is not None:
            counter += 1
            createFolder = (args['output'] + str(today) + "/" +
                            inputDict[query][0])
            if(os.path.isdir(os.path.abspath(createFolder)) and
               args['skipPreAnalyzed']):
                continue
            print('Performing analysis for: {}'.format(query))
            args['query'] = [inputDict[query][0]]
            args['alignment'] = [inputDict[query][1]]
            args['pdb'] = inputDict[query][2]
            try:
                AnalyzeAlignment(args)
                print('Completed successfully: {}'.format(query))
            except(ValueError):
                print('Analysis for {} incomplete!'.format(query))
                incomplete.append(query)
    print('{} analyses performed'.format(counter))
    print('Incomplete analyses for:\n' + '\n'.join(incomplete))
