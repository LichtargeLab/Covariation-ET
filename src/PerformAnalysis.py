'''
Created on Aug 17, 2017

@author: daniel
'''
from multiprocessing import cpu_count
import datetime
import argparse
import time
import os
from SeqAlignment import SeqAlignment
from PDBReference import PDBReference
from ETMIPC import ETMIPC
from IPython import embed


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
    parser.add_argument('--alignment', metavar='A', type=str, nargs=1,
                        help='The file path to the alignment to analyze in this run.')
    parser.add_argument('--pdb', metavar='P', type=str, nargs='?',
                        help='The file path to the PDB structure associated with the provided alignment.')
    parser.add_argument('--query', metavar='Q', type=str, nargs=1,
                        help='The name of the protein being queried in this analysis.')
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
                        default='sum', choices=['sum', 'average'],
                        help='')
    parser.add_argument('--combineClusters', metavar='c', type=str, nargs='?',
                        default='sum',
                        choices=['sum', 'average', 'size_weighted',
                                 'evidence_weighted', 'evidence_vs_size'],
                        help='How information should be integrated across clusters resulting from the same clustering constant.')
    parser.add_argument('--ignoreAlignmentSize', metavar='i', type=bool, nargs='?',
                        default=False,
                        help='Whether or not to allow alignments with fewer than 125 sequences as suggested by PMID:16159918.')
    parser.add_argument('--lowMemoryMode', metavar='l', type=bool, nargs='?',
                        default=False, help='Whether to use low memory mode or not. If low memory mode is engaged intermediate values in the ETMIPC class will be written to file instead of stored in memory. This will reduce the memory footprint but may increase the time to run. Only recommended for very large analyses.')
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
#     print args
#     embed()
#     exit()
    return args


def AnalyzeAlignment(args):
    start = time.time()
    ###########################################################################
    # Set up global variables
    ###########################################################################
    today = str(datetime.date.today())
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aaDict = {aa_list[i]: i for i in range(len(aa_list))}
    ###########################################################################
    # Set up output location
    ###########################################################################
    startDir = os.getcwd()
    print startDir
    createFolder = os.path.join(args['output'], str(today), args['query'][0])
    if not os.path.exists(createFolder):
        os.makedirs(createFolder)
        print "Creating output folder"
    os.chdir(createFolder)
    createFolder = os.getcwd()
    ###########################################################################
    # Import alignment
    ###########################################################################
    print 'Importing alignment'
    # Create SeqAlignment object to represent the alignment for this analysis.
    if(args['alignment'][0].startswith('..')):
        queryAlignment = SeqAlignment(queryID=args['query'][0],
                                      fileName=(os.path.join(startDir, args['alignment'][0])))
    else:
        queryAlignment = SeqAlignment(fileName=args['alignment'][0],
                                      queryID=args['query'][0])
    # Import alignment information from file.
    queryAlignment.importAlignment(saveFile='alignment_dict.pkl')
    # Check if alignment meets analysis criteria:
    if((not args['ignoreAlignmentSize']) and (queryAlignment.size < 125)):
        raise ValueError('The multiple sequence alignment is smaller than recommended for performing this analysis ({} < 125, see PMID:16159918), if you wish to proceed with the analysis anyway please call the code again using the --ignoreAlignmentSize option.'.format(queryAlignment.size))
    if(queryAlignment.size < max(args['clusters'])):
        raise ValueError('The analysis could not be performed because the alignment has fewer sequences than the requested number of clusters ({} < {}), please provide an alignment with more sequences or change the clusters requested by using the --clusters option when using this software.'.format(
            queryAlignment.size, max(args['clusters'])))
    # Remove gaps from aligned query sequences
    queryAlignment.removeGaps(saveFile='ungapped_alignment.pkl')
    # Create matrix converting sequences of amino acids to sequences of integers
    # representing sequences of amino acids.
    queryAlignment.alignment2num(aaDict)
    # Write the ungapped alignment to file.
    queryAlignment.writeOutAlignment(fileName='UngappedAlignment.fa')
    # Compute distance between all sequences in the alignment
    queryAlignment.computeDistanceMatrix(saveFile='X')
    # Determine the full clustering tree for the alignment and the ordering of
    # its sequences.
    queryAlignment.setTreeOrdering()
    print('Query Sequence:')
    print(queryAlignment.querySequence)
    ###########################################################################
    # Import the PDB if provided.
    ###########################################################################
    if(args['pdb']):
        # Create PDBReference object to represent the structure for this
        # analysis.
        if(args['pdb'].startswith('..')):
            queryStructure = PDBReference(os.path.join(startDir, args['pdb']))
        else:
            queryStructure = PDBReference(args['pdb'])
        # Import the structure information from the file.
        queryStructure.importPDB(saveFile='pdbData.pkl')
        # Map between the query sequence in the alignment and the structure.
        queryStructure.mapAlignmentToPDBSeq(queryAlignment.querySequence)
        # Determine the shortest distance between residue pairs.
        queryStructure.findDistance(saveFile='PDBdistances')
        print('PDB Sequence')
        print(queryStructure.seq[queryStructure.fastaToPDBMapping[0]])
    else:
        queryStructure = None
    ###########################################################################
    # Perform multiprocessing of clustering method
    ###########################################################################
    print 'Starting ETMIP'
    # Create ETMIPC object to represent the analysis being performed.
    etmipObj = ETMIPC(queryAlignment, args['clusters'], queryStructure,
                      createFolder, args['processes'], args['lowMemoryMode'])
    # Calculate the MI scores for all residues across all sequences
    etmipObj.determineWholeMIP('evidence' in args['combineClusters'])
    # Calculate the the ETMIPC scores for various clustering constants.
    etmipObj.calculateClusteredMIPScores(aaDict=aaDict,
                                         wCC=args['combineClusters'])
    # Combine the clustering results across all clustering constants tested.
    etmipObj.combineClusteringResults(combination=args['combineKs'])
    # Compute normalized scores for ETMIPC and evaluate against PDB if
    # provided.
    etmipObj.computeCoverageAndAUC(threshold=args['threshold'])
    # Write out cluster specific scores and produce figures.
    etmipObj.produceFinalFigures(today, cutOff=args['threshold'],
                                 verbosity=args['verbosity'])
    # Write out the AUCs and final times for the different clustering constants
    # tested.
    etmipObj.writeFinalResults(today, args['threshold'])
    # If low memory mode was used clear out intermediate files saved in this
    # process.
    if(args['lowMemoryMode']):
        etmipObj.clearIntermediateFiles()
    print "Generated results in: ", createFolder
    os.chdir(startDir)
    end = time.time()
    print('ET MIP took {} minutes to run!'.format((end - start) / 60.0))


if __name__ == '__main__':
    # Read input from the command line
    args = parseArguments()
    # Perform analysis
    AnalyzeAlignment(args)
