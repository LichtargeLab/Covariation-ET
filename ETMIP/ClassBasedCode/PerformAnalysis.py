'''
Created on Aug 17, 2017

@author: daniel
'''
from multiprocessing import cpu_count
import numpy as np
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
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--alignment', metavar='A', type=str, nargs=1,
                        help='The file path to the alignment to analyze in this run.')
    parser.add_argument('--pdb', metavar='P', type=str, nargs='?',
                        help='The file path to the PDB structure associated with the provided alignment.')
    parser.add_argument('--query', metavar='Q', type=str, nargs=1,
                        help='The name of the protein being queried in this analysis.')
    parser.add_argument('--threshold', metavar='T', type=float, nargs='?',
                        help='The distance within the molecular structure at which two residues are considered interacting.')
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--output', metavar='O', type=str, nargs='?',
                        default='./', help='File path to a directory where the results can be generated.')
    args = parser.parse_args()
    return vars(args)


def writeFinalResults(qName, today, sequenceOrder, seqLength, cutoff, outDict):
    '''
    Write final results

    This method writes the final results to file for an analysis.  In this case
    that consists of the cluster numbers, the resulting AUCs, and the time
    spent in processing.

    Parameters:
    -----------
    qName: str
        The id for the query sequence.
    today: str
        The current date in string format.
    sequenceOrder: list
        A list of the sequence ids used in the alignment in sorted order.
    seqLength: int
        The length of the gap removed sequences from the alignment.
    cutoff: float
        The distance threshold for interaction between two residues in a
        protein structure.
    outDict: dict
        A dictionary with the lines of output mapped to the clustering
        constants for which they were produced.
    '''
    o = '{}_{}etmipAUC_results.txt'.format(qName, today)
    outfile = open(o, 'w+')
    proteininfo = ("Protein/id: " + qName + " Alignment Size: " +
                   str(len(sequenceOrder)) + " Length of protein: " +
                   str(seqLength) + " Cutoff: " + str(cutoff) + "\n")
    outfile.write(proteininfo)
    outfile.write("#OfClusters\tAUC\tRunTime\n")
    for key in sorted(outDict.keys()):
        outfile.write(outDict[key])


if __name__ == '__main__':
    ###########################################################################
    # BODY OF CODE ##
    ###########################################################################
    start = time.time()
    # Read input from the command line
    args = parseArguments()
    ###########################################################################
    # Set up global variables
    ###########################################################################
    today = str(datetime.date.today())
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aaDict = {aa_list[i]: i for i in range(len(aa_list))}
    ###########################################################################
    # Set up input variables
    ###########################################################################
    processes = args['processes']
    pCount = cpu_count()
    if(processes > pCount):
        processes = pCount
    ###########################################################################
    # Set up output location
    ###########################################################################
    startDir = os.getcwd()
    print startDir
    createFolder = (args['output'] + str(today) + "/" + args['query'][0])
    if not os.path.exists(createFolder):
        os.makedirs(createFolder)
        print "creating new folder"
    os.chdir(createFolder)
    createFolder = os.getcwd()
    ###########################################################################
    # Import alignment and perform initial analysis
    ###########################################################################
    print 'Starting ETMIP'
    # Import alignment information: this will be our alignment
    queryAlignment = SeqAlignment(fileName=startDir + '/' + args['alignment'][0],
                                  queryID=args['query'][0])
    queryAlignment.importAlignment(saveFile='alignment_dict.pkl')
    # Remove gaps from aligned query sequences
    queryAlignment.removeGaps(saveFile='ungapped_alignment.pkl')
    # Create matrix converting sequences of amino acids to sequences of integers
    # representing sequences of amino acids
    queryAlignment.alignment2num(aaDict)
    queryAlignment.writeOutAlignment(fileName='UngappedAlignment.fa')
    print('Query Sequence:')
    print(queryAlignment.querySequence)
    # I will get a corr_dict for method x for all residue pairs FOR ONE PROTEIN
    queryAlignment.computeDistanceMatrix(saveFile='X')
    # Generate MIP Matrix
    ###########################################################################
    # Set up for remaining analysis
    ###########################################################################
    if(args['pdb']):
        queryStructure = PDBReference(startDir + '/' + args['pdb'])
        queryStructure.importPDB(saveFile='pdbData.pkl')
        queryStructure.mapAlignmentToPDBSeq(queryAlignment.querySequence)
        sortedPDBDist = queryStructure.findDistance(
            queryAlignment.querySequence, saveFile='PDBdistances')
    else:
        queryStructure = None
    print('PDB Sequence')
    print(queryStructure.seq)
    ###########################################################################
    # Perform multiprocessing of clustering method
    ###########################################################################
    clusters = [2, 3, 5, 7, 10, 25]

    etmipObj = ETMIPC(queryAlignment, clusters, queryStructure, createFolder,
                      processes)
    etmipObj.determineWholeMIP()
    etmipObj.calculateClusteredMIPScores(aaDict=aaDict)
#     etmipObj.combineClusteringResults(combination='addative')
    etmipObj.combineClusteringResults(combination='average')
    etmipObj.computeCoverageAndAUC(sortedPDBDist=sortedPDBDist,
                                   threshold=args['threshold'])
    outDict = {}
    for c in clusters:
        clusterDir = '{}/'.format(c)
        if(not os.path.exists(clusterDir)):
            os.mkdir(clusterDir)
        os.chdir(clusterDir)
        cStart = time.time()
        if(args['pdb']):
            etmipObj.plotAUC(args['query'][0], c, today, args['threshold'])
        etmipObj.writeOutClusterScoring(today, args['query'][0], c)
        etmipObj.writeOutClusteringResults(today, args['query'][0],
                                           args['threshold'], c, sortedPDBDist)
        etmipObj.heatmapPlot('Raw Score Heatmap K {}'.format(c),
                             normalized=False, cluster=c)
        etmipObj.surfacePlot('Raw Score Surface K {}'.format(c),
                             normalized=False, cluster=c)
        etmipObj.heatmapPlot('Coverage Heatmap K {}'.format(c), normalized=True,
                             cluster=c)
        etmipObj.surfacePlot('Coverage Surface K {}'.format(c), normalized=True,
                             cluster=c)
        cEnd = time.time()
        timeElapsed = cEnd - cStart
        etmipObj.resultTimes[c] += timeElapsed
        outDict[c] = "\t{0}\t{1}\t{2}\n".format(c,
                                                round(etmipObj.aucs[c][2], 2),
                                                round(etmipObj.resultTimes[c], 2))
        os.chdir('..')
    writeFinalResults(args['query'][0], today, queryAlignment.seqOrder,
                      queryAlignment.seqLength, args['threshold'], outDict)
    print "Generated results in", createFolder
    os.chdir(startDir)
    end = time.time()
    print('ET MIP took {} minutes to run!'.format((end - start) / 60.0))