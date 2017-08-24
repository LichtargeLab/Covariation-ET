'''
Created on Aug 17, 2017

@author: daniel
'''
from multiprocessing import cpu_count
from seaborn import heatmap
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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


# def writeOutClusteringResults(today, qName, cutoff, clus, scorePositions,
#                               etmipResScoreList, etmiplistCoverage, seq,
#                               sortedPDBDist, pdbPositions):
#     '''
#     Write out clustering results
#
#     This method writes the results of the clustering to file.
#
#     Parameters:
#     today: date
#         Todays date.
#     qName: str
#         The name of the query protein
#     clus: int
#         The number of clusters created
#     scorePositions: list
#         A list of the order in which the sequence distances are presented.
#         The element format is {}_{} where each {} is the number of a sequence
#         in the alignment.
#     etmiplistCoverage: list
#         The coverage of a specific sequence comparison
#     sortedPDBDist: numpy nd array
#         Array of the distances between sequences, sorted by sequence indices.
#     seq: Str
#         The query alignment sequence.
#     '''
#     start = time.time()
#     convertAA = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'B': 'ASX',
#                  'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'Z': 'GLX', 'G': 'GLY',
#                  'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET',
#                  'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
#                  'Y': 'TYR', 'V': 'VAL'}
#     e = "{}_{}_{}.etmipCVG.clustered.txt".format(today, qName, clus)
#     etMIPOutFile = open(e, "w+")
#     header = '{}\t({})\t{}\t({})\t{}\t{}\t{}\t{}\n'.format(
#         'Pos1', 'AA1', 'Pos2', 'AA2', 'ETMIp_Score', 'ETMIp_Coverage',
#         'Residue_Dist', 'Within_Threshold', 'Cluster')
#     etMIPOutFile.write(header)
#     counter = 0
#     for i in range(0, len(seq)):
#         for j in range(i + 1, len(seq)):
#             if(sortedPDBDist is None):
#                 res1 = i + 1
#                 res2 = j + 1
#                 r = '-'
#                 dist = '-'
#             else:
#                 res1 = pdbPositions[i]
#                 res2 = pdbPositions[j]
#                 dist = round(sortedPDBDist[counter], 2)
#                 if(sortedPDBDist[counter] <= cutoff):
#                     r = 1
#                 elif(np.isnan(sortedPDBDist[counter])):
#                     r = '-'
#                 else:
#                     r = 0
#             key = '{}_{}'.format(i + 1, j + 1)
#             ind = scorePositions.index(key)
# #             embed()
# #             exit()
#             etMIPOutputLine = '{}\t({})\t{}\t({})\t{}\t{}\t{}\t{}\t{}\n'.format(
#                 res1, convertAA[seq[i]], res2, convertAA[seq[j]],
#                 round(etmipResScoreList[i, j], 2),
#                 round(etmiplistCoverage[ind], 2),
#                 dist, r, clus)
#             etMIPOutFile.write(etMIPOutputLine)
#             counter += 1
#     etMIPOutFile.close()
#     end = time.time()
#     print('Writing the ETMIP worker data to file took {} min'.format(
#         (end - start) / 60.0))


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


def heatmapPlot(name, dataMat):
    print('In Heatmapping')
    dmMax = np.max(dataMat)
    print('Identified max')
    dmMin = np.min(dataMat)
    print('Identified min')
    plotMax = max([dmMax, abs(dmMin)])
    print('Determined highest value')
    heatMap = heatmap(data=dataMat, cmap=cm.jet, center=0.0, vmin=-1 * plotMax,
                      vmax=plotMax, cbar=True, square=True)
    print('Plotted heat map')
    plt.title(name)
    print('Altered title')
    plt.savefig(name.replace(' ', '_') + '.pdf')
    print('Saved figure')
    plt.clf()
    print('Cleared figure')


def surfacePlot(name, dataMat):
    dmMax = np.max(dataMat)
    dmMin = np.min(dataMat)
    plotMax = max([dmMax, abs(dmMin)])
    X = Y = np.arange(max(dataMat.shape))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, dataMat, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1 * plotMax, plotMax)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(name.replace(' ', '_') + '.pdf')
    plt.clf()


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
#     X = queryAlignment.distanceMatrix(saveFile='X')
    queryAlignment.distanceMatrix(saveFile='X')
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
            print('making dir')
            os.mkdir(clusterDir)
        else:
            print('not making dir')
        os.chdir(clusterDir)
        cStart = time.time()
        if(args['pdb']):
            etmipObj.plotAUC(args['query'][0], c, today, args['threshold'])
        etmipObj.writeOutClusterScoring(today, args['query'][0], c)
        etmipObj.writeOutClusteringResults(today, args['query'][0],
                                           args['threshold'], c, sortedPDBDist)
        heatmapPlot('Raw Score Heatmap K {}'.format(c),
                    etmipObj.summaryMatrices[c])
        surfacePlot('Raw Score Surface K {}'.format(c),
                    etmipObj.summaryMatrices[c])
#         heatmapPlot('Coverage Heatmap K {}'.format(c), resCoverage[c])
#         surfacePlot('Coverage Surface K {}'.format(c), resCoverage[c])
        cEnd = time.time()
        timeElapsed = cEnd - cStart
        etmipObj.resultTimes[c] += timeElapsed
        try:
            outDict[c] = "\t{0}\t{1}\t{2}\n".format(c,
                                                    round(
                                                        etmipObj.aucs[c][2], 2),
                                                    round(etmipObj.resultTimes[c], 2))
        except(TypeError):
            outDict[c] = "\t{0}\t{1}\t{2}\n".format(c, '-',
                                                    round(etmipObj.resultTimes[c], 2))
        os.chdir('..')
    writeFinalResults(args['query'][0], today, queryAlignment.seqOrder,
                      queryAlignment.seqLength, args['threshold'], outDict)
    print "Generated results in", createFolder
    os.chdir(startDir)
    end = time.time()
    print('ET MIP took {} minutes to run!'.format((end - start) / 60.0))
