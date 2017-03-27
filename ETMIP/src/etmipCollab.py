'''
Created on Mar 10, 2017

@author: Benu Atri
'''

from sklearn.metrics import roc_curve, auc, mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Pool, cpu_count, Manager
from itertools import izip
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
import datetime
import time
import sys
import re
import os


def importAlignment(files, saveFile=None):
    '''
    Import alignments:

    This method imports alignments into an existing dictionary.

    Parameters:
    -----------
    files: File
        File object holding a handle to an alignment file.
    saveFile: str
        Path to file in which the desired alignment was stored previously.
    Returns:
    --------
    alignment_dict: dict    
        Dictionary which will be used to store alignments from the file.
    '''
    start = time.time()
    if((saveFile is not None) and (os.path.exists(saveFile))):
        alignment_dict = pickle.load(open(saveFile, 'rb'))
    else:
        alignment_dict = {}
        for line in files:
            if line.startswith(">"):
                key = line.rstrip()
                alignment_dict[key] = ''
            else:
                alignment_dict[key] += line.rstrip()
        if(saveFile is not None):
            pickle.dump(alignment_dict, open(saveFile, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Importing alignment took {} min'.format((end - start) / 60.0))
    return alignment_dict


def removeGaps(alignment_dict, saveFile=None):
    '''
    Remove Gaps

    Removes all gaps from the query sequence and removes characters at the
    corresponding positions in all other sequences.

    Parameters:
    -----------
    alignment_dict: dict
        Dictionary mapping query name to sequence.
    saveFile: str
        Path to a file where the alignment with gaps in the query sequence
        removed was stored previously.

    Returns:
    --------
    str
        A new query name.
    dict
        A transform of the input dictionary without gaps.
    '''
    # Getting gapped columns for query
    start = time.time()
    if((saveFile is not None) and os.path.exists(saveFile)):
        query_name, new_alignment_dict = pickle.load(
            open(saveFile, 'rb'))
    else:
        gap = ['-', '.', '_']
        query_gap_index = []
        for key, value in alignment_dict.iteritems():
            if "query" in key.lower():
                query_name = key
                for idc, char in enumerate(value):
                    if char in gap:
                        query_gap_index.append(idc)
        if(len(query_gap_index) > 0):
            query_gap_index.sort()
            new_alignment_dict = {}
            for key, value in alignment_dict.iteritems():
                new_alignment_dict[key] = value[0:query_gap_index[0]]
                for i in range(1, len(query_gap_index) - 1):
                    new_alignment_dict[key] += value[query_gap_index[i]:
                                                     query_gap_index[i + 1]]
                new_alignment_dict[key] += value[query_gap_index[-1]:]
        else:
            new_alignment_dict = alignment_dict
        if(saveFile is not None):
            pickle.dump((query_name, new_alignment_dict),
                        open(saveFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Removing gaps took {} min'.format((end - start) / 60.0))
    return query_name, new_alignment_dict


def distanceMatrix(alignment_dict, saveFiles=None):
    '''
    Distance matrix

    Computes the sequence identity distance between a set of sequences and
    returns a matrix of the pairwise distances.

    Parameters:
    -----------
    alignment_dict: dict
        Dictionary of aligned sequences. This is meant to be a corrected
        dictionary where all gaps have been removed from the query sequence,
        and the same positions have been removed from other sequences.
    saveFiles: tuple
        A tuple or list containing two file paths the first should be the path
        for a .npz file containing distances between sequences in the alignment
        (leave out the .npz as it will be added automatically) and the file
        path for a .pkl file containing the the sequence order for the distance
        matrix.
    Returns:
    --------
    matrix
        A symmetric matrix of pairwise distance computed between two sequences
        using the sequence identity metric.
    list
        List of the sequence identifiers in the order in which they appear in
        the matrix.
    '''
    # Generate distanceMatrix: Calculating Sequence Identity
    start = time.time()
    if((saveFiles is not None) and os.path.exists(saveFiles[0]) and
       os.path.exists(saveFiles[1])):
        valuematrix = np.load(saveFiles[0] + '.npz')['X']
        key_list = pickle.load(open(saveFiles[1], 'rb'))
    else:
        key_list = sorted(alignment_dict.keys())
        valuematrix = np.zeros([len(alignment_dict), len(alignment_dict)])
        seq_length = len(alignment_dict[key_list[0]])
        for i in range(len(key_list)):
            seq1 = alignment_dict[key_list[i]]
            for j in range(i + 1, len(key_list)):
                seq2 = alignment_dict[key_list[j]]
                simm = np.sum(ch1 == ch2 for ch1, ch2 in izip(seq1, seq2))
                valuematrix[i, j] += simm
                valuematrix[j, i] += simm
        valuematrix /= seq_length
        if(saveFiles is not None):
            np.savez(saveFiles[0], X=valuematrix)
            pickle.dump(key_list, open(saveFiles[1], 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Computing the distance matrix took {} min'.format(
        (end - start) / 60.0))
    return valuematrix, key_list


def aggClustering(n_cluster, X, key_list, precomputed=False):
    '''
    Agglomerative clustering

    Performs agglomerative clustering on a matrix of pairwise distances between
    sequences in the alignment being analyzed.

    Parameters:
    -----------
    n_cluster: int
        The number of clusters to separate sequences into.
    X: numpy nd_array
        The distance matrix computed between the sequences.
    alignment_dict: dict
        The corrected alignment dictionary containing sequences with all gaps
        removed for the query sequence.
    Returns:
    --------
    dict
        A dictionary with cluster number as the key and a list of sequences in
        the specified cluster as a value.
    set
        A unique sorted set of the cluster values.
    '''
    start = time.time()
    if(precomputed):
        affinity = 'precomputed'
        linkage = 'complete'
    else:
        affinity = 'euclidean'
        linkage = 'ward'
    model = AgglomerativeClustering(affinity=affinity, linkage=linkage,
                                    n_clusters=n_cluster)
    model.fit(X)
    # ordered list of cluster ids
    # unique and sorted cluster ids for e.g. for n_cluster = 2, g = [0,1]
    clusterlist = model.labels_.tolist()
    cluster_dict = {}
    ####---------------------------------------#####
    #       Mapping Clusters to Sequences
    ####---------------------------------------#####
    for i in range(len(clusterlist)):
        key = clusterlist[i]
        if(key not in cluster_dict):
            cluster_dict[key] = []
        cluster_dict[key].append(key_list[i])
    end = time.time()
    print('Performing agglomerative clustering took {} min'.format(
        (end - start) / 60.0))
    return cluster_dict, set(clusterlist)


def wholeAnalysis(alignment, aa_dict, saveFile=None):
    '''
    Whole Analysis

    Generates the MIP matrix.

    Parameters:
    -----------
    alignment_dict: dict
        Dictionary of aligned sequences. This is meant to be a corrected
        dictionary where all gaps have been removed from the query sequence,
        and the same positions have been removed from other sequences.
    aa_list: list
        List of amino acids in a fixed order.
    saveFile: str
        File path to a previously stored MIP matrix (.npz should be excluded as
        it will be added automatically).
    Returns:
    --------
    matrix
        Matrix of MIP scores which has dimensions seq_length by seq_length.
    '''
    start = time.time()
    if((saveFile is not None) and os.path.exists(saveFile + '.npz')):
        MIP_matrix = np.load(saveFile + '.npz')['wholeMIP']
    else:
        overallMMI = 0.0
        key_list = alignment.keys()
        seq_length = len(alignment[alignment.keys()[0]])
        # generate an MI matrix for each cluster
        MI_matrix = np.zeros((seq_length, seq_length))
        # Vector of 1 column
        MMI = np.zeros(seq_length)
        APC_matrix = np.zeros((seq_length, seq_length))
        MIP_matrix = np.zeros((seq_length, seq_length))
        # Create matrix converting sequences of amino acids to sequences of integers
        # representing sequences of amino acids.
        alignment2Num = np.zeros((len(alignment), seq_length))
        for i in range(len(key_list)):
            for j in range(seq_length):
                alignment2Num[i, j] = aa_dict[alignment[key_list[i]][j]]
        # Generate MI matrix from alignment2Num matrix, the MMI matrix,
        # and overallMMI
        for i in range(seq_length):
            for j in range(i + 1, seq_length):
                column_i = alignment2Num[:, i]
                column_j = alignment2Num[:, j]
                currMIS = mutual_info_score(
                    column_i, column_j, contingency=None)
                # AW: divides by individual entropies to normalize.
                MI_matrix[i, j] = MI_matrix[j, i] = currMIS
                overallMMI += currMIS
        MMI += np.sum(MI_matrix, axis=1)
        MMI -= MI_matrix[np.arange(seq_length), np.arange(seq_length)]
        MMI /= (seq_length - 1)
        overallMMI = 2.0 * (overallMMI / (seq_length - 1)) / seq_length
        # Calculating APC
        APC_matrix += np.outer(MMI, MMI)
        APC_matrix[np.arange(seq_length), np.arange(seq_length)] = 0
        APC_matrix /= overallMMI
        # Defining MIP matrix
        MIP_matrix += MI_matrix - APC_matrix
        MIP_matrix[np.arange(seq_length), np.arange(seq_length)] = 0
        if(saveFile is not None):
            np.savez(saveFile, wholeMIP=MIP_matrix)
    end = time.time()
    print('Whole analysis took {} min'.format((end - start) / 60.0))
    return MIP_matrix


def importPDB(pdbFile, saveFile=None):
    '''
    importPDB

    This method imports a PDB files information generating a list of lists. Each
    list contains the Amino Acid 3-letter abbreviation, residue number, x, y,
    and z coordinate.

    Parameters:
    -----------
    filename: string
        The file path to the PDB file.
    saveFile: str
        The file path to a previously stored PDB file data structure.
    Returns:
    --------
    list:
        A list of lists containing the relevant information from the PDB file.
    '''
    start = time.time()
    if((saveFile is not None) and os.path.exists(saveFile)):
        rows = pickle.load(open(saveFile, 'rb'))
    else:
        rows = []
        pdbPattern = r'ATOM\s*(\d+)\s*(\w*)\s*([A-Z]{3})\s*([A-Z])\s*(\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*([A-Z])'
        for line in pdbFile:  # for a line in the pdb
            res = re.match(pdbPattern, line)
            if res:
                terms = [res.group(i) for i in [3, 5, 6, 7, 8]]
                rows.append(terms)
        pdbFile.close()
        if(saveFile is not None):
            pickle.dump(rows, open(saveFile, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Importing the PDB file took {} min'.format((end - start) / 60.0))
    return rows


def findDistance(pdbData, saveFiles=None):
    '''
    Find distance

    This code takes in an input of a pdb file and outputs a dictionary with the
    nearest atom distance between two residues.

    Parameters:
    -----------
    filename: string
        The file path to the pdb file.
    saveFile: str
        File name and/or location of file containing a previously computed set
        of distance data for a PDB structure.
    Returns:
    dict
        Dictionary of distances
    list
        List of residues from the PDB file.
    dict
        Dictionary of residue.
    --------
    '''
    # create dictionary of every atom in each individual residue. 3
    # Dimensional coordinates of each residue position
    start = time.time()
    if((saveFiles is not None) and os.path.exists(saveFiles[0]) and
       os.path.exists(saveFiles[1])):
        distancedict, PDBresidueList, ResidueDict, residuedictionary = pickle.load(
            open(saveFile, 'rb'))
        sortedPDBDist = np.load(saveFiles[1] + '.npz')['pdbDists']
    else:
        residuedictionary = {}
        PDBresidueList = []
        ResidueDict = {}
        prevRes = None
        for selectline in pdbData:
            resname = selectline[0]
            resnumdict = int(selectline[1])
            resatomlisttemp = np.asarray([float(selectline[2]), float(selectline[3]),
                                          float(selectline[4])])
            try:
                residuedictionary[resnumdict].append(resatomlisttemp)
            except KeyError:
                if(prevRes):
                    residuedictionary[prevRes] = np.vstack(
                        residuedictionary[prevRes])
                prevRes = resnumdict
                residuedictionary[resnumdict] = [resatomlisttemp]
                PDBresidueList.append(resnumdict)
                ResidueDict[resnumdict] = resname
        residuedictionary[prevRes] = np.vstack(residuedictionary[prevRes])
        # list of sorted residues - necessary for those where res1 is not 1
        PDBresidueList = list(sorted(PDBresidueList))
        sortedPDBDist = []
        # Loop over all residues in the pdb
        for i in range(len(PDBresidueList)):
            # Loop over residues to calculate distance between all residues i
            # and j
            for j in range(i + 1, len(PDBresidueList)):
                matvalue = []
                # Getting the 3d coordinates for every atom in each residue.
                # iterating over all pairs to find all distances
                key1 = PDBresidueList[i]
                key2 = PDBresidueList[j]
                for k in range(residuedictionary[key1].shape[0]):
                    matvalue.append(
                        np.min(np.linalg.norm(residuedictionary[key2] -
                                              residuedictionary[key1][k],
                                              axis=1)))
                # finding the minimum value from the distance array
                # Making dictionary of all min values indexed by the two residue
                # names
                sortedPDBDist.append(np.min(matvalue))
        sortedPDBDist = np.asarray(sortedPDBDist)
        if(saveFiles is not None):
            pickle.dump((PDBresidueList, ResidueDict, residuedictionary),
                        open(saveFiles[0], 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            np.savez(saveFiles[1], pdbDists=sortedPDBDist)
    end = time.time()
    print('Computing the distance matrix based on the PDB file took {} min'.format(
        (end - start) / 60.0))
    return PDBresidueList, ResidueDict, sortedPDBDist


def plotAUC(fpr, tpr, roc_auc, qName, clus, today, cutoff):
    '''
    Plot AUC

    This function plots and saves the AUCROC.  The image will be stored in the
    eps format with dpi=1000 using a name specified by the query name, cutoff,
    clustering constant, and date.

    Parameters:
    fpr: list
        List of false positive rate values.
    tpr: list
        List of true positive rate values.
    roc_auc: float
        Float specifying the calculated AUC for the curve.
    qName: str
        Name of the query protein
    clus: int
        Number of clusters created
    today: date
        The days date
    cutoff: int
        The distance used for proximity cutoff in the PDB structure.
    '''
    start = time.time()
    pl.plot(fpr, tpr, label='(AUC = {0:.2f})'.format(roc_auc))
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    title = 'Ability to predict positive contacts in {}, Cluster = {}'.format(
        qName, clus)
    pl.title(title)
    pl.legend(loc="lower right")
    imagename = '{}{}A_C{}_{}roc.eps'.format(
        qName, cutoff, clus, today)
    pl.savefig(imagename, format='eps', dpi=1000, fontsize=8)
    pl.close()
    end = time.time()
    print('Plotting the AUC plot took {} min'.format((end - start) / 60.0))


def writeOutClusteringResults(today, qName, clus, scorePositions,
                              etmiplistCoverage, sortedPDBDist, PDBresidueList,
                              residues_dict):
    '''
    Write out clustering results

    This method writes the results of the clustering to file.

    Parameters:
    today: date
        Todays date.
    qName: str
        The name of the query protein
    clus: int
        The number of clusters created
    scorePositions: list
        A list of the order in which the sequence distances are presented.
        The element format is {}_{} where each {} is the number of a sequence
        in the alignment.
    etmiplistCoverage: list
        The coverage of a specific sequence comparison
    sortedPDBDist: numpy nd array
        Array of the distances between sequences, sorted by sequence indices.
    '''
    start = time.time()
    e = "{}_{}_{}.etmipCVG.clustered.txt".format(today, qName, clus)
    etmipoutfile = open(e, "w+")
    counter = 0
    for i in range(0, len(PDBresidueList)):
        for j in range(i + 1, len(PDBresidueList)):
            res1 = PDBresidueList[i]
            res2 = PDBresidueList[j]
            key = '{}_{}'.format(res1, res2)
            if sortedPDBDist[counter] <= cutoff:
                r = 1
            else:
                r = 0
            ind = scorePositions.index(key)
            etmipoutputline = '{} ({}) {} ({}) {} {} {} {}'.format(
                res1, residues_dict[res1], res2, residues_dict[res2],
                round(etmiplistCoverage[ind], 2),
                round(sortedPDBDist[counter], 2), r, clus)
            etmipoutfile.write(etmipoutputline)
            etmipoutfile.write("\n")
            counter += 1
    etmipoutfile.close()
    end = time.time()
    print('Writing the ETMIP worker data to file took {} min'.format(
        (end - start) / 60.0))


# def etMIPWorker(today, qName, cutoff, aa_dict, clus, X, fixed_alignment_dict,
#                 sorted_PDB_dist, distDict):
#     '''
#     ETMIP Worker
#
#     Performs the repeated portion of analysis in this workflow.
#
#     Parameters:
#     today: date
#         Todays date, used for filenames
#     qName: str
#         The name of the query protein
#     cutoff: int
#         The distance for cut off in a PDB structure to consider two atoms as
#         interacting.
#     aa_dict: dict
#         Dictionary mapping amino acid single letter code to a numerical value
#     clus: int
#         Number of clusters to create
#     X: numpy nd_array
#         The pairwise distance between a set of sequences in an alignment
#     fixed_alignment_dict: dict
#         Dictionary of sequences in the alignment with all gaps removed from the
#         query sequences.
#     summed_Matrix: numpy nd_array
#         Tracks the ETMIP score over iterations of clustering
#     sorted_PDB_dist: dict
#         Dictionary of sorted pairwise atom interactions
#     distDict: dict
#         Dictionary of distances between pairs of sequences in the alignment
#     '''
#     start = time.time()
#     print "Starting clustering: K={}".format(clus)
#     cluster_dict, clusterset = AggClustering(clus, X, fixed_alignment_dict)
#     size = np.sqrt(len(distDict))
#     res = np.zeros((size, size))
#     for c in clusterset:
#         new_alignment = {}
#         for key in cluster_dict[c]:
#             new_alignment[key] = fixed_alignment_dict[key]
#         clusteredMIP_matrix = wholeAnalysis(new_alignment, aa_dict)
#         res += clusteredMIP_matrix
#
#     # this is where we can do i, j by running a second loop
#     scorePositions = []
#     etmipResScoreList = []
#     for i in range(0, len(sorted_res_list)):
#         for j in range(i + 1, len(sorted_res_list)):
#             newkey1 = "{}_{}".format(sorted_res_list[i], sorted_res_list[j])
#             scorePositions.append(newkey1)
#             etmipResScoreList.append(summed_Matrix[i][j])
#     etmipResScoreList = np.asarray(etmipResScoreList)
#
#     # Converting to coverage
#     etmiplistCoverage = []
#     numPos = float(len(etmipResScoreList))
#     for i in range(len(etmipResScoreList)):
#         computeCoverage = (((np.sum((etmipResScoreList[i] >= etmipResScoreList)
#                                     * 1.0) - 1) * 100) / numPos)
#         etmiplistCoverage.append(computeCoverage)
#
#     # AUC computation
#     if len(etmiplistCoverage) != len(sorted_PDB_dist):
#         print "lengths do not match"
#         sys.exit()
#     y_true1 = ((sorted_PDB_dist <= cutoff) * 1)
#
#     writeOutClusteringResults(today, qName, clus, scorePositions,
#                               etmiplistCoverage, distdict)
#     fpr1, tpr1, _thresholds = roc_curve(y_true1, etmiplistCoverage,
#                                         pos_label=1)
#     roc_auc1 = auc(fpr1, tpr1)
#     plotAUC(fpr1, tpr1, roc_auc1, qName, clus, today, cutoff)
#     # print "Area under the ROC curve : %f" % roc_auc1, sys.argv[1]
#     time_elapsed = (time.time() - start)
#     output = "\t{0}\t{1}\t{2}\n".format(
#         clus, round(roc_auc1, 2), round(time_elapsed, 2))
#     print('ETMIP worker took {} min'.format(time_elapsed / 60.0))
#     return (clus, output, res)


def etMIPWorker2(inTup):
    '''
    ETMIP Worker

    Performs the repeated portion of analysis in this workflow.

    Parameters:
    today: date
        Todays date, used for filenames
    qName: str
        The name of the query protein
    cutoff: int
        The distance for cut off in a PDB structure to consider two atoms as
        interacting.
    aa_dict: dict
        Dictionary mapping amino acid single letter code to a numerical value
    clus: int
        Number of clusters to create
    X: numpy nd_array
        The pairwise distance between a set of sequences in an alignment
    fixed_alignment_dict: dict
        Dictionary of sequences in the alignment with all gaps removed from the
        query sequences.
    summed_Matrix: numpy nd_array
        Tracks the ETMIP score over iterations of clustering
    sorted_PDB_dist: dict
        Dictionary of sorted pairwise atom interactions
    '''
    today, qName, cutoff, aa_dict, clus, X, sortedKeys, fixed_alignment_dict, sortedPDBDist, residues_dict, PDBresidueList = inTup
    print('IN THREAD!')
    cTested = []
    outputs = []
    resMatrix = None
    while(not cQueue.empty()):
        clus = cQueue.get_nowait()
        cTested.append(clus)
        start = time.time()
        print "Starting clustering: K={}".format(clus)
        cluster_dict, clusterset = aggClustering(clus, X, sortedKeys)
        for c in clusterset:
            new_alignment = {}
            for key in cluster_dict[c]:
                new_alignment[key] = fixed_alignment_dict[key]
            clusteredMIP_matrix = wholeAnalysis(new_alignment, aa_dict)
            if(resMatrix is None):
                resMatrix = clusteredMIP_matrix
            else:
                resMatrix += clusteredMIP_matrix

        # this is where we can do i, j by running a second loop
        scorePositions = []
        etmipResScoreList = []
        for i in range(0, len(PDBresidueList)):
            for j in range(i + 1, len(PDBresidueList)):
                newkey1 = "{}_{}".format(
                    PDBresidueList[i], PDBresidueList[j])
                scorePositions.append(newkey1)
                etmipResScoreList.append(summed_Matrix[i][j])
        etmipResScoreList = np.asarray(etmipResScoreList)

        # Converting to coverage
        etmiplistCoverage = []
        numPos = float(len(etmipResScoreList))
        for i in range(len(etmipResScoreList)):
            computeCoverage = (((np.sum((etmipResScoreList[i] >= etmipResScoreList)
                                        * 1.0) - 1) * 100) / numPos)
            etmiplistCoverage.append(computeCoverage)

        # AUC computation
        if len(etmiplistCoverage) != len(sortedPDBDist):
            print "lengths do not match"
            sys.exit()
        y_true1 = ((sortedPDBDist <= cutoff) * 1)

        writeOutClusteringResults(today, qName, clus, scorePositions,
                                  etmiplistCoverage, sortedPDBDist,
                                  PDBresidueList, residues_dict)
        fpr1, tpr1, _thresholds = roc_curve(y_true1, etmiplistCoverage,
                                            pos_label=1)
        roc_auc1 = auc(fpr1, tpr1)
        plotAUC(fpr1, tpr1, roc_auc1, qName, clus, today, cutoff)
        # print "Area under the ROC curve : %f" % roc_auc1, sys.argv[1]
        time_elapsed = (time.time() - start)
        output = "\t{0}\t{1}\t{2}\n".format(
            clus, round(roc_auc1, 2), round(time_elapsed, 2))
        outputs.append(output)
        print('ETMIP worker took {} min'.format(time_elapsed / 60.0))
    return (cTested, outputs, resMatrix)


def writeFinalResults(qName, today, sequence_order, seq_length, cutoff, outDict):
    o = '{}_{}etmipAUC_results.txt'.format(qName, today)
    outfile = open(o, 'w+')
    proteininfo = ("Protein/id: " + qName + " Alignment Size: " +
                   str(len(sequence_order)) + " Length of protein: " +
                   str(seq_length) + " Cutoff: " + str(cutoff) + "\n")
    outfile.write(proteininfo)
    outfile.write("#OfClusters\tAUC\tRunTime\n")
    for key in sorted(outDict.keys()):
        outfile.write(outDict[key])


####--------------------------------------------------------#####
    ### BODY OF CODE ##
####--------------------------------------------------------#####
if __name__ == '__main__':
    start = time.time()
    ###########################################################################
    # Set up global variables
    ###########################################################################
    today = str(datetime.date.today())
    neighbor_list = []
    gap_list = ["-", ".", "_"]
    aa_list = []
    # comment out for actual dataset
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    aa_gap_list = aa_list + gap_list
    i_j_list = []
    # {seq1_seq2} = distancescore
    seq12_distscore_dict = {}
    key = ''
    temp_aa = ''
    ###########################################################################
    # Set up input variables
    ###########################################################################
    files = open(sys.argv[1], 'rb')
    pdbfilename = open(sys.argv[2], 'rb')
    cutoff = float(sys.argv[3])
    qName = str(sys.argv[4])
    try:
        outDir = sys.argv[5]
    except:
        outDir = "/cedar/atri/projects/coupling/OutputsforETMIP_BA/"
    try:
        processes = int(sys.argv[6])
        pCount = cpu_count()
        if(processes > pCount):
            processes = pCount
    except:
        processes = 1
    ###########################################################################
    # Set up output location
    ###########################################################################
    startDir = os.getcwd()
    createFolder = (outDir + str(today) + "/" + qName)
    if not os.path.exists(createFolder):
        os.makedirs(createFolder)
        print "creating new folder"
    os.chdir(createFolder)

    print 'Starting ETMIP'
    # Import alignment information: this will be our alignment
    alignment_dict = importAlignment(files, 'alignment_dict.pkl')
    # Remove gaps from aligned query sequences
    query_name, fixed_alignment_dict = removeGaps(alignment_dict,
                                                  'ungapped_alignment.pkl')
    # I will get a corr_dict for method x for all residue pairs FOR ONE PROTEIN
    X, sequence_order = distanceMatrix(fixed_alignment_dict,
                                       ('X', 'seq_order.pkl'))
    # Generate MIP Matrix
    wholeMIP_Matrix = wholeAnalysis(fixed_alignment_dict, aa_dict, 'wholeMIP')
    ###########################################################################
    # Set up for remaining analysis
    ###########################################################################
    seq_length = len(fixed_alignment_dict[fixed_alignment_dict.keys()[0]])
    summed_Matrix = np.zeros((seq_length, seq_length))
    summed_Matrix += wholeMIP_Matrix
    pdbData = importPDB(pdbfilename, 'pdbData.pkl')
    PDBresidueList, residues_dict, sortedPDBDist = findDistance(
        pdbData, ('PDBdist.pkl', 'PDBdistances'))
    ###########################################################################
    # Perform multiprocessing of clustering method
    ###########################################################################
    manager = Manager()
    aa_dict = manager.dict(aa_dict)
    fixed_alignment_dict = manager.dict(fixed_alignment_dict)
    cQueue = manager.Queue()
    sequence_order = manager.list(sequence_order)
    for clus in [2, 3, 5, 7, 10, 25]:
        cQueue.put(clus)
    # etMIPWorker2((today, qName, cutoff, aa_dict, cQueue, X,
    #              sequence_order, fixed_alignment_dict, sortedPDBDist,
    #              residues_dict, PDBresidueList))
    # exit()
    pool = Pool(processes=processes)
    res = pool.map_async(etMIPWorker2,
                         [(today, qName, cutoff, aa_dict, cQueue, X,
                           sequence_order, fixed_alignment_dict, sortedPDBDist,
                           residues_dict, PDBresidueList)] * processes)
    pool.close()
    pool.join()
    res = res.get()
    outDict = {}
    for r in res:
        for i in range(len(r[0])):
            outDict[r[0][i]] = r[1][i]
        print type(r[2])
        try:
            summed_Matrix += r[2]
        except:
            print r
            exit()
    writeFinalResults(qName, today, sequence_order,
                      seq_length, cutoff, outDict)
    print "Generated results in", createFolder
    os.chdir(startDir)
    end = time.time()
    print('ET MIP took {} minutes to run!'.format((end - start) / 60.0))
