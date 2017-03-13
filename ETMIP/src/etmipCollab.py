'''
Created on Mar 10, 2017

@author: Benu Atri
'''

from sklearn.metrics import roc_curve, auc, mutual_info_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib
from matplotlib.sphinxext.plot_directive import align
from numpy import float64
matplotlib.use('Agg')
import numpy as np
import pylab as pl
import datetime
import time
import math
import sys
import re
import os


def importAlignment(files, alignment_dict):
    '''
    Import alignments:

    This method imports alignments into an existing dictionary.

    Parameters:
    -----------
    files: File
        File object holding a handle to an alignment file.
    alignment_dict: dict    
        Dictionary which will be used to store alignments from the file.
    '''
    for line in files:
        if line.startswith(">"):
            if "query" in line.lower():
                query_desc = line
            key = line.rstrip()
            alignment_dict[key] = ''
        else:
            alignment_dict[key] += line.rstrip()


def remove_gaps(alignment_dict):
    '''
    Remove Gaps

    Removes all gaps from the query sequence and removes characters at the
    corresponding positions in all other sequences.

    Parameters:
    -----------
    alignment_dict: dict
        Dictionary mapping query name to sequence.

    Returns:
    --------
    str
        A new query name.
    dict
        A transform of the input dictionary without gaps.
    '''
    # Getting gapped columns for query
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
        return query_name, new_alignment_dict
    else:
        return query_name, alignment_dict


def distance_matrix(alignment_dict):
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
    Returns:
    --------
    matrix
        A symmetric matrix of pairwise distance computed between two sequences
        using the sequence identity metric.
    list
        List of the sequence identifiers in the order in which they appear in
        the matrix.
    '''
    # Generate distance_matrix: Calculating Sequence Identity
    key_list = alignment_dict.keys()
    valuematrix = np.zeros([len(alignment_dict), len(alignment_dict)])
    for i in range(len(alignment_dict)):
        for j in range(i + 1, len(alignment_dict)):
            for idc, char in enumerate(alignment_dict[key_list[i]]):
                if (alignment_dict[key_list[i]][idc] ==
                        alignment_dict[key_list[j]][idc]):
                    valuematrix[i, j] += 1.0
                    valuematrix[j, i] += 1.0
    valuematrix /= len(alignment_dict[key_list[0]])
    return valuematrix, key_list


def wholeAnalysis(alignment, aa_list):
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
    Returns:
    --------
    matrix
        Matrix of MIP scores which has dimensions seq_length by seq_length.
    '''
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
            alignment2Num[i, j] = aa_list.index(alignment[key_list[i]][j])
    # Generate MI matrix from alignment2Num matrix, the MMI matrix,
    # and overallMMI
    for i in range(seq_length):
        for j in range(i):
            column_i = alignment2Num[:, i]
            column_j = alignment2Num[:, j]
            currMIS = mutual_info_score(column_i, column_j, contingency=None)
            MI_matrix[i][j] = currMIS
            # AW: divides by individual entropies to normalize.
            MI_matrix[j][i] = currMIS
            overallMMI += currMIS
    MMI += np.sum(MI_matrix, axis=1)
    MMI -= MI_matrix[np.arange(seq_length), np.arange(seq_length)]
    MMI /= (seq_length - 1)
    overallMMI = 2.0 * (overallMMI / (seq_length - 1)) / seq_length
    ####--------------------------------------------#####
    # Calculating APC
    ####--------------------------------------------#####
    APC_matrix += np.outer(MMI, MMI)
    APC_matrix[np.arange(seq_length), np.arange(seq_length)] = 0
    APC_matrix /= overallMMI
    # Defining MIP matrix
    MIP_matrix += MI_matrix - APC_matrix
    MIP_matrix[np.arange(seq_length), np.arange(seq_length)] = 0
    return MIP_matrix


def find_distance(filename):  # takes PDB
    '''
    Find distance

    Desc

    Parameters:
    -----------
    filename: string
        The file path to the pdb file.
    Returns:
    dict
        Dictionary of distances
    list
        List of residues from the PDB file.
    dict
        Dictionary of residue.
    --------
    '''
    # This code takes in an input of a pdb file and outputs a dictionary with
    # the nearest atom distance between two residues
    ##########################################################################
    ##########################################################################
    minvalue = 10000000000
    FileValue = 0
    originlist = []
    file = open(filename)
    rows = []
    loopcounter = 0
    Resnumarraynew = []
    loopcounter1 = 0
    for line in file:  # for a line in the pdb
        if line.startswith('ATOM '):
            try:
                rows.append(line)
            except Exception:
                rows = line
    file.close()
    pdbPattern = r'(ATOM)\s*(\d+)\s*(\w*)\s*([A-Z]{3})\s*([A-Z])\s*(\d+)\s*(\d+\.\d+)\s*(-\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*([A-Z])'
    # number of residues
    loop1var = re.match(pdbPattern, rows[-1]).group(6)
    residuedictionary = {}

    # create dictionary of every atom in each individual residue. 3
    # Dimensional coordinates of each residue position
    resatomlist = []
    PDBresidueList = []
    ResidueDict = {}
    for i, selectline in enumerate(rows):
        items = re.match(pdbPattern, selectline)
        if(not items):
            continue
        resname = items.group(4)
        resnumdict = items.group(6)
        resatomlisttemp = np.asarray(items.group(7), items.group(8),
                                     items.group(9))
        try:
            residuedictionary[resnumdict].append(resatomlisttemp)
        except KeyError:
            residuedictionary[resnumdict] = [resatomlisttemp]
            PDBresidueList.append(resnumdict)
            ResidueDict[resnumdict] = resname
    '''Loops for comparing one residues atoms to a second list of atoms in seperate residue'''
    '''print(residuedictionary)'''
    arrminval = []

    distancedict = {}
    for i in PDBresidueList:  # Loop over all residues in the pdb
        Dmatrixsumarr = []
        resnumnew = PDBresidueList[int(i):]
        # Loop over residues to calculate distance between all residues i and j
        for j in PDBresidueList:
            matvalue = []
            tempvalue = ()
            minvaluetemp = []
            loopcount1 = 0

            for k in range(0, len(residuedictionary[i])):
                # Getting the 3d coordinates for every atom in each residue.
                # iterating over all pairs to find all distances
                for m in range(0, len(residuedictionary[j])):
                    '''print("k equals", k)
                    print("m equals", m)'''
                    tempvalue = np.linalg.norm(
                        residuedictionary[i][k] - residuedictionary[j][m],
                        dtype=float64)
                    '''print("tempvalue equals", tempvalue)'''
                    try:  # Saving all distances to an array
                        matvalue.append(float(tempvalue))
                    except Exception:
                        matvalue = [float(tempvalue), ]
                    '''print("matvalue equals", matvalue)'''
                loopcount1 += 1
                '''print("loopcount1 equals", loopcount1)'''
            # finding the minimum value from the distance array
            minvalue = float(min(matvalue))
            '''print("size of ETvalues is equal to", len(ETvalues))'''
            key = str(i) + '_' + str(j)
            # Making dictionary of all min values indexed by the two residue
            # names
            distancedict[key] = minvalue
    return distancedict, PDBresidueList, ResidueDict


def AggClustering(n_cluster, X, alignment_dict):
    key_list = []
    cluster_dict = {}
    linkage = 'ward'
    model = AgglomerativeClustering(linkage=linkage, n_clusters=n_cluster)
    model.fit(X)
    clusterlist = model.labels_.tolist()  # ordered list of cluster ids
    # unique and sorted cluster ids for e.g. for n_cluster = 2, g = [0,1]
    g = set(clusterlist)

    for key in alignment_dict:
        key_list.append(key)

    ####---------------------------------------#####
    #       Mapping Clusters to Sequences
    ####---------------------------------------#####
    for i1 in g:
        clusteredkeylist = []
        for i in range(0, len(clusterlist)):
            if clusterlist[i] == i1:
                # list of keys in a given cluster
                clusteredkeylist.append(key_list[i])
        # cluster_dict[0 or 1] = [list of keys]
        cluster_dict[i1] = clusteredkeylist
    return cluster_dict, g

####--------------------------------------------------------#####
    ### BODY OF CODE ##
####--------------------------------------------------------#####
if __name__ == '__main__':
    ###########################################################################
    # Set up global variables
    ###########################################################################
    today = datetime.date.today()
    neighbor_list = []
    gap_list = ["-", ".", "_"]
    aa_list = []
    # comment out for actual dataset
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aa_gap_list = aa_list + gap_list
    i_j_list = []
    # this will be our alignment
    alignment_dict = {}
    # {seq1_seq2} = distancescore
    seq12_distscore_dict = {}
    key = ''
    temp_aa = ''

    ###########################################################################
    # Set up input variables
    ###########################################################################
    files = open(sys.argv[1], "r")  # provide complete path to fasta alignment
    pdbfilename = sys.argv[2].strip()
    cutoff = float(sys.argv[3])
    qName = str(sys.argv[4])
    try:
        outDir = sys.argv[5]
    except:
        outDir = "/cedar/atri/projects/coupling/OutputsforETMIP_BA/"

    ###########################################################################
    # Set up output location
    ###########################################################################
    createFolder = (outDir + str(today) + "/" + qName)
    if not os.path.exists(createFolder):
        os.makedirs(createFolder)
        print "creating new folder"

    print 'Starting ETMIP'
    # Import alignment information
    importAlignment(files, alignment_dict)
    print 'Imported alignment'
    # Remove gaps from aligned query sequences
    query_name, fixed_alignment_dict = remove_gaps(alignment_dict)
    print 'Removed gaps'
    # I will get a corr_dict for method x for all residue pairs FOR ONE PROTEIN
    X, sequence_order = distance_matrix(fixed_alignment_dict)
    print 'Computed Distance Matrix'
    # Generate MIP Matrix
    wholeMIP_Matrix = wholeAnalysis(fixed_alignment_dict, aa_list)
    print 'Computed MIP Matrix'
    ###########################################################################
    # Set up for remaining analysis
    ###########################################################################
    seq_length = len(fixed_alignment_dict[fixed_alignment_dict.keys()[0]])
    summed_Matrix = np.zeros((seq_length, seq_length))
    summed_Matrix = wholeMIP_Matrix
    time_start = time.clock()
    o = '{}{}/{}/{}_{}etmipAUC_results.txt'.format(outDir, today, qName, qName,
                                                   today)
    outfile = open(o, 'w+')
    proteininfo = ("Protein/id: " + qName + " Alignment Size: " +
                   str(len(sequence_order)) + " Length of protein: " +
                   str(seq_length) + " Cutoff: " + str(cutoff) + "\n")
    outfile.write(proteininfo)
    outfile.write("#OfClusters\tAUC\tRunTime\n")
    # e.g. 206_192 6.82
    distdict, PDBresidueList, residues_dict = find_distance(pdbfilename)
    time_stop = time.clock()
    print(time_stop - time_start)
    exit()
    #
    #
    #

    PDBdist_Classifylist = []
    sorted_PDB_dist = []
    sorted_res_list = []

    for i in PDBresidueList:
        sorted_res_list.append(int(i))
    # list of sorted residues - necessary for those where res1 is not 1
    sorted(list(set(sorted_res_list)))
    # this is where we can do i, j by running a second loop
    for i in sorted_res_list:
        for j in sorted_res_list:
            if i >= j:
                continue
            newkey1 = str(i) + "_" + str(j)
            sorted_PDB_dist.append(distdict[newkey1])

    # NAME, ALIGNMENT SIZE, PROTEIN LENGTH
    print qName, len(sequence_order), str(seq_length)

    ls = [2, 3, 5, 7, 10, 25]
    for clus in ls:
        # print "starting clustering"
        e = outDir + str(today) + "/" + str(
            qName) + "/" + qName + "_" + str(clus) + "_" + str(today) + ".etmipCVG.clustered.txt"
        etmipoutfile = open("{0}".format(e), "w+")
        # setoffiles.append(e)
        cluster_dict, clusterset = AggClustering(clus, X, fixed_alignment_dict)
        for c in clusterset:
            new_alignment = {}
            cluster_list = cluster_dict[c]
            for key in fixed_alignment_dict:
                if key in cluster_list:
                    new_alignment[key] = fixed_alignment_dict[key]
            clusteredMIP_matrix = wholeAnalysis(new_alignment)
            summed_Matrix = np.add(summed_Matrix, clusteredMIP_matrix)

        etmiplist = []
        etmip_dict = {}
        PDBdist_Classifylist = []
        y_score1 = []
        y_true1 = []

        # this is where we can do i, j by running a second loop
        for i in range(0, len(sorted_res_list)):
            for j in range(0, len(sorted_res_list)):
                if i >= j:
                    continue
                key = str(sorted_res_list[i]) + "_" + str(sorted_res_list[j])
                etmip_dict[key] = summed_Matrix[i][j]
        etmipResScoreList = []
        forOutputCoverageList = []
        etmiplistCoverage = []

        for i in range(0, len(sorted_res_list)):
            for j in range(0, len(sorted_res_list)):
                if i >= j:
                    continue
                newkey1 = str(
                    sorted_res_list[i]) + "_" + str(sorted_res_list[j])
                etmipResScoreList.append(newkey1)
                etmipResScoreList.append(etmip_dict[newkey1])

        # Converting to coverage

        for i in range(1, len(etmipResScoreList), 2):
            etmipRank = 0
            for j in range(1, len(etmipResScoreList), 2):
                if i != j:
                    if float(etmipResScoreList[i]) >= float(etmipResScoreList[j]):
                        etmipRank += 1
            computeCoverage = (etmipRank * 100) / \
                (float(len(etmipResScoreList)) / 2)
            etmiplistCoverage.append(computeCoverage)

            forOutputCoverageList.append(etmipResScoreList[i - 1])
            forOutputCoverageList.append(computeCoverage)
            # print computeCoverage
        # print  "Coverage computation finished"
        # AUC computation
        if len(etmiplistCoverage) == len(sorted_PDB_dist):
            for i in range(0, len(etmiplistCoverage)):
                y_score1.append(etmiplistCoverage[i])

                if (float(sorted_PDB_dist[i]) <= cutoff):
                    PDBdist_Classifylist.append(1)
                    y_true1.append(1)
                else:
                    PDBdist_Classifylist.append(0)
                    y_true1.append(0)
        else:
            print "lengths do not match"
            sys.exit()
        # print  "AUC computation finished"

        # this is where we can do i, j by running a second loop
        for i in range(0, len(sorted_res_list)):
            # this is where we can do i, j by running a second loop
            for j in range(0, len(sorted_res_list)):
                if i >= j:
                    continue
                else:
                    key = str(sorted_res_list[i]) + \
                        "_" + str(sorted_res_list[j])
                    if distdict[key] <= cutoff:
                        r = 1
                    else:
                        r = 0
                    res1 = str(sorted_res_list[i])
                    res2 = str(sorted_res_list[j])
                    ind = forOutputCoverageList.index(key)
                    etmipoutputline = res1 + " (" + residues_dict[res1] + ") " + res2 + " (" + residues_dict[res2] + ") " + str(
                        round(forOutputCoverageList[ind + 1], 2)) + " " + str(round(distdict[key], 2)) + " " + str(r) + " " + str(clus)
                    etmipoutfile.write(etmipoutputline)
                    etmipoutfile.write("\n")
        fpr1, tpr1, thresholds1 = roc_curve(y_true1, y_score1, pos_label=1)
        roc_auc1 = auc(fpr1, tpr1)
        # print "Area under the ROC curve : %f" % roc_auc1, sys.argv[1]
        time_elapsed = (time.clock() - time_start)
        output = "\t{0}\t{1}\t{2}\n".format(
            str(clus), round(roc_auc1, 2), round(time_elapsed, 2))
        outfile.write(output)

        pl.clf()
        pl.plot(fpr1, tpr1, label='(AUC = %0.2f)' % roc_auc1)  # change here
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        title = 'Ability to predict positive contacts in ' + \
            qName + ", Cluster = " + str(clus)
        pl.title(title)
        pl.legend(loc="lower right")
        # pl.show()
        imagename = outDir + str(today) + "/" + qName + "/" + str(
            sys.argv[4]) + str(int(cutoff)) + "A_C" + str(clus) + "_" + str(today) + "roc.eps"  # change here
        pl.savefig(imagename, format='eps', dpi=1000, fontsize=8)
    print "Generated results in", createFolder
