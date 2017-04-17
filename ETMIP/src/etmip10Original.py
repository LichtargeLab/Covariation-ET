import numpy as np
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from subprocess import call
from sklearn.cluster import AgglomerativeClustering
import argparse
#from Bio.PDB import *
import operator
import sys
import time
from subprocess import Popen, PIPE
from sys import stdin
from math import exp
from collections import Counter
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, mutual_info_score, normalized_mutual_info_score
from operator import itemgetter
import matplotlib.pyplot as plt
from pylab import *
import datetime
today = datetime.date.today()
neighbor_list = []
gap_list = ["-", ".", "_"]
aa_list = []
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
           'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']  # comment out for actual dataset
aa_gap_list = aa_list + gap_list
i_j_list = []
alignment_dict = {}  # this will be our alignment
seq12_distscore_dict = {}  # {seq1_seq2} = distancescore
key = ''
temp_aa = ''
cutoff = float(sys.argv[3])


def remove_gaps(alignment_dict):
    # Getting gapped columns for query
    gap = ['-', '.', '_']
    query_gap_index = []
    for key, _value in alignment_dict.iteritems():
        if "query" in key.lower():
            query_name = key
            for idc, char in enumerate(alignment_dict[key]):
                if char in gap:
                    query_gap_index.append(idc)

    new_alignment_dict = {}
    for key, _value in alignment_dict.iteritems():
        new_alignment_dict[key] = ''
        for idc, char in enumerate(alignment_dict[key]):
            if idc in query_gap_index:
                continue
            else:
                new_alignment_dict[key] = new_alignment_dict[key] + char

    return query_name, new_alignment_dict


def distance_matrix(alignment_dict):
    #	Generate distance_matrix: Calculating Sequence Identity
    key_list = []
    pairwise_dist_score = {}
    seq_length = len(alignment_dict.itervalues().next())

    for key in alignment_dict:  # {seqid} = sequence
        for key2 in alignment_dict:
            sum = 0.0
            if key > key2:
                for idc, char in enumerate(alignment_dict[key]):

                    if (alignment_dict[key][idc] == alignment_dict[key2][idc]):
                        # print alignment_dict[key][idc],
                        # alignment_dict[key2][idc]
                        sum += 1.0
                newkey = key + "_" + key2
                # (# of identical positions) / (aligned positions)
                seq_identity = (sum / seq_length)
                # {seq1_seq2} = distancescore
                pairwise_dist_score[newkey] = seq_identity

    valuematrix = np.zeros([len(alignment_dict), len(alignment_dict)])

    for key in alignment_dict:
        key_list.append(key)

    for key in key_list:
        for key2 in key_list:
            if key == key2:
                continue
            pair = key + "_" + key2
            if pair in pairwise_dist_score:
                valuematrix[key_list.index(key), key_list.index(
                    key2)] = pairwise_dist_score[pair]
                valuematrix[key_list.index(key2), key_list.index(
                    key)] = pairwise_dist_score[pair]
    return valuematrix, key_list


def wholeAnalysis(alignment):
    overallMMI = 0.0
    seq_length = len(alignment.itervalues().next())
    # generate a MI matrix for each cluster
    MI_matrix = np.zeros([seq_length, seq_length])
    MMI = np.zeros([seq_length, 1])  # Vector of 1 column
    APC_matrix = np.zeros([seq_length, seq_length])
    MIP_matrix = np.zeros([seq_length, seq_length])

    alignment2Num = []

    for key in alignment:
        seq2Num = []
        for _idc, c in enumerate(alignment[key]):
            seq2Num.append(aa_list.index(c))
        alignment2Num.append(seq2Num)

    for i in range(0, seq_length):
        MMI[i][0] = 0.0
        column_i = []
        column_j = []
        for j in range(0, seq_length):
            if i >= j:
                continue
            column_i = [int(item[i]) for item in alignment2Num]
            column_j = [int(item[j]) for item in alignment2Num]
            MI_matrix[i][j] = mutual_info_score(
                column_i, column_j, contingency=None)
            # AW: divides by individual entropies to normalize.
            MI_matrix[j][i] = MI_matrix[i][j]

    for i in range(0, seq_length):  # this is where we can do i, j by running a second loop
        for j in range(0, seq_length):
            if i != j:
                MMI[i][0] += MI_matrix[i][j]
                if i > j:
                    overallMMI += MI_matrix[i][j]
        MMI[i][0] = MMI[i][0] / (seq_length - 1)

    overallMMI = 2.0 * (overallMMI / (seq_length - 1)) / seq_length
    ####--------------------------------------------#####
    # Calculating APC
    ####--------------------------------------------#####
    for i in range(0, seq_length):
        for j in range(0, seq_length):
            if i == j:
                continue
            APC_matrix[i][j] = (MMI[i][0] * MMI[j][0]) / overallMMI

    for i in range(0, seq_length):
        for j in range(0, seq_length):
            if i == j:
                continue
            MIP_matrix[i][j] = MI_matrix[i][j] - APC_matrix[i][j]
    return MIP_matrix


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


def find_distance(filename):  # takes PDB

    # This code takes in an input of a pdb file and outputs a dictionary with the nearest atom distance between two residues
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
        if line[0:5] == 'ATOM ':
            try:
                rows.append(line)
            except Exception:
                rows = line
    file.close()
    loop1var = rows[-1][23:26].strip()  # number of residues
    # print("loop1var",loop1var)
    # raw_input()
    residuedictionary = {}

    # create dictionary of every atom in each individual residue. 3
    # Dimensional coordinates of each residue position
    resatomlist = []
    PDBresidueList = []
    ResidueDict = {}
    for i, selectline in enumerate(rows):

        resnumdict = (selectline[22:26].strip())
        resname = (selectline[17:20].strip())
        xvaluedict = float(selectline[31:38].strip())
        yvaluedict = float(selectline[39:46].strip())
        zvaluedict = float(selectline[47:55].strip())
        resatomlisttemp = (xvaluedict, yvaluedict, zvaluedict)
        try:
            residuedictionary[resnumdict].append(resatomlisttemp)
        except KeyError:
            residuedictionary[resnumdict] = [resatomlisttemp]
            PDBresidueList.append(resnumdict)
            ResidueDict[resnumdict] = resname
    # print PDBresidueList
    # print("residuedictionary",ResidueDict)
    # raw_input()
    '''Loops for comparing one residues atoms to a second list of atoms in seperate residue'''
    '''print(residuedictionary)'''
    arrminval = []

    distancedict = {}
    for i in PDBresidueList:  # Loop over all residues in the pdb
        Dmatrixsumarr = []
        resnumnew = PDBresidueList[int(i):]
        for j in PDBresidueList:  # Loop over residues to calculate distance between all residues i and j
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
                    rix = residuedictionary[i][k][0]
                    riy = residuedictionary[i][k][1]
                    riz = residuedictionary[i][k][2]
                    rjx = residuedictionary[j][m][0]
                    rjy = residuedictionary[j][m][1]
                    rjz = residuedictionary[j][m][2]
                    tempvalue = float(
                        math.sqrt((rix - rjx)**2 + (riy - rjy)**2 + (riz - rjz)**2))
                    '''print("tempvalue equals", tempvalue)'''
                    try:  # Saving all distances to an array
                        matvalue.append(float(tempvalue))
                    except Exception:
                        matvalue = [float(tempvalue), ]
                    '''print("matvalue equals", matvalue)'''
                loopcount1 = loopcount1 + 1
                '''print("loopcount1 equals", loopcount1)'''
            minvalue = float(
                min(matvalue))  # finding the minimum value from the distance array
            '''print("size of ETvalues is equal to", len(ETvalues))'''
            key = str(i) + '_' + str(j)
            # Making dictionary of all min values indexed by the two residue
            # names
            distancedict[key] = minvalue
    # print distancedict
    return distancedict, PDBresidueList, ResidueDict


####--------------------------------------------------------#####
    ### BODY OF CODE ##
####--------------------------------------------------------#####
files = open(sys.argv[1], "r")  # provide complete path to fasta alignment
for line in files:
    if line.startswith(">"):
        if "query" in line.lower():
            query_desc = line
        key = line.rstrip()
        alignment_dict[key] = ''
    else:
        alignment_dict[key] = alignment_dict[key] + line.rstrip()
createFolder = ("../Output/" +
                str(today) + "/" + str(sys.argv[4]))

if not os.path.exists(createFolder):
    os.makedirs(createFolder)
    print "creating new folder"


query_name, fixed_alignment_dict = remove_gaps(alignment_dict)
# I will get a corr_dict for method x for all residue pairs FOR ONE PROTEIN
X, sequence_order = distance_matrix(fixed_alignment_dict)
wholeMIP_Matrix = wholeAnalysis(fixed_alignment_dict)
seq_length = len(fixed_alignment_dict.itervalues().next())
summed_Matrix = np.zeros([seq_length, seq_length])
summed_Matrix = wholeMIP_Matrix
pdbfilename = sys.argv[2].strip()

time_start = time.clock()

o = "../Output/" + str(today) + "/" + str(
    sys.argv[4]) + "/" + str(sys.argv[4]) + "_" + str(today) + "etmipAUC_results.txt"
outfile = open(o, 'w+')
proteininfo = ("Protein/id: " + str(sys.argv[4]) + " Alignment Size: " + str(len(
    sequence_order)) + " Length of protein: " + str(seq_length) + " Cutoff: " + str(cutoff) + "\n")
outfile.write(proteininfo)
outfile.write("#OfClusters\tAUC\tRunTime\n")
distdict, PDBresidueList, residues_dict = find_distance(
    pdbfilename)  # e.g. 206_192 6.82
PDBdist_Classifylist = []
sorted_PDB_dist = []
sorted_res_list = []

for i in PDBresidueList:
    sorted_res_list.append(int(i))
# list of sorted residues - necessary for those where res1 is not 1
sorted(list(set(sorted_res_list)))
for i in sorted_res_list:  # this is where we can do i, j by running a second loop
    for j in sorted_res_list:
        if i >= j:
            continue
        newkey1 = str(i) + "_" + str(j)
        sorted_PDB_dist.append(distdict[newkey1])

# NAME, ALIGNMENT SIZE, PROTEIN LENGTH
print str(sys.argv[4]), len(sequence_order), str(seq_length)

ls = [2, 3, 5, 7, 10, 25]
for clus in ls:
    clusterStart = time.time()
    # print "starting clustering"
    e = "../Output/" + str(today) + "/" + str(
        sys.argv[4]) + "/" + str(sys.argv[4]) + "_" + str(clus) + "_" + str(today) + ".etmipCVG.clustered.txt"
    etmipoutfile = open("{0}".format(e), "w+")
#     setoffiles.append(e)
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
            newkey1 = str(sorted_res_list[i]) + "_" + str(sorted_res_list[j])
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
                key = str(sorted_res_list[i]) + "_" + str(sorted_res_list[j])
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
#     time_elapsed = (time.clock() - time_start)
    time_elapsed = (time.time() - clusterStart)
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
        str(sys.argv[4]) + ", Cluster = " + str(clus)
    pl.title(title)
    pl.legend(loc="lower right")
    # pl.show()
    imagename = "../Output/" + str(today) + "/" + str(sys.argv[4]) + "/" + str(
        sys.argv[4]) + str(int(cutoff)) + "A_C" + str(clus) + "_" + str(today) + "roc.eps"  # change here
    pl.savefig(imagename, format='eps', dpi=1000, fontsize=8)
print "Generated results in", createFolder
