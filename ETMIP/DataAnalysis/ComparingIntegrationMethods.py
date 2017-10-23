'''
Created on Oct 4, 2017

@author: daniel
'''
import os
import csv
import pandas
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from seaborn import boxplot, factorplot, barplot


def computeWilcoxonRankSum(df, name):
    fileHandle = open(name + '_TestStatistics.txt', 'wb')
    writer = csv.writer(fileHandle, delimiter='\t')
    writer.writerow(['Metric1', 'Metric2', 'WilcoxonRankSumTestStatistic',
                     'P-Value'])
    for pair in itertools.combinations(df.Method.unique(), 2):
        data1 = df.loc[(df['Method'] == pair[0]), 'AUC']
        data2 = df.loc[(df['Method'] == pair[1]), 'AUC']
        res = ranksums(data1, data2)
        writer.writerow([pair[0], pair[1], res[0], res[1]])
    fileHandle.close()


def plotBoxPlot(df, name, x='Method', y='AUC', hue=None, hue_order=None):
    sns.set_style('whitegrid')
    if(hue):
        boxplot(data=df, x=x, y=y, hue=hue, hue_order=hue_order)
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        boxplot(data=df, x=x, y=y)
    plt.xticks(rotation=90)
    plt.ylim([0.5, 1.0])
    plt.tight_layout()
    if(hue):
        plt.savefig('{}.pdf'.format(name), dpi=150, bbox_inches='tight',
                    bbox_extra_artists=[lgd])
    else:
        plt.savefig('{}.pdf'.format(name), dpi=150, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    ##########################################################################
    # Import data
    ##########################################################################
    os.chdir('/Users/dmkonecki/Desktop/ETMIPC/KAndClusterIntegration/')
#     os.chdir('/media/daniel/ExtraDrive1/Results/ETMIPC/KAndClusterIntegration/')
    df = pandas.read_csv('KAndClusteringIntegrationAUCs.csv',
                         delimiter='\t', header=0)
    sns.set_palette("bright", 8)
    df = df.loc[df['K_Integration'] == 'Sum']
    df = df.loc[df['K'] == 25]
    protein_order = df.sort_values(
        by='AlignmentSize', ascending=True).Protein.unique()
    clus_int_order = df.Cluster_Integration.unique()
    sns.set_style('whitegrid')
    barplot(data=df, hue='Cluster_Integration', x='Protein', y='AUC',
            order=protein_order, hue_order=clus_int_order, ci=None,
            palette=sns.color_palette(["#FF2626", "#2c8c99", "#f28c26",
                                       "#cb48b7", "#1ac8ed", ], 5))
    plt.xticks(rotation=45)
    plt.ylim([0.5, 0.85])
    plt.ylabel('AUC')
    plt.tight_layout()
#     lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    lgd = plt.legend(loc=2)
    for t in lgd.texts:
        if(t.get_text() == 'Sum'):
            continue
        elif(t.get_text() == 'Average'):
            t.set_text('1/# Clusters')
        elif(t.get_text() == 'Size_Weighted'):
            t.set_text('1/|Cluster|')
        elif(t.get_text() == 'Evidence_Weighted'):
            t.set_text('1/Ungapped Seq.')
        elif(t.get_text() == 'Evidence_Vs_Size'):
            t.set_text('Ungapped Seq./|Cluster|')
        else:
            pass
    plt.savefig('KIntSum_K25_ProteinByClusInt.pdf', dpi=150, bbox_inches='tight',
                bbox_extra_artists=[lgd])
    exit()
    ##########################################################################
    # Add a column for methods
    ##########################################################################
    df['Method'] = df[['K_Integration', 'Cluster_Integration']].apply(
        lambda x: '\n'.join(list(set(x))), axis=1)
    df = df.loc[(df['K'] == 25)]
    protein_order = df.sort_values('AlignmentSize').Protein.unique()
    k_int_order = ['sum', 'average']
    clus_int_order = ['sum', 'average', 'size_weighted', 'evidence_weighted',
                      'evidence_vs_size']
    ##########################################################################
    # Compute statistics and plot data across all tested parameters
    ##########################################################################
    computeWilcoxonRankSum(df, 'All')
    plotBoxPlot(df, 'All')
    plotBoxPlot(df, 'All_K_Integration', x='K_Integration')
    plotBoxPlot(df, 'All_Cluster_Integration', x='Cluster_Integration')
    ##########################################################################
    # Compute statistics and plot data by method for K_Integration
    ##########################################################################
    plotBoxPlot(df, 'K_Integration_Hue', x='Cluster_Integration',
                hue='K_Integration')
    f = factorplot(data=df, x='Cluster_Integration', y='AUC',
                   col='K_Integration', kind='box', aspect=1.0)
    f.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig('K_Integration.pdf', dpi=150)
    plt.clf()
    for sub in df.K_Integration.unique():
        currDF = df.loc[(df['K_Integration'] == sub)]
        name = 'K_Integration_{}'.format(sub)
        computeWilcoxonRankSum(currDF, name)
        plotBoxPlot(currDF, name, x='Cluster_Integration')
    ##########################################################################
    # Compute statistics and plot data by method for Cluster_Integration
    ##########################################################################
    plotBoxPlot(df, 'Cluster_Integration_Hue', x='K_Integration',
                hue='Cluster_Integration')
    f = factorplot(data=df, x='K_Integration', y='AUC',
                   col='Cluster_Integration', kind='box', aspect=1.0)
    f.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig('Cluster_Integration.pdf', dpi=150)
    plt.clf()
    for sub in df.Cluster_Integration.unique():
        currDF = df.loc[(df['Cluster_Integration'] == sub)]
        name = 'Cluster_Integration_{}'.format(sub)
        computeWilcoxonRankSum(currDF, name)
        plotBoxPlot(currDF, name, x='K_Integration')
    ##########################################################################
    # Compute statistics and plot data by clustering constant
    ##########################################################################
    for k in df.K.unique():
        currDF = df.loc[(df['K'] == k)]
        name = 'K_{}'.format(k)
        computeWilcoxonRankSum(currDF, name)
        plotBoxPlot(currDF, name, x='Cluster_Integration', hue='K_Integration')
    plotBoxPlot(df, 'K_Hue', hue='K')
    f = factorplot(data=df, x='Method', y='AUC',
                   col='K', kind='box', aspect=1.0)
    f.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig('Ks.pdf', dpi=150)
    plt.clf()
    ##########################################################################
    # Compute statistics and plot data by clustering constant
    ##########################################################################
    for p in df.Protein.unique():
        currDF = df.loc[(df['Protein'] == p)]
        name = 'Protein_{}'.format(p)
        computeWilcoxonRankSum(currDF, name)
        plotBoxPlot(currDF, name, x='Cluster_Integration', hue='K_Integration')
    plotBoxPlot(df, 'Protein_Hue', x='Protein', hue='Method')
    f = factorplot(data=df, x='Method', y='AUC',
                   col='Protein', kind='box', aspect=1.0, order=clus_int_order)
    f.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig('Proteins.pdf', dpi=150)
    plt.clf()
