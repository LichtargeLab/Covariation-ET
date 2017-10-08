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
from seaborn import boxplot, factorplot


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


def plotBoxPlot(df, name, x='Method', y='AUC', hue=None):
    sns.set_style('whitegrid')
    if(hue):
        boxplot(data=df, x=x, y=y, hue=hue)
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        boxplot(data=df, x=x, y=y)
    plt.xticks(rotation=90)
    plt.ylim([0.5, 1.0])
    plt.tight_layout()
    if(hue):
        plt.savefig('{}.png'.format(name), dpi=150, bbox_inches='tight',
                    bbox_extra_artists=[lgd])
    else:
        plt.savefig('{}.png'.format(name), dpi=150, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    ##########################################################################
    # Import data
    ##########################################################################
    os.chdir('/Users/dmkonecki/Desktop/ETMIPC/')
#     os.chdir('/media/daniel/ExtraDrive1/Results/ETMIPC/ClusteringDistanceAndLinkage/')
    df = pandas.read_csv('ClusteringDistanceAUCs.txt',
                         delimiter='\t', header=0)
    ##########################################################################
    # Add a column for methods
    ##########################################################################
    df['Method'] = df[['ClusteringDistance', 'SequenceDistance', 'Linkage']].apply(
        lambda x: '\n'.join(list(set(x))), axis=1)
    ##########################################################################
    # Compute statistics and plot data across all tested parameters
    ##########################################################################
    computeWilcoxonRankSum(df, 'All')
    plotBoxPlot(df, 'All')
    ##########################################################################
    # Compute statistics and plot data by clustering constant
    ##########################################################################
    for k in df.K.unique():
        currDF = df.loc[(df['K'] == k)]
        name = 'K_{}'.format(k)
        computeWilcoxonRankSum(currDF, name)
        plotBoxPlot(currDF, name)
    plotBoxPlot(df, 'K_Hue', hue='K')
    f = factorplot(data=df, x='Method', y='AUC',
                   col='K', kind='box', aspect=1.0)
    f.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig('Ks.png', dpi=150)
    plt.clf()
    ##########################################################################
    # Compute statistics and plot data by clustering constant
    ##########################################################################
    for p in df.Protein.unique():
        currDF = df.loc[(df['Protein'] == p)]
        name = 'Protein_{}'.format(p)
        computeWilcoxonRankSum(currDF, name)
        plotBoxPlot(currDF, name)
    plotBoxPlot(df, 'Protein_Hue', x='Protein', hue='Method')
    f = factorplot(data=df, x='Method', y='AUC',
                   col='Protein', kind='box', aspect=1.0)
    f.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig('Proteins.png', dpi=150)
    plt.clf()
