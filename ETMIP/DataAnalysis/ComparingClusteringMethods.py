'''
Created on Oct 4, 2017

@author: daniel
'''
import os
import csv
import pandas
import itertools
import numpy as np
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


def plotBoxPlot(df, name, x='Method', y='AUC', hue=None, orient="v"):
    #     sns.set_style('whitegrid')
    #     sns.set_context('poster')
    sns.set(font_scale=1.5, style='whitegrid', context='poster')
    if(hue):
        boxplot(data=df, x=x, y=y, hue=hue, orient=orient)
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        boxplot(data=df, x=x, y=y, hue=hue, orient=orient, color='grey')

    if(orient == 'h'):
        plt.yticks(rotation=45)
        plt.xlim([0.5, 1.0])
        plt.xticks(np.arange(0.5, 1.01, 0.1))
    else:
        plt.xticks(rotation=45)
        plt.ylim([0.5, 1.0])
        plt.yticks(np.arange(0.5, 1.01, 0.1))
    plt.tight_layout()
    plt.rcParams['figure.figsize'] = (10, 10)
    if(hue):
        plt.savefig('{}.pdf'.format(name), dpi=150, bbox_inches='tight',
                    bbox_extra_artists=[lgd])
    else:
        plt.savefig('{}.pdf'.format(name), dpi=150, bbox_inches='tight')
    plt.close()


def plotFactorPlot(df, col, name, x='Method', y='AUC', hue=None):
    f = factorplot(data=df, x=x, y=y, col=col, hue=hue, kind='box', aspect=1.0,
                   legend_out=True)
    f.set_xticklabels(rotation=45)
    plt.tight_layout()
    plt.savefig('{}.pdf'.format(name), dpi=150)
    plt.close()


if __name__ == '__main__':
    ##########################################################################
    # Import data
    ##########################################################################
    os.chdir('/Users/dmkonecki/Desktop/ETMIPC/ClusteringDistanceAndLinkage/')
#     os.chdir('/media/daniel/ExtraDrive1/Results/ETMIPC/ClusteringDistanceAndLinkage/')
    originalDF = pandas.read_csv('ClusteringDistanceAUCs.txt',
                                 delimiter='\t', header=0)
    if(not os.path.isdir('FiguresAndStats')):
        os.mkdir('FiguresAndStats')
    os.chdir('FiguresAndStats')
    #######################################################################
    # Add a column for methods
    #######################################################################
    originalDF['Method'] = originalDF[['ClusteringDistance', 'SequenceDistance',
                                       'Linkage']].apply(
        lambda x: '\n'.join(list(set(x))), axis=1)
    #######################################################################
    #######################################################################
#     df = originalDF.loc[originalDF['SequenceDistance'].isin(
#         ['Identity', 'Random'])]
#     df['Clustering Method'] = df[['ClusteringDistance', 'Linkage']].apply(
#         lambda x: '\n'.join(list(set(x))), axis=1)
#     print df
#     currDF = df.loc[(df['K'] == 25)]
#     name = 'K_{}_{}'.format(25, 'Identity')
#     plotBoxPlot(currDF, name + '_Vertical', x='AUC',
#                 y='Clustering Method', orient="h")
#     exit()
    #######################################################################
    # Compute statistics and plot data across all tested parameters
    #######################################################################
    computeWilcoxonRankSum(originalDF, 'All')
    plotBoxPlot(originalDF, 'All')
    plotBoxPlot(originalDF, 'K_Hue', hue='K')
    plotFactorPlot(originalDF, col='K', name='Ks')
    plotBoxPlot(originalDF, 'Protein_Hue', x='Protein', hue='Method')
    plotFactorPlot(originalDF, col='Protein', name='Proteins')
    plotFactorPlot(originalDF, col='SequenceDistance', name='SequenceDistance',
                   hue='ClusteringDistance', x='Linkage')
    for s in (set(originalDF.SequenceDistance.unique()) -
              set(['Random']) | set(['Identity-EditDistance'])):
        print s
        if(not os.path.isdir(s)):
            os.mkdir(s)
        os.chdir(s)
        #######################################################################
        # Subset dataframe and alter column for methods
        #######################################################################
        df = originalDF.loc[originalDF['SequenceDistance'].isin(
            s.split('-') + ['Random'])]
        df['Clustering Method'] = df[['ClusteringDistance', 'Linkage']].apply(
            lambda x: '\n'.join(list(set(x))), axis=1)
        #######################################################################
        # Compute statistics and plot data across all tested parameters
        #######################################################################
        computeWilcoxonRankSum(df, 'All_{}'.format(s))
        plotBoxPlot(df, 'All_{}'.format(s), x='Clustering Method')
        #######################################################################
        # Compute statistics and plot data by clustering constant
        #######################################################################
        if(not os.path.isdir('K')):
            os.mkdir('K')
        os.chdir('K')
        for k in df.K.unique():
            currDF = df.loc[(df['K'] == k)]
            name = 'K_{}_{}'.format(k, s)
            computeWilcoxonRankSum(currDF, name)
            plotBoxPlot(currDF, name, x='Clustering Method')
        plotBoxPlot(df, 'K_Hue_{}'.format(s), hue='K', x='Clustering Method')
        plotFactorPlot(df, col='K', name='Ks_{}'.format(s),
                       x='Clustering Method')
        os.chdir('..')
        #######################################################################
        # Compute statistics and plot data by clustering constant
        #######################################################################
        if(not os.path.isdir('Proteins')):
            os.mkdir('Proteins')
        os.chdir('Proteins')
        for p in df.Protein.unique():
            currDF = df.loc[(df['Protein'] == p)]
            name = 'Protein_{}_{}'.format(p, s)
            computeWilcoxonRankSum(currDF, name)
            plotBoxPlot(currDF, name, x='Clustering Method')
        plotBoxPlot(df, 'Protein_Hue_{}'.format(s), x='Protein', hue='Method')
        plotFactorPlot(df, col='Protein', name='Proteins_{}'.format(s),
                       x='Clustering Method')
        os.chdir('..')
        os.chdir('..')
    os.chdir('..')
