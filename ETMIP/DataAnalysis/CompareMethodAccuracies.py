'''
Created on Oct 18, 2017

@author: daniel
'''
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import re
from IPython import embed
import seaborn as sns
from seaborn import barplot
import matplotlib.pyplot as plt
from pandas import DataFrame, Series, read_csv
from DCAResult import DCAResult
from ETMIPResult import ETMIPResult
from ClassBasedCode.PDBReference import PDBReference


def parseETMIPTimeFile(filename):
    times = {}
    fileHandle = open(filename, 'rb')
    for line in fileHandle:
        match = re.match(r'(\d\.\d+)\t(\d+\.\d+)\t(\d[a-z|\d]{3}[A-Z]?)', line)
        if(match):
            times[match.group(3)] = float(match.group(2))
    fileHandle.close()
    return times


def parseETMIPCResult(filename):
    fileHandle = open(filename, 'rb')
    aucData = {}
    timeData = {}
    proteinLength = None
    alignmentSize = None
    for line in fileHandle:
        match = re.match(r'\t(\d+)\t(\d\.\d+)\t(\d+\.\d+)', line)
        if(match):
            aucData[int(match.group(1))] = float(match.group(2))
            timeData[int(match.group(1))] = float(match.group(3))
        else:
            match2 = re.match(
                r'.*Alignment Size: (\d+) Length of protein: (\d+).*', line)
            if(match2):
                alignmentSize = int(match2.group(1))
                proteinLength = int(match2.group(2))
            else:
                continue
    fileHandle.close()
    return proteinLength, alignmentSize, aucData, timeData


def collectFiles(pdbDir, dcaDir, etmipcDir):
    fileDict = {}
    for f in os.listdir(pdbDir):
        match = re.match(r'.*(\d[a-z|\d]{3}[A-Z]?).*.pdb', f)
        if(match):
            query = match.group(1)
            fileDict[query] = {}
            fileDict[query]['PDB'] = f
        else:
            continue
    for f in os.listdir(dcaDir):
        query = re.match(r'(\d[a-z|\d]{3}[A-Z]?).*.txt', f).group(1)
        fileDict[query]['DCA'] = f
    for f in os.listdir(etmipDir):
        query = re.match(
            r'^(\d[a-z|\d]{3}[A-Z]?).tree_mip_sorted$', f)
        if(query):
            fileDict[query.group(1)]['ETMIP'] = f
        else:
            print f
            continue
    for d in os.listdir(etmipcDir):
        fName = d + '/'
        if(not os.path.isdir(etmipcDir + fName)):
            continue
        for f in os.listdir(etmipcDir + fName):
            if(re.match(r'(\d[a-z|\d]{3}[A-Z]?).*_results.txt', f)):
                fName += f
                fileDict[d]['ETMIPC'] = fName
                break
            else:
                continue
    return fileDict


def calculateRelativeTimes(dataFrame):
    dataFrame['Time(%ET-MIp)'] = Series([])
    grp = dataFrame['Time(sec)'].groupby(dataFrame['Method'])
    base = grp.get_group('ET-MIp')
    for m in dataFrame['Method'].unique():
        print m
        currG = grp.get_group(m)
        print currG
        indices = currG.index
        dataFrame.at[indices, 'Time(%ET-MIp)'] = currG.values / base.values
    return


if __name__ == '__main__':
    # Results directory
    resDir = '/Users/dmkonecki/Desktop/ETMIPC/MethodComparisons/'
#     resDir = '/media/daniel/ExtraDrive1/Results/ETMIPC/MethodComparisons/'
    # Location of pdbs
    pdbDir = os.path.abspath('../Input/23TestGenes/') + '/'
    # Location of DCA results
#     dcaDir = '/cedar/atri/projects/Coupling/DannySymposiumData/DCA-results/'
    dcaDir = '/Users/dmkonecki/Desktop/ETMIPC/DCA-Results/'
    # Location of ETMIP results
#     etmipDir = '/cedar/atri/projects/Coupling/DannySymposiumData/ETMIP-AW/'
    etmipDir = '/Users/dmkonecki/Desktop/ETMIPC/ETMIP-Results/'
    # Location of ETMIPC Results
#     etmipcDir = '/media/daniel/ExtraDrive1/Results/ETMIPC/KAndClusterIntegration/SumKsSumClusters/'
    etmipcDir = '/Users/dmkonecki/Desktop/ETMIPC/KAndClusterIntegration/SumKsSumClusters/'
    # DataFrame to store results
    df = DataFrame(index=range(23 * 3), columns=['Protein', 'SequenceLength',
                                                 'AlignmentSize', 'Method',
                                                 'AUC'])
    df2 = DataFrame(index=range(23 * 8), columns=['Protein', 'SequenceLength',
                                                  'AlignmentSize', 'Method',
                                                  'Time(sec)'])
    # Finding relevant result files
    os.chdir(resDir)
    if(not os.path.exists('Method_AUC_Results.csv')):
        fileDict = collectFiles(pdbDir, dcaDir, etmipcDir)
        ETMIPTimes = parseETMIPTimeFile('ETMIP-times.txt')
        queries = []
        alignmentSize = []
        sequenceLength = []
        methods = []
        AUCs = []
        queries2 = []
        alignmentSize2 = []
        sequenceLength2 = []
        methods2 = []
        times = []
        for q in fileDict:
            print q
            currPDB = PDBReference(pdbDir + fileDict[q]['PDB'])
            currPDB.importPDB()
            currDCA = DCAResult(dcaDir + fileDict[q]['DCA'])
            currDCA.importData()
            try:
                currDCAAUC = currDCA.checkAccuracy(currPDB, 'A')
            except:
                print 'Skipping'
                continue
            print currDCAAUC
            currETMIP = ETMIPResult(etmipDir + fileDict[q]['ETMIP'])
            currETMIP.importData()
            try:
                currETMIPAUC = currETMIP.checkAccuracy(currPDB, chain='A')
            except:
                print 'Skipping'
                continue
            print currETMIPAUC
            protLen, alignSize, aucData, timeData = parseETMIPCResult(
                etmipcDir + fileDict[q]['ETMIPC'])
            print aucData[max(aucData.keys())]
            AUCs += [currDCAAUC, currETMIPAUC, aucData[max(aucData.keys())]]
            queries += [q] * 3
            alignmentSize += [alignSize] * 3
            sequenceLength += [protLen] * 3
            methods += ['DCA', 'ET-MIp', 'cET-MIp']
            currTimes = ([ETMIPTimes[q]] + [timeData[x]
                                            for x in sorted(timeData.keys())] +
                         [sum(timeData.values())])
            queries2 += [q] * len(currTimes)
            alignmentSize2 += [alignSize] * len(currTimes)
            sequenceLength2 += [protLen] * len(currTimes)
            methods2 += (['ET-MIp'] + ['cET-MIp:{}'.format(x)
                                       for x in sorted(timeData.keys())] +
                         ['cET-MIp:Combined'])
            times += currTimes
        df['Protein'] = Series(queries)
        df['SequenceLength'] = Series(sequenceLength)
        df['AlignmentSize'] = Series(alignmentSize)
        df['Method'] = Series(methods)
        df['AUC'] = Series(AUCs)
        df2['Protein'] = Series(queries2)
        df2['SequenceLength'] = Series(sequenceLength2)
        df2['AlignmentSize'] = Series(alignmentSize2)
        df2['Method'] = Series(methods2)
        df2['Time(sec)'] = Series(times)
        df.dropna(axis=0, how='all', inplace=True)
        df.to_csv('Method_AUC_Results.csv', sep='\t', header=True, index=False)
        df2.dropna(axis=0, how='all', inplace=True)
        df2.to_csv('Method_Time_Results.csv',
                   sep='\t', header=True, index=False)
    else:
        df = read_csv('Method_AUC_Results.csv', header=0, sep='\t')
        df2 = read_csv('Method_Time_Results.csv', header=0, sep='\t')
    ###########################################################################
#     sns.set_palette(["#e8546c", "#95e34d", "#de5dc7", "#60df92", "#8d79ea",
#                      "#d3c144", "#df7731", "#69a83d"])
    colors = ["#292934", "#FF2626", "#2c8c99", "#f28c26", "#cb48b7", "#1ac8ed",
              "#a0bc03"]
#     sns.set_palette(sns.xkcd_palette(["blue", "red", "electric green",
#                                       "bright purple", "amber", "cyan",
#                                       "magenta", "black"]))
#     sns.set_palette("bright", 7)
#     sns.set_palette("hls", 7)
    sns.set_style('whitegrid')
    ###########################################################################
    protein_order = df.sort_values(
        by='AlignmentSize', ascending=True).Protein.unique()
    barplot(data=df, hue='Method', x='Protein', y='AUC',
            order=protein_order, hue_order=['DCA', 'ET-MIp', 'cET-MIp'],
            ci=None, palette=sns.color_palette(colors[4:], 3))
    plt.xticks(rotation=45)
    plt.ylim([0.5, 0.85])
    plt.ylabel('AUC')
    plt.tight_layout()
    lgd = plt.legend(loc=1)
    plt.savefig('Method_Accuracy_Comparison.pdf', dpi=150, bbox_inches='tight',
                bbox_extra_artists=[lgd])
    plt.close()
    ###########################################################################
    calculateRelativeTimes(df2)
    ###########################################################################
    barplot(data=df2.loc[df2['Method'] != 'cET-MIp:Combined'], hue='Method',
            x='Protein', y='Time(sec)', ci=None, order=protein_order,
            hue_order=['ET-MIp', 'cET-MIp:K=2', 'cET-MIp:K=3', 'cET-MIp:K=5',
                       'cET-MIp:K=7', 'cET-MIp:K=10', 'cET-MIp:K=25'],
            palette=sns.color_palette(colors, 7))
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.ylabel('Time(sec)')
    plt.tight_layout()
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('Method_Time_Sec_Comparison.pdf', dpi=150, bbox_inches='tight',
                bbox_extra_artists=[lgd])
    plt.close()
    ###########################################################################
    sns.set_style('whitegrid')
    barplot(data=df2.loc[(df2['Method'] != 'ET-MIp') &
                         (df2['Method'] != 'cET-MIp:Combined')],
            hue='Method', x='Protein',
            y='Time(%ET-MIp)', ci=None, order=protein_order,
            hue_order=['cET-MIp:K=2', 'cET-MIp:K=3', 'cET-MIp:K=5',
                       'cET-MIp:K=7', 'cET-MIp:K=10', 'cET-MIp:K=25'],
            palette=sns.color_palette(colors[1:], 6))
    plt.xticks(rotation=45)
    plt.ylim([0.0, 1.0])
    plt.ylabel('Time(%ET-MIp)')
    plt.tight_layout()
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('Method_Time_Perc_Comparison.pdf', dpi=150, bbox_inches='tight',
                bbox_extra_artists=[lgd])
    plt.close()
    ###########################################################################
