'''
Created on Oct 18, 2017

@author: daniel
'''
import os
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


if __name__ == '__main__':
    # Results directory
    resDir = '/media/daniel/ExtraDrive1/Results/ETMIPC/MethodComparisons/'
    # Location of pdbs
    pdbDir = os.path.abspath('../Input/23TestGenes/') + '/'
    # Location of DCA results
    dcaDir = '/cedar/atri/projects/Coupling/DannySymposiumData/DCA-results/'
    # Location of ETMIP results
    etmipDir = '/cedar/atri/projects/Coupling/DannySymposiumData/ETMIP-AW/'
    # Location of ETMIPC Results
    etmipcDir = '/media/daniel/ExtraDrive1/Results/ETMIPC/KAndClusterIntegration/SumKsSumClusters/'
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
        ETMIPTimes = parseETMIPTimeFile('ETMIP-times.txt')
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
                r'(\d[a-z|\d]{3}[A-Z]?).tree_mip_sorted', f).group(1)
            fileDict[query]['ETMIP'] = f
        for d in os.listdir(etmipcDir):
            fName = d + '/'
            for f in os.listdir(etmipcDir + fName):
                if(re.match(r'(\d[a-z|\d]{3}[A-Z]?).*_results.txt', f)):
                    fName += f
                    fileDict[d]['ETMIPC'] = fName
                    break
                else:
                    continue
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
            methods += ['DCA', 'ET-MIp', 'ET-MIp-C']
            currTimes = ([ETMIPTimes[q]] + [timeData[x]
                                            for x in sorted(timeData.keys())] +
                         [sum(timeData.values())])
            queries2 += [q] * len(currTimes)
            alignmentSize2 += [alignSize] * len(currTimes)
            sequenceLength2 += [protLen] * len(currTimes)
            methods2 += (['ET-MIp'] + ['ET-MIp-C:{}'.format(x)
                                       for x in sorted(timeData.keys())] +
                         ['ET-MIp-C:Combined'])
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
    protein_order = df.sort_values(
        by='AlignmentSize', ascending=True).Protein.unique()
    sns.set_style('whitegrid')
    barplot(data=df, hue='Method', x='Protein', y='AUC',
            order=protein_order, hue_order=['DCA', 'ET-MIp', 'ET-MIp-C'],
            ci=None)
    plt.xticks(rotation=45)
    plt.ylim([0.5, 1.0])
    plt.ylabel('AUC')
    plt.tight_layout()
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('Method_Accuracy_Comparison.pdf', dpi=150, bbox_inches='tight',
                bbox_extra_artists=[lgd])
    plt.close()
    ###########################################################################
    embed()
    df2['Time(%ET-MIp)'] = Series([])

    sns.set_style('whitegrid')
    barplot(data=df2, hue='Method', x='Protein', y='Time(sec)', ci=None,
            order=protein_order, hue_order=['ET-MIp', 'ET-MIp-C:2',
                                            'ET-MIp-C:3', 'ET-MIp-C:5',
                                            'ET-MIp-C:7', 'ET-MIp-C:10',
                                            'ET-MIp-C:25', 'ET-MIp-C:Combined'])
    plt.xticks(rotation=45)
#     plt.ylim([0.5, 1.0])
    plt.ylabel('Time(sec)')
    plt.tight_layout()
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('Method_Time_Comparison.pdf', dpi=150, bbox_inches='tight',
                bbox_extra_artists=[lgd])
    plt.close()
