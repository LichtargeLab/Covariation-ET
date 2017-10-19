'''
Created on Oct 18, 2017

@author: daniel
'''
import os
import re
from pandas import DataFrame, Series
from DCAResult import DCAResult
from ClassBasedCode.PDBReference import PDBReference


if __name__ == '__main__':
    #
    pdbDir = '../Input/23TestGenes/'
    #
    dcaDir = '/cedar/atri/projects/DannySymposiumData/DCA-results/'
    #
    etmipDir = '/cedar/atri/projects/DannySymposiumData/ETMIP-AW/'
    #
    etmipcDir = ''
    #
    df = DataFrame(index=range(22), row=['Query', 'DCA', 'ET-MIp', 'ET-MIp-C'])
    #
    fileDict = {}
    for f in os.listdir(pdbDir):
        query = re.match(r'.*(\d[a-z|\d]{3}[A-Z]?).*.pdb', f).group(1)
        fileDict[query] = {}
        fileDict[query]['PDB'] = f
    for f in os.listdir(dcaDir):
        query = re.match(r'(\d[a-z|\d]{3}[A-Z]?).*.txt', f).group(1)
        fileDict[query]['DCA'] = f
    df['Query'] = Series(sorted(fileDict.keys()))
    dcaRes = []
    for q in df['Query'].values:
        currDCA = DCAResult(fileDict[q]['DCA'])
        currPDB = PDBReference(fileDict[q]['PDB'])
        currAUC = currDCA.checkAccuracy(currPDB)
        dcaRes.appen(currAUC)