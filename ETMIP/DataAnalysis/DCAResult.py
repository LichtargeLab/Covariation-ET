'''
Created on Oct 18, 2017

@author: daniel
'''
from sklearn.metrics import auc, roc_curve
from pandas import read_csv
import numpy as np


class DCAResult(object):
    '''
    classdocs
    '''

    def __init__(self, filePath):
        '''
        Constructor
        '''
        self.path = filePath
        self.data = None
        self.positions = None
        self.size = None

    def importData(self):
        self.data = read_csv(self.path, delimiter=' ', header=None,
                             names=['Position1', 'Position2', 'Score'])
        self.data.sort_values(by=['Position1', 'Position2'],
                              ascending=[True, True], inplace=True)
        self.positions = sorted(set(self.data['Position1'].unique()) |
                                set(self.data['Position2'].unique()))
        self.size = self.data.shape[0]

    def checkAccuracy(self, pdbRef, chain, threshold=8.0):
        pdbRef.findDistance()

        df = self.data.copy(deep=True)
        df['Index1'] = df[['Position1']].apply(lambda x: x - 1, axis=1)
        df['Index2'] = df[['Position2']].apply(lambda x: x - 1, axis=1)
        df['Distance'] = pdbRef.residueDists[chain][df['Index1'].values,
                                                    df['Index2'].values]
        df['Truth'] = np.where(df['Distance'] <= threshold, 1, 0)
        fpr, tpr, _thresholds = roc_curve(df['Truth'].values,
                                          df['Score'].values,
                                          pos_label=1)
        roc_auc = auc(fpr, tpr)
        return roc_auc


if __name__ == '__main__':
    test = '/cedar/atri/projects/DannySymposiumData/DCA-results/1a26A-03-11-2017_DCAresults.txt'
    dcaRes = DCAResult(test)
    dcaRes.importData()
    from ClassBasedCode.PDBReference import PDBReference
    test2 = '../Input/23TestGenes/query_1a26A.pdb'
    pdbRef = PDBReference(test2)
    pdbRef.importPDB()
    print dcaRes.checkAccuracy(pdbRef, 'A')
