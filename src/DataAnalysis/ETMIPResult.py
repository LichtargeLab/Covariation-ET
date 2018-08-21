'''
Created on Oct 19, 2017

@author: daniel
'''
from sklearn.metrics import auc, roc_curve
from pandas import read_csv
import numpy as np


class ETMIPResult(object):
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
        self.data = read_csv(self.path, delim_whitespace=True, skiprows=1,
                             names=['Sort', 'ResI', 'i(AA)', 'ResJ', 'j(AA)',
                                    'sorted', 'cvg(sort)', 'interface',
                                    'contact', 'number', 'AveContact'])
        self.data.sort_values(by=['ResI', 'ResJ'],
                              ascending=[True, True], inplace=True)
        self.positions = sorted(set(self.data['ResI'].unique()) |
                                set(self.data['ResJ'].unique()))
        self.size = self.data.shape[0]

    def checkAccuracy(self, pdbRef, chain, threshold=8.0):
        pdbRef.findDistance()
        df = self.data.copy(deep=True)
        df['Index1'] = df[['ResI']].apply(lambda x: x - 1, axis=1)
        df['Index2'] = df[['ResJ']].apply(lambda x: x - 1, axis=1)
        df['Distance'] = pdbRef.residueDists[chain][df['Index1'].values,
                                                    df['Index2'].values]
        df['Truth'] = np.where(df['Distance'] <= threshold, 1, 0)
        fpr, tpr, _thresholds = roc_curve(df['Truth'].values,
                                          df['sorted'].values,
                                          pos_label=1)
        roc_auc = auc(fpr, tpr)
        return roc_auc


if __name__ == '__main__':
    test = '/cedar/atri/projects/DannySymposiumData/ETMIP-AW/1a26A.tree_mip_sorted'
    etmipRes = ETMIPResult(test)
    etmipRes.importData()
    from ClassBasedCode.PDBReference import PDBReference
    test2 = '../Input/23TestGenes/query_1a26A.pdb'
    pdbRef = PDBReference(test2)
    pdbRef.importPDB()
    print etmipRes.checkAccuracy(pdbRef, 'A')
