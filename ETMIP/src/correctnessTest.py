import unittest
from etmipCollab import (importAlignment, removeGaps, distanceMatrix,
                         wholeAnalysis, aggClustering, importPDB, findDistance)
from etmip10forOptimizatoin import (import_alignment, remove_gaps,
                                    distance_matrix, whole_analysis,
                                    agg_clustering, import_pdb, find_distance)
import numpy as np


class Test(unittest.TestCase):

    #     def testAlignmentImport(self):
    #         testFile = '/Users/dmkonecki/git/ETMIP/ETMIP/Input/1c17A.fa'
    #         AD = import_alignment(testFile)
    #         newRes = importAlignment(testFile)
    #         self.assertEqual(len(AD), len(newRes),
    #                          'Different numbers of elements')
    #         for key in AD:
    #             self.assertEqual(AD[key], newRes[key], 'Line not equal: {}, {}, {}'.format(
    #                 key, AD[key], newRes[key]))

    #     def testRemoveGaps(self):
    #         testFile = '/Users/dmkonecki/git/ETMIP/ETMIP/Input/1c17A.fa'
    #         AD = import_alignment(testFile)
    #         newAD = importAlignment(testFile)
    #         qn1, nad1 = remove_gaps(AD)
    #         qn2, nad2 = removeGaps(newAD)
    #         self.assertEqual(qn1, qn2, 'Queries not equal')
    #         self.assertEqual(len(nad1), len(nad2), 'Different numbers of elements')
    #         for key in nad1:
    #             self.assertEqual(nad1[key], nad2[key], 'Uneven removal: {}, {}, {}'.format(
    #                 key, nad1[key], nad2[key]))

    #     def testDistanceMatrix(self):
    #         testFile = '/Users/dmkonecki/git/ETMIP/ETMIP/Input/1c17A.fa'
    #         AD = import_alignment(testFile)
    #         newAD = importAlignment(testFile)
    #         qn1, nad1 = remove_gaps(AD)
    #         qn2, nad2 = removeGaps(newAD)
    #         vm1, kl1 = distance_matrix(nad1)
    #         vm2, kl2 = distanceMatrix(nad2)
    #         self.assertEqual(len(kl1), len(kl2), 'Key list lengths differ')
    #         for e in kl1:
    #             self.assertTrue(
    #                 e in kl2, 'Element not in both lists: {}'.format(e))
    #         self.assertEqual(vm1.shape, vm2.shape, 'Matrix dimensions differ')
    #         for i in range(vm1.shape[0]):
    #             for j in range(vm2.shape[1]):
    #                 self.assertEqual(vm1[i, j], vm2[i, j],
    #                                  'Elements different: {}, {}'.format(vm1[i, j], vm2[i, j]))

    #     def testAggClustering(self):
    #         testFile = '/Users/dmkonecki/git/ETMIP/ETMIP/Input/1c17A.fa'
    #         AD = import_alignment(testFile)
    #         newAD = importAlignment(testFile)
    #         qn1, nad1 = remove_gaps(AD)
    #         qn2, nad2 = removeGaps(newAD)
    #         vm1, kl1 = distance_matrix(nad1)
    #         vm2, kl2 = distanceMatrix(nad2)
    #         cd1, g1, l1 = agg_clustering(2, vm1, nad1, precomputed=True)
    #         cd2, g2, l2 = aggClustering(2, vm2, kl2, precomputed=True)
    #         self.assertEqual(g1, g2, 'Cluster sets are not identical')
    #         self.assertEqual(len(cd1), len(cd2),
    #                          'Cluster dictionaries differ in size')
    #         self.assertEqual(len(l1), len(l2), 'Cluster label lengths differ')
    #         for i in range(len(l1)):
    #             self.assertEqual(
    #                 l1[i], l2[i], 'Cluster labels differ:{}\n{}\n{}'.format(i, l1, l2))
    #         queryKey = '>query_1c17A'
    #         c0_1 = None
    #         c0_2 = None
    #         c1_1 = None
    #         c1_2 = None
    #         for key in cd1:
    #             if(queryKey in cd1[key]):
    #                 c0_1 = sorted(cd1[key])
    #             else:
    #                 c1_1 = sorted(cd1[key])
    #             if(queryKey in cd2[key]):
    #                 c0_2 = sorted(cd2[key])
    #             else:
    #                 c1_2 = sorted(cd2[key])
    #         self.assertEqual(c0_1, c0_2, 'Cluster 0 does not match:\n{}\n{}'.format(
    #             c0_1, c0_2))
    #         self.assertEqual(c1_1, c1_2, 'Cluster 1 does not match:\n{}\n{}'.format(
    #             c1_1, c1_2))

    #     def testWholeAnalysis(self):
    #         testFile = '/Users/dmkonecki/git/ETMIP/ETMIP/Input/1c17A.fa'
    #         aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
    #                    'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    #         aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    #         AD = {}
    #         files = open(testFile, "r")
    #         for line in files:
    #             if line.startswith(">"):
    #                 if "query" in line.lower():
    #                     query_desc = line
    #                 key = line.rstrip()
    #                 AD[key] = ''
    #             else:
    #                 AD[key] = AD[key] + line.rstrip()
    #         qn1, nad1 = remove_gaps(AD)
    #         MM1 = whole_analysis(nad1, aa_list)
    #         newAD = importAlignment(testFile)
    #         qn2, nad2 = removeGaps(newAD)
    #         MM2 = wholeAnalysis(nad2, aa_dict)
    #         self.assertEqual(MM1.shape, MM2.shape, 'Matries have different dims')
    #         for i in range(MM1.shape[0]):
    #             for j in range(MM2.shape[1]):
    #                 self.assertLess(MM1[i, j] - MM2[i, j], 1e-15, 'Matrices differ: ({}, {}): {} - {} = {}'.format(
    #                     i, j, MM1[i, j], MM2[i, j], (MM1[i, j] - MM2[i, j])))

    def testImportPDB(self):
        fileName = '/Users/dmkonecki/git/ETMIP/ETMIP/Input/query_1c17A.pdb'
        pdbData1 = import_pdb(fileName)
        pdbData2 = importPDB(fileName)
        self.assertEqual(len(pdbData1), len(pdbData2), 'Num elements mismatch')
        for i in range(len(pdbData1)):
            row1 = pdbData1[i]
            row2 = pdbData2[i]
            resname1 = (row1[17:20].strip())
            resname2 = row2[0]
            self.assertEqual(resname1, resname2,
                             'Resname {} differs: {} vs {}'.format(
                                 i, resname1, resname2))
            resnumdict1 = int(row1[22:26].strip())
            resnumdict2 = int(row2[1])
            self.assertEqual(resnumdict1, resnumdict2,
                             'Resnumdict differs: {} - {} vs {}'.format(
                                 i, resnumdict1, resnumdict2))
            xvaluedict = float(row1[31:38].strip())
            yvaluedict = float(row1[39:46].strip())
            zvaluedict = float(row1[47:55].strip())
            resatomlisttemp1 = list((xvaluedict, yvaluedict, zvaluedict))
            resatomlisttemp2 = np.asarray([float(row2[2]), float(row2[3]),
                                           float(row2[4])]).tolist()
            self.assertEqual(resatomlisttemp1, resatomlisttemp2,
                             'Atom lists differ: {} - {} vs {}'.format(
                                 i, resatomlisttemp1, resatomlisttemp2))
#     def testComputePDBDist(self):
#         distancedict, PDBresidueList, ResidueDict = find_distance(fileName)
#         distancedict2, PDBresidueList2, ResidueDict2, _residuedictionary = findDistance(
#             pdbData)
#         self.assertEqual(len(distancedict), len(distancedict2),
#                          'Size distance dict not equal')
#         for key in distancedict:
#             self.assertEqual(distancedict[key], distancedict2[key],
#                              'distdict elements not equal {}: {} vs {}'.format(
#                                  key, distancedict[key], distancedict2[key]))
#             print('{} - {} vs {}'.format(key,
#                                          distancedict[key], distancedict2[key]))
#         self.assertEqual(len(PDBresidueList), len(PDBresidueList2),
#                          'Number of PDB residues not equal')
#         self.assertEqual(PDBresidueList, PDBresidueList2, 'Lists not equal')
#         self.assertEqual(len(ResidueDict), len(ResidueDict2),
#                          'Residue dicts differ in length')
#         for key in ResidueDict:
#             self.assertEqual(ResidueDict[key], ResidueDict2[key],
#                              'Elements not equal: {}\n{}\n{}'.format(
#                 key, ResidueDict[key], ResidueDict2[key]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGenerateMMatrix']
    unittest.main()
