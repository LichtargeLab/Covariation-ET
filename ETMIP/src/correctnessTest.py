import unittest
from etmipCollab import importAlignment, removeGaps, distanceMatrix
from etmip10forOptimizatoin import remove_gaps, distance_matrix


class Test(unittest.TestCase):

    def testAlignmentImport(self):
        testFile = '/Users/dmkonecki/git/ETMIP/ETMIP/Input/1c17A.fa'
        alignment_dict = {}
        files = open(testFile, "r")
        for line in files:
            if line.startswith(">"):
                if "query" in line.lower():
                    query_desc = line
                key = line.rstrip()
                alignment_dict[key] = ''
            else:
                alignment_dict[key] = alignment_dict[key] + line.rstrip()
        newRes = importAlignment(testFile)
        self.assertEqual(len(alignment_dict), len(newRes),
                         'Different numbers of elements')
        for key in alignment_dict:
            self.assertEqual(alignment_dict[key], newRes[key], 'Line not equal: {}, {}, {}'.format(
                key, alignment_dict[key], newRes[key]))

    def testRemoveGaps(self):
        testFile = '/Users/dmkonecki/git/ETMIP/ETMIP/Input/1c17A.fa'
        alignment_dict = {}
        files = open(testFile, "r")
        for line in files:
            if line.startswith(">"):
                if "query" in line.lower():
                    query_desc = line
                key = line.rstrip()
                alignment_dict[key] = ''
            else:
                alignment_dict[key] = alignment_dict[key] + line.rstrip()
        qn1, nad1 = remove_gaps(alignment_dict)
        newAD = importAlignment(testFile)
        qn2, nad2 = removeGaps(newAD)
        self.assertEqual(qn1, qn2, 'Queries not equal')
        self.assertEqual(len(nad1), len(nad2), 'Different numbers of elements')
        for key in nad1:
            self.assertEqual(nad1[key], nad2[key], 'Uneven removal: {}, {}, {}'.format(
                key, nad1[key], nad2[key]))

    def testDistanceMatrix(self):
        testFile = '/Users/dmkonecki/git/ETMIP/ETMIP/Input/1c17A.fa'
        alignment_dict = {}
        files = open(testFile, "r")
        for line in files:
            if line.startswith(">"):
                if "query" in line.lower():
                    query_desc = line
                key = line.rstrip()
                alignment_dict[key] = ''
            else:
                alignment_dict[key] = alignment_dict[key] + line.rstrip()
        qn1, nad1 = remove_gaps(alignment_dict)
        vm1, kl1 = distance_matrix(nad1)
        newAD = importAlignment(testFile)
        qn2, nad2 = removeGaps(newAD)
        vm2, kl2 = distanceMatrix(nad2)
        self.assertEqual(len(kl1), len(kl2), 'Key list lengths differ')
        for e in kl1:
            self.assertTrue(
                e in kl2, 'Element not in both lists: {}'.format(e))
        self.assertEqual(vm1.shape, vm2.shape, 'Matrix dimensions differ')
        print vm1
        print vm2.T
        for i in range(vm1.shape[0]):
            for j in range(vm2.shape[1]):
                self.assertEqual(vm1[i, j], vm2[i, j],
                                 'Elements different: {}, {}'.format(vm1[i, j], vm2[i, j]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGenerateMMatrix']
    unittest.main()
