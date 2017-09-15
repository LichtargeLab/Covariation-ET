'''
Created on Sep 15, 2017

@author: daniel
'''
from subprocess import Popen
from sys import argv
import os
import re

if __name__ == '__main__':
    inDir = argv[1]
    files = os.listdir(inDir)
    print files
    inputDict = {}
    for f in files:
        check = re.search(r'(\d[\d|a-z]{3}[A-Z])', f)
        query = check.group(1)
        if(query not in inputDict):
            inputDict[query] = [None, None]
        if(f.endswith('fa')):
            inputDict[query][0] = f
        elif(f.endswith('pdb')):
            inputDict[query][0] = f
        else:
            pass
    print inputDict