import unittest
import getopt
import sys
import os

## parse inputs
from unittests.TestsApi import ApiTest

try:
    optlist, args = getopt.getopt(sys.argv[1:],'v')
except getopt.GetoptError:
    print(getopt.GetoptError)
    print(sys.argv[0] + "-v")
    print("... the verbose flag (-v) may be used")
    sys.exit()

VERBOSE = False
RUNALL = False

sys.path.append(os.path.realpath(os.path.dirname(__file__)))

for o, a in optlist:
    if o == '-v':
        VERBOSE = True

## api tests
from unittests.TestsApi import ApiTest
ApiTestSuite = unittest.TestLoader().loadTestsFromTestCase(ApiTest)

## model tests
from unittests.TestsModel import *
ModelTestSuite = unittest.TestLoader().loadTestsFromTestCase(ModelTest)

## logger tests
from unittests.TestsLogger import *
LoggerTestSuite = unittest.TestLoader().loadTestsFromTestCase(LoggerTest)

MainSuite = unittest.TestSuite([LoggerTestSuite,ModelTestSuite,ApiTestSuite])
