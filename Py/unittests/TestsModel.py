#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
#sys.path.insert(1, os.path.join('..', os.getcwd()))
os.path.join(os.getcwd(), "models")
path = os.path.join(os.getcwd(), "models")
print("path ... :",path)
## import model specific functions and variables

from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """
        ## train the model
#        data_dir = os.path.join(os.getcwd(), "data", "cs-train")
#        model_train(test=True,data_dir)
        print(os.path.exists(os.path.join(os.getcwd(), "models", "test-all-0_1.joblib")))

        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), "models", "test-all-0_1.joblib")))

    def test_02_load(self):
        """
        test the train functionality        """
        ## train the model
#        model = model_load(prefix='sl', data_dir=None, training=True)
        print("Test2")
        path = os.path.join('../../..', os.getcwd(), 'models')
        print("path: ...", (os.path.join('../../..', os.getcwd(), "models")))
        print(os.path.exists(os.path.join('../../..', os.getcwd())))
        import glob
        list= glob.glob(path + '\\sl*.*')
        print(list)
        print(len(list)==0)
        self.assertFalse(len(list)==0)


    def test_03_predict(self):
        """
        test the predict function input
        country, year, month, day, all_models=None, test=None
        """
        ## load model first
        print("Test 3")
        model = model_load(prefix='sl', data_dir=None, training=True)

        ## ensure that a list can be passed
#        country = 'all'
#        year = '2019'
#        month = '11'
#        day = '05'
#    query = ['all', 2019, 11, 12]
#        y_pred = model_predict(query)
#        y_pred = model_predict(country, year, month, day)
#        print("y_pred",y_pred)
#        result = model_predict(query, model, test=True)
#        y_pred = result['y_pred']
#        self.assertTrue(y_pred)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
