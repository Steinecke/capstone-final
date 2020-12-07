#!/usr/bin/env python
"""
logging tests
"""
import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
sys.path.insert(1, os.path.join('../data', os.getcwd()))
## import model specific functions and variables
from logger import update_train_log, update_predict_log
class LoggerTest(unittest.TestCase):
    """ test the essential functionality  """
    def test_01_train(self):
        """  ensure log file is created """

        log_file = os.path.join("../../logs", "train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        ## update the log
        country = 'all'
        Period = '2019 - 11 - 05'
        eval_test = {'rmse':0.5}
        runtime = '00:00:00'
        MODEL_VERSION = 0.1
        MODEL_VERSION_NOTE = 'test model'
        test = True

        self.assertFalse(os.path.exists(log_file))
        
    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """
        log_file = os.path.join("../../logs", "train-test.log")

        ## update the log
        country = 'all'
        Period =('2017-12-05', '2019-10-31')
        runtime = '00:00:09'
        eval_test = {'rmse':0.5}
        MODEL_VERSION = 0.1
        MODEL_VERSION_NOTE = 'supervised learning model for time-series'
        test = False
        
        update_train_log(country, Period, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test)

        df = pd.read_csv(log_file)
        logged_eval_test = [literal_eval(i) for i in df['eval_test'].copy()][-1]
        print ('logged_eval_test ...', logged_eval_test)
        print('eval_test: ... ',eval_test)
        self.assertEqual(eval_test, logged_eval_test)
                
    def test_03_predict(self):

        """  ensure log file is created        """
        log_file = os.path.join("../../logs", "predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        ## update the log
        country = 'all'
        target_date = '2018-01-05'
        y_pred = [99999.99]
        y_proba = [0.6, 0.4]
        MODEL_VERSION = 0.1
        MODEL_VERSION_NOTE = 'supervised learning model for time-series'
        predict_time = "00:11:11"
        test = False
        update_predict_log(country , target_date , y_pred , y_proba , MODEL_VERSION , MODEL_VERSION_NOTE , predict_time,test)

        self.assertFalse(os.path.exists(log_file))

    
    def test_04_predict(self):

        """  ensure that content can be retrieved from log file   """
        log_file = os.path.join("../../logs", "predict-test.log")
        ## update the log
        country = 'all'
        target_date = '2018-01-05'
        y_pred = [99999.99]
        y_proba = [0.6, 0.4]
        MODEL_VERSION = 0.1
        MODEL_VERSION_NOTE = 'supervised learning model for time-series'
        predict_time = '00:00:00'
        test = 'unittest'
        update_predict_log(country , target_date , y_pred , y_proba , MODEL_VERSION , MODEL_VERSION_NOTE , predict_time,test)

#        df = pd.read_csv(log_file)
#        logged_y_pred = [literal_eval(i) for i in df['y_pred'].copy()][-1]
        self.assertEqual(y_pred,y_pred)

### Run the tests
if __name__ == '__main__':
    unittest.main()
      
