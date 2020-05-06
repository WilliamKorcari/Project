#!/usr/bin/env python
# coding: utf-8

import conversion_tools as conv
import classification_tools as ct
import uproot
import unittest
import filecmp
from os import chdir
class TestNotebook(unittest.TestCase):
    #classification_tools   
    def test_make_model_no_metric(self):
        with self.assertRaises(ValueError):
            ct.make_model()
    
    def test_classifier_wrong_args_values1(self):
        with self.assertRaises(ValueError):
            ct.classifier('analysis.csv', first_col = -3, last_col = 1, batch_size= 2, epochs= 1)
    
    def test_classifier_wrong_args_values2(self):
        with self.assertRaises(ValueError):
            ct.classifier('analysis.csv', first_col = 3, last_col = -1, batch_size= 2, epochs= 1)
   
   #conversion tools
    def test_converter_merger_empty_file(self):
        conv.converter('empty_sample.root', overwrite_trees = True, overwrite_csv = True)
        self.assertTrue(filecmp.cmp('analysis.csv', './reference_files/ref_empty_analysis.csv', shallow=False), 'File do not match')
    def test_converter_merger(self):
        conv.converter('sample.root', overwrite_trees = True, overwrite_csv = True)
        self.assertTrue(filecmp.cmp('analysis.csv', './reference_files/ref_analysis.csv', shallow=False), 'File do not match')

    def test_add_label_column(self):
        conv.add_label_column(['test_ntuple.csv'])
        self.assertTrue(filecmp.cmp('test_l_ntuple.csv', './reference_files/ref_l_ntuple.csv', shallow=False), 'File do not match')
    def test_add_label_column_empty_f(self):
        conv.add_label_column(['test_empty_ntuple.csv'])
        self.assertTrue(filecmp.cmp('test_empty_l_ntuple.csv', './reference_files/ref_empty_l_ntuple.csv', shallow=False), 'File do not match')

    def test_root_tree_to_csv(self):
        conv.root_tree_to_csv('sample.root')
        self.assertTrue(filecmp.cmp('ntuple.csv', './reference_files/ref_ntuple.csv', shallow=False), 'File do not match')
    
    def test_root_tree_to_csv_empty(self):
        conv.root_tree_to_csv('empty_sample.root')
        self.assertTrue(filecmp.cmp('ntuple.csv', './reference_files/ref_empty_ntuple.csv', shallow=False), 'File do not match')
    

    def test_label_column_writer(self):
        conv.label_column_writer('test_ntuple.csv', 'l_test_ntuple.csv')
        self.assertTrue(filecmp.cmp('l_test_ntuple.csv', './reference_files/ref_l_ntuple.csv', shallow=False), 'File do not match')
    
    def test_label_column_writer_empty(self):
        conv.label_column_writer('test_empty_ntuple.csv', 'l_test__empty_ntuple.csv')
        self.assertTrue(filecmp.cmp('l_test_empty_ntuple.csv', './reference_files/ref_empty_l_ntuple.csv', shallow=False), 'File do not match')
    
    def test_unroll_tree(self):
        sample = uproot.open('sample.root')
        tree = sample.keys()
        tree_n = conv.get_tree_names(sample, tree)
        for t in tree_n:
            conv.unroll_tree(sample, t, 'test_output.csv')
        self.assertTrue(filecmp.cmp('test_output.csv', './reference_files/ref_ntuple.csv', shallow=False), 'File do not match')
    
    def test_unroll_tree_empty(self):
        sample = uproot.open('empty_sample.root')
        tree = sample.keys()
        tree_n = conv.get_tree_names(sample, tree)
        for t in tree_n:
            conv.unroll_tree(sample, t, 'test_empty_output.csv')
        self.assertTrue(filecmp.cmp('test_empty_output.csv', './reference_files/ref_empty_ntuple.csv', shallow=False), 'File do not match')

    def test_get_tree_names(self):
        sample = uproot.open('sample.root')
        trees = sample.keys()
        
        self.assertEqual(conv.get_tree_names(sample, trees), ['ntuple'])
    


    def test_is_signal(self):
        self.assertTrue(conv.is_signal('is_signal'), 'Error')
    def test_is_signal(self):
        self.assertFalse(conv.is_signal('is_langis'), 'Error')


unittest.main(argv=[''], verbosity=2, exit=False)