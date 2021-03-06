#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import uproot
import csv
import os
import glob


def is_signal(file_name):
	"""
	If input string contains signal returns True. 
	"""
	if 'signal' in file_name:
		return True
	return False

def get_tree_names(file, trees):
    """
    file = open root file (by uproot.open(file))
    trees = trees contained in files (list)
    Returns list of cleaned tree names contained in file.
    """
  
    tree_names=[]

    print("\n Tree names successfully stored. They are: \n")
    for tree in trees:
        tree = str(tree)
        tree = tree[2:len(tree)-3]
        tree_names.append(tree)
        print(tree)
        print('\n')
        
    return tree_names



def unroll_tree(data, tree, of_name):
    """
    - data = open root file
    - tree: list of string representing the names of the trees inside the file to convert;

    - of_name: string that names the csv file produced by this function.

    This function is used inside root_tree_to_csv() (see next) function and so the arguments are initialized automatically inside that function. 
    This method returns a .csv file that contains the converted data from a single tree of the initial .root file. 
    """

    files_out= [] 
    names = data[tree].keys()


    out = pd.DataFrame.from_dict(data[tree].arrays(names), dtype= str)
    out.to_csv(of_name, index=False)
    files_out.append(of_name)
    print ('\nCreated csv file ' + of_name)




def root_tree_to_csv(file, overwrite = True):
    """
    - file: string. File to convert
    - overwrite: bool. True overwrites already existing file with same name.

    Converts input root file in to a csv file.
    """

    f = uproot.open(file)
    trees = f.keys()
    woods = get_tree_names(f, trees)
    

    files_out = []

    for tree in woods:    
        out_file_name = tree + '.csv'
        files_out.append(out_file_name)

        if os.path.isfile(out_file_name):
            if overwrite:
                print ("\nOverwriting tree {} in file {}".format(tree, f))
                unroll_tree(f, tree, out_file_name)                
            else:
                print( out_file_name + " already exist and will not be overwritten...")

        else: 
            print ("\nWriting tree {} in file {}".format(tree, f))
            unroll_tree(f, tree, out_file_name)

    return files_out



def label_column_writer(infile, outfile):
    """

    - infile: string. Sets the input file name;

    - outfile: string. Sets the output file name;
    
    Adds column with header 'Label' and fills rows with either 'signal' or background' depending on type of file

    """

    df = pd.read_csv(infile)
    ncols = len(df.columns)
    nrows = df.shape[0]

    if is_signal(infile):
        sgn = ["signal" for row in range(nrows)]
        df['label'] = sgn
    else:
        bkg = ["background" for row in range(nrows)]
        df['label'] = bkg
    df.to_csv(outfile, index=False)



def add_label_column(files = [], overwrite = True):
    """

    - files: list of file names;

    - overwrite: True/False. False by default.

    Takes a list of file and applies label_column_writer() method to each one. Overwrites, depending on set argument, if file already exists.
    returns list of string (names ofproduced files).
    
    """

    out_files = []
    print("The following files will be modified: \n")
    print (files)
    print('\n')
 
    for file in files:
        out_file = "l_"+ file
        out_files.append(out_file)
        if os.path.isfile(out_file):
            if overwrite:
                print( "\nOverwriting " + out_file) 
                label_column_writer(file, out_file)
            else: 
                print( out_file + " already exist and will not be overwritten..\n")
        else:
            print( "\nWriting " + out_file)
            label_column_writer(file, out_file)
    return out_files



def file_merger(infiles, file_out_name, overwrite = True):
    """
    - infiles = list of strings. Sets the name of file to merge.

    - file_out_name: string. Sets the name of the file produced as an outcome of the method;

    - overwrite: True/False (default False).

    This function takes all file given as input and merges them together into a unique csv file.


"""
    if os.path.isfile(file_out_name):
        if not overwrite: 
            print( file_out_name + " already exist and will not be overwritten...\n")
    
    
    
        if overwrite:
            
            print("Overwriting already existing file " + file_out_name)
            
            all_filenames = [f for f in infiles]
            combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], sort = False)
            
            
            combined_csv.to_csv( file_out_name, index=False, encoding='utf-8-sig')
    else:
        print('Writing {} file'.format(file_out_name))
        all_filenames = [f for f in infiles]
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], sort = False)
        combined_csv.to_csv( file_out_name, index=False, encoding='utf-8-sig')


def converter(f, label = True, out_f_name = 'analysis.csv', overwrite_trees = True, overwrite_csv = True):
    """
    Executes all the functions to convert the root file in to a csv file.
    - f: string indicating file to convert;
    - label = boolean. True produces files with label column;
    - out_f_name = string. Sets name of output file;
    - overwrite_trees = boolean. True overwrites tree files if file with the same name is found in dir;
    - overwrite_csv = boolean. True overwrites output file if file with the same name is found in dir.
    
    """
    
    
    if label:
        csv_files = add_label_column(root_tree_to_csv(file = f, overwrite = overwrite_trees), overwrite = overwrite_csv)
    else:
        csv_files = root_tree_to_csv(file = f, overwrite = overwrite_trees)

    for element in csv_files:
        print('Created ' + element)
    
    file_merger(infiles= csv_files, file_out_name = out_f_name, overwrite = overwrite_csv)





