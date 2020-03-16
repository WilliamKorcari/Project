# Project for the Software and Computing exam.

## Conversion of a .root file (physics data) into a .csv file for the use in a TF NeuralNet.

The software developed for this project is basically divided in two main algorithms: one for the conversion of the data from **.root** to **.csv** file using the module "***uproot***" which can be found in the file "*data_converter.ipynb*" and one for the implementation of the neural network using **Keras/Tensorflow** modules which is in the file "*signal_classifier.ipynb*".
The pourpose of the software is to classify signal and background starting from a '.root' file which contains so called *trees* for background and signal data. The implementation here given is good starting point to experiment with different model on the given data, but no good solution has already been found.

## Documentation.

### Prerequisites:
The project is written in Python 2.7 due to the ROOT library requirements.

If you do not have ROOT installed you can install it from the site making sure to get a version >= 6.12 or:

     - source root_install.sh

inside the main directory. That will install ROOT 6.18.04 in a directory called 'root' inside the project dir.

You also must install uproot library which can be obtained via pip command:

    - [sudo] pip install uproot [--user]
    
or via conda. 

List of other common python libraries needed to execute this software:
    1. Numpy
    2. Pandas
    3. Matplotlib
    4. csv
    5. unittest
All of this are easily obtained via pip or conda.



### Tutotrial:

To efficiently use this software one has to first execute the data_converter notebook giving as an input the '.root' dataset name that one wants to convert in to '.csv' as follows:

    - main('name_of_dataset.root', *options) 
    
After that in the signal_classifier notebook one can run the classification choosing number of epochs and batch size.
e.g.:
    
    - classifier(10, 1000)
    
Hypertuning has to be performed in the make_model method.

### **data_converter.ipynb**:

- get_file_name(fname = "analysis.root"):

	Takes a string as an argument (default "analysis.root"). Checks if a file named as the argument string:

		1. exists;

		2. is located in the same folder as the 'data_converter.ipynb' file;

		3. ends with the extension '.root'.

	If the conditions are satisfied the function **returns** the file name as a string.

- get_tree_names(f_name): 

	Takes as an argument a file name (*string*). Looks for root trees inside that file and **returns** a list of the cleaned up tree names.
	***It is strongly raccomended to use what _get_file_name()_ returns in order to avoid errors.***

- unroll_tree(file_name, ttree, of_name):

	- file_name: string that contains the name of the file to convert into *.csv* format file;
	
	- ttree: list of string representing the names of the trees inside the file to convert;
	
	- of_name: string that names the *.csv* file produced by this function.
	
This function is used inside root_tree_to_csv() (see next) function and so the arguments are initialized automatically inside that function. 
This method __returns__ a _.csv_ file that contains the converted data from a single tree of the initial _.root_ file.

- root_tree_to_csv(overwrite = False):

	This function uses the previous methods to convert every root tree inside the desired file in to a _.csv_ file.
Also checks if the resulting file from the operation already exists and overwrites them depending on how the __overwrite__ is set (__False__ by default).
**returns** list of string (names ofproduced files).
	

- label_column_writer(infile, outfile, fsignal = "signal_bbA_MA300tree.csv"):

	- infile: string. Sets the input file name;

	- outfile: string. Sets the output file name;

	- fsignal: A string used to identify the signal file. Each file name is compared to that string, if the strings match the algorithm identifies the file as a signal file.

	This method manipulates the input file in order to add a *label* column and to fill in values ('background' or 'signal' for each entry of the file.

- add_label_column(f_to_modify = [], overwrite = False):

	- f_to_modify: list of file names of files (.csv format);
	
	- overwrite: True/False. False by default.

Takes a list of file and applies label_column_writer() method to each one. Overwrites, depending on set argument, if file already exists.
**returns** list of string (names ofproduced files).

- file_merger(infile_names, outfile_name, overwrite = False):
	
	- infile_names = list of strings. Sets the name of file to merge.
	
	- outfile_name: string. Sets the name of the file produced as an outcome of the method;

	- overwrite: True/False (default False).

	This function takes all file given as input and merges them together into a unique csv file.



### **signal_classifier.ipynb**:

This file is an attempt of signal classification by the application of a very basic neural network model (by default).
Very few methods are defined whithin this file.

- plot_metrics(history): plots different variables after performing training of the aNN.

- plot_confusion_matrix(y_true, y_pred, classes,
                          class_names = None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greens): plots the confusion Matrix. Can normalize results by setting 'normalize = True'.

- make_model(metrics = METRICS, output_bias = None): defines the aNN model.
    1. metrics: list of metrics to be used for classification
    2. output_bias: bias to apply (hypertuning)

- classifier(first_col, last_col, epochs, batch_size, seed): 
    1. first_col: first column to consider for the dataset
    2. last_col: last column to consider for the dataset (tipically label column)
    3. epochs: positive integer number.
    4. batch_size: positive integer number. 
    5. seed: positive integer number.
    
    Performs classification and plots metrics and confusion matrix.
                                        

        output_bias can be set in order to try to improve training results.










