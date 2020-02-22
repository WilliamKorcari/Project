# Project for the Software and Computing exam.

## Conversion of a .root file (physics data) into a .csv file for the use in a TF NeuralNet.

The software developed for this project is basically divided in two main algorithms: one for the conversion of the data from **.root** to **.csv** file using the module "***uproot***" which can be found in the file "*data_converter.ipynb*" and one for the implementation of the neural network using **Keras/Tensorflow** modules which is in the file "*signal_classifier.ipynb*".

## Documentation.

### How to:
Start by using the data converter in order to produce the *.csv* on which classification is doable. 

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

	This function takes all file that have the 'label column' and merges them all together.

#### How to use:

- initialize variable "FILE" with _get_file_name(fname)_ method;

- initialize variable "CSV_FILES" with _root_tree_to_csv(file = FILE)_ method;

- (optional): initialize variable "LABELED_CSV_FILES" with _add_label_column(CSV_FILES)_ method;

- call _file_merger()_.


### **signal_classifier.ipynb**:

This file is an attempt of signal classification by the application of a very basic neural network model.
Very few methods are defined whithin this file.

- plot_metrics(history): plots different variables after performing training of the aNN.

- plot_confusion_matrix(y_true, y_pred, classes,
                          class_names = None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greens): plots the confusion Matrix. Can normalize results by setting 'normalize = True'.

- make_model(metrics = METRICS, output_bias = None): defines the aNN model. 

        output_bias can be set in order to try to improve training results.










