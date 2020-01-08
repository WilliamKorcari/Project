# Project for the Software and Computing exam.

## Conversion of a .root file (physics data) into a .csv file for the use in a TF NeuralNet.

The software developed for this project is basically divided in two main algorithms: one for the conversion of the data from **.root** to **.csv** file using the module "***uproot***" which can be found in the file "*data_converter.ipynb*" and one for the implementation of the neural network using **Keras/Tensorflow** modules which is in the file "*signal_classifier.ipynb*".

## Documentation.

### **data_converter.ipynb**:

- get_file_name(fname = "analysis.root"):

	Takes a string as an argument and checks if a file named as the argument string:

		1. exists;

		2. is located in the same folder as the 'data_converter.ipynb' file;

		3. ends with the extension '.root'.

	If the conditions are satisfied the function **returns** the file name as a string.

- get_tree_names(f_name): 

	Takes as an argument a file name (*string*). Looks for root trees inside that file and **returns** a list of the cleaned up tree names.
	***It is strongly raccomended to use what _get_file_name()_ returns in order to avoid errors***

- unroll_tree(file_name, ttree, of_name):

	- file_name: string that contains the name of the file to convert into *.csv* format file;
	
	- ttree: list of string representing the names of the trees inside the file to convert;
	
	- of_name: string that names the *.csv* file produced by this function.
	
This function is used inside root_tree_to_csv() (see next) function and so the arguments are initialized automatically inside that function. 
This method __returns__ a _.csv_ file that contains the converted data from a single tree of the initial _.root_ file.

- root_tree_to_csv(overwrite = False):

	This function uses the previous methods to convert every root tree inside the desired file in to a _.csv_ file.
Also checks if the resulting file from the operation already exists and overwrites them depending on how the __overwrite__ is set (__False__ by default).
	

label_column_writer(infile, outfile, fsignal = "signal_bbA_MA300tree.csv"):
