# Project for the Software and Computing exam.

## Conversion of a .root file (physics data) into a .csv file for the use in a TF NeuralNet.

The software developed for this project is basically divided in two main algorithms: one for the conversion of the data from **.root** to **.csv** file using the module "***uproot***" which can be found in the file "*data_converter.ipynb*" and one for the implementation of the neural network using **Keras/Tensorflow** modules which is in the file "*signal_classifier.ipynb*".

## Documentation.

### **data_converter.ipynb**:

- get_file_name(fname = "analysis.root"):
	Takes a string as an argument and checks if a file named as the argument string:
		1. **exists**;
		2. **is located in the same folder** as the *data_converter.ipynb* file;
		3. **ends with the extension *.root***.
	If the conditions are satisfied the function **returns** the file name as a string.
