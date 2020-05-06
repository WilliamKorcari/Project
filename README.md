# Project for the Software and Computing exam.

## Conversion of a .root file (physics data) into a .csv file for the use in a TF NeuralNet.

The software developed for this project is basically divided in two main algorithms: one for the conversion of the data from **.root** to **.csv** file using the module "***uproot***" which can be found in the file "*conversion_tools.py*" and one for the implementation of the neural network using **Keras/Tensorflow** modules which is in the file "*classification_tools.py*".
The pourpose of the software is to classify signal and background starting from a '.root' file which contains so called *trees* for background and signal data. The implementation here given is good starting point to experiment with different model on the given data, but no good solution has already been found.

## Documentation.

### Prerequisites:
The project is written in Python3.

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
    6. tensorflow
    7. keras
    
All of this are easily obtained via pip or conda.



### Tutotrial:

To efficiently use this software one has to first execute the main.ipynb notebook giving as an input the '.root' dataset name that one wants to convert in to '.csv' as showed in the mentioned notebook:

    - conv.converter('name_of_dataset.root', *options) 
    
After that one can execute the classification by calling the classification function conteined in the classification_tools.py,
e.g.:
    
    - ct.classifier('file.csv', first col, last col, n_epochs, batch_size, seed)
    
Hypertuning has to be performed in the make_model method inside classification_tools.py.

