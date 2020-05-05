#!/usr/bin/env python
# coding: utf-8


import os
import tempfile
import unittest
import numpy as np
from pandas import read_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC, Accuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler



def plot_metrics(history):
    """
    This function plots performance of classifier considering different metrics.
    """
    metrics =  ['loss', 'AUC', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()



def plot_confusion_matrix(y_true, y_pred, classes,
                          class_names = None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    if class_names is None:
        x_labels = y_labels = classes
    else:
        x_labels = y_labels = class_names
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=x_labels, yticklabels=y_labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def make_model(metrics = None, output_bias = None):   
    """
    This function defines the classifier model
    """
    if metrics == None:
        print('Found no metric to use. Add at least one metric to continue')
        raise ValueError
        
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)    
    #Initialising NN
    model = Sequential()

    #First layer
    model.add(Dense(8, activation='relu', input_shape=(13,)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))

    #Second layer
    model.add(Dense(12, activation='relu'))
   # model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='sigmoid'))


    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(lr= 0.001),
                  metrics=metrics
                 )
    return model



def load_dataset(dataset_name, start_col, end_col):
    """
    This method loads the dataset if it exists
    """
    if not os.path.isfile(dataset_name):
        raise FileNotFoundError
    if start_col < 0 or start_col > end_col or end_col<0:
        print("Wrong column initialization...")
        raise ValueError
        return

    info = {
        "dataset_name" : dataset_name,
        "first_column"     : start_col,
        "last_column"      : end_col
    }
    
    return info



def classifier(first_col, last_col, epochs, batch_size, seed = 13):

    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']    #plot configurations


    if epochs<=0 or batch_size <=0:
        print ('Wrong parameters given: check epochs and batch_size')
        raise ValueError
        return
    np.random.seed(seed) 
    METRICS = [                                                  
                AUC(name = 'AUC'),
                Accuracy(name = 'accuracy'),
                Precision(name = 'precision'),
                Recall(name = 'recall')    
               ]                                        #metrics: modify here to add or remove metric


    dset = load_dataset("analysis.csv", start_col=first_col, end_col=last_col)    
    dataframe = read_csv(dset["dataset_name"], header=0)
    dataset = dataframe.values


    X = dataset[:,int(dset["first_column"] ):int(dset["last_column"])].astype(float)   
    Y = dataset[:,int(dset["last_column"])] #label column (15th) into Y 

    #Encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    #One-hot encoding
    transformed_Y = to_categorical(encoded_Y)
    
    bkg, sgn = np.bincount(encoded_Y)

    total =  bkg + sgn
    print('Samples:\n Total: {}\n Background: {} \n Signal: {} \n Signal samples are {:.2f}% of the total'.format(total, bkg, sgn, 100*sgn/total))

    weight_for_0 = (1 / bkg)*(total)/2.0
    weight_for_1 = (1 / sgn)*(total)/2.0


    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        transformed_Y,
                                                        test_size=0.25,
                                                        random_state=seed,
                                                        shuffle = True)

    scaler = StandardScaler()
    X_train =scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print('Training labels shape:', Y_train.shape)
    print('Validation labels shape:', Y_test.shape)


    print('Training features shape:', X_train.shape)
    print('Validation features shape:', X_test.shape)

    EPOCHS = epochs
    BATCH_SIZE = batch_size
    val_data = (X_test,Y_test)
    #checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_precision', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        verbose = 1,
        patience = 200,
        mode = 'min',
        restore_best_weights = True)

    model = make_model(metrics = METRICS)
    model.summary()
    
    initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
    model.save_weights(initial_weights)
    
    model.load_weights(initial_weights)
    
    
    history=model.fit(X_train,
                      Y_train,
                      epochs = epochs,
                      shuffle = True,
                      validation_data=val_data,
                      #validation_freq=5,
                      callbacks = [early_stopping],
                      batch_size = batch_size,
                      class_weight=class_weight
                     )
    
    model.save_weights(initial_weights)
    plot_metrics(history)
    
    
    
    
    
    #compute predictions
    predictions = model.predict(X_test)

    y_pred = np.array([np.argmax(probas) for probas in predictions])
    y_test = np.array([np.argmax(label) for label in Y_test])

    classes = unique_labels(y_test, y_pred)
    class_names = unique_labels(Y)

    #confusion matrix

    plot_confusion_matrix(y_test, y_pred, classes, class_names)





class TestNotebook(unittest.TestCase):
    def test_load_dataset_f_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset('a.csv', 0, 3)
    def test_load_dataset_column_err(self):
        with self.assertRaises(ValueError):
            load_dataset('analysis.csv', -1, -12)
    def test_make_model_no_metric(self):
        with self.assertRaises(ValueError):
            make_model()
    def test_classifier_wrong_args_values1(self):
        with self.assertRaises(ValueError):
            classifier(first_col = -3, last_col = 10, batch_size= 2, epochs= 1)
    def test_classifier_wrong_args_values2(self):
        with self.assertRaises(ValueError):
            classifier(first_col = 3, last_col = -10, batch_size= 2, epochs= 1)

#unittest.main(argv=[''], verbosity=2, exit=False)


# In[ ]:




