# to use accented characters in the code
# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v06 (Python3)
# ===============================


#_______________________________________________________________________________
# Some Utility Functions
from pandas import read_csv, DataFrame
from numpy import array, set_printoptions
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import Orange as DM
def my_print( aStr ):
   separator = lambda x: "_" * len( x )
   print( separator( aStr ) )
   print( aStr )



def load(fileName):
   try:
      dataset = DM.data.Table(fileName)

   except:
      my_print("--->>> error - can not open the file: %s" % fileName)
      exit()
   return dataset


def show_data(data, numFirstRows=10):
   firstRows = data[0:numFirstRows]
   set_printoptions(precision=3)
   print(">> summarized data (max = {n:d} instances)".format(n=numFirstRows), \
         firstRows, sep="\n")


def dataPrep2(data):
   newList = []
   categories = [3, 5, 6, 7, 8]
   X = data[:, 0:-1]
   yy = data[:, 9]
   for x in range(len(data)):
      thisList = []
      for y in categories:
         thisList.append(str(X[x][y]))

      thisList.append(str(yy[x][0]))
      newList.append(thisList)
   result = np.array(newList)
   result = DataFrame(result)
   result.columns = ["age", "tear_rate_name", "myope", "astigmatic", "hypermetrope", "y"]
   return result


def dataPrep2(data):
   orangesplitTrain=data[:13]
   orangesplitTest = data[13:]
   return orangesplitTrain,orangesplitTest

def dataPrep(data):
   cut = int(len(data)*0.85)
   orangesplitTrain=data[:cut]
   orangesplitTest = data[cut:]
   return orangesplitTrain,orangesplitTest

def plot_my_graph(y_test, y_predict,label, f_classifier):
    conf_matrix = confusion_matrix(y_test, y_predict, labels=label)
    #
    # Print the confusion matrix using Matplotlib
    #
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    ax.xaxis.set_ticklabels(np.hstack(([''],label)))
    ax.yaxis.set_ticklabels(np.hstack(([''],label)))
    plt.xlabel('y_predict', fontsize=18)
    plt.ylabel('y_test', fontsize=18)
    plt.title('Matrix de confusao\n' + f_classifier, fontsize=18)
    plt.show()