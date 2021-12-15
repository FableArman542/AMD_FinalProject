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


def dataPrep(data):
   orangesplitTrain=data[:13]
   orangesplitTest = data[13:]
   return orangesplitTrain,orangesplitTest