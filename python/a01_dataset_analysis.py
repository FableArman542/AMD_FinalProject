# to use accented characters in the code
# -*- coding: cp1252 -*-
# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v07 (Python3)
# ===============================


#__________________________________________
# Orange Documentation:
# http://docs.orange.biolab.si
#
# Orange Reference Manual:
# http://docs.orange.biolab.si/3/data-mining-library/#reference
#
# Tutorial:
# http://docs.orange.biolab.si/3/data-mining-library/#tutorial
#
# details about data (attribute+class) characterization:
# http://docs.orange.biolab.si/3/data-mining-library/tutorial/data.html#data-input
#__________________________________________

#_______________________________________________________________________________
# Modules to Evaluate
import sys
from u01_util import my_print
import Orange as DM
import numpy as np



#_______________________________________________________________________________
# Auxiliary Functions
#_______________________________________________________________________________
# load the dataset
def load( fileName ):
   try:
      dataset = DM.data.Table( fileName )
   except:
      my_print( "--->>> error - can not open the file: %s" % fileName )
      exit()
   return dataset



#_______________________________________________________________________________
# get the variable dataset-structure given a string with its name
def get_variableFrom_str( dataset, str_name ):
   variable_list = dataset.domain.variables
   for variable in variable_list:
      if( variable.name == str_name ): return variable
   my_print( ">>error>> \"{}\" is not a variable name in dataset!".format( str_name ) )
   return None



#_______________________________________________________________________________
# Data Analysis Functions
#_______________________________________________________________________________
# percentage of missing values of each variable
def get_missingValuePercentage( dataset ):
   variable_list = dataset.domain.variables
   var_len = len( variable_list )
   missingValue_list = [0.0] * var_len
   for instance in dataset:
      for var_index in range( var_len ):
        if np.isnan( instance[ var_index ] ): missingValue_list[ var_index ] += 1
   missingValue_list = list( map( lambda x, dim = len( dataset ): x / dim*100.0, missingValue_list ) )
   return ( variable_list, missingValue_list )



#_______________________________________________________________________________
# contingencyMatrix; i.e., the joint-frequency table
# this implementation does not account for missing-values
# (i.e., missing-values are not included in the variables-domain)
# M(row, column)
def get_contingencyMatrix( dataset, rowVar, colVar ):
   if( isinstance( rowVar, str ) ): rowVar = get_variableFrom_str( dataset, rowVar )
   if( isinstance( colVar, str ) ): colVar = get_variableFrom_str( dataset, colVar )
   if( not (rowVar and colVar) ): return ( [], [], None )
   if( not (rowVar.is_discrete and colVar.is_discrete) ):
      my_print( ">>error>> variables are expected to be discrete" )
      return ( [], [], None )
   
   rowDomain, colDomain = rowVar.values, colVar.values
   len_rowDomain, len_colDomain = len( rowDomain ), len( colDomain )
   contingencyMatrix = np.zeros( (len_rowDomain, len_colDomain) )
   for instance in dataset:
      rowValue, colValue = instance[rowVar], instance[colVar]
      if( np.isnan( rowValue ) or np.isnan( colValue ) ): continue
      
      rowIndex, colIndex = rowDomain.index(rowValue), colDomain.index( colValue )
      contingencyMatrix[ rowIndex, colIndex ] += 1
   return ( rowDomain, colDomain, contingencyMatrix )

   

#_______________________________________________________________________________
# P( H | E )
# H means Hypothesis, E means Evidence
# a frequency approach
def get_conditionalProbability( dataset, H, E ):
   if( isinstance( H, str ) ): H = get_variableFrom_str( dataset, H )
   if( isinstance( E, str ) ): E = get_variableFrom_str( dataset, E )
   if( not (H and E) ): return ( [], [], None )
   ( rowDomain, colDomain, cMatrix ) = get_contingencyMatrix( dataset, H, E )

   len_rowDomain, len_colDomain = len( rowDomain ), len( colDomain )
   E_marginal = np.zeros( len_colDomain )
   for col in range(len_colDomain): E_marginal[col] = sum( cMatrix[:, col] )
   
   for row in range(len_rowDomain):
      for col in range(len_colDomain):
         cMatrix[row, col] = cMatrix[row, col] / E_marginal[col]
   return ( rowDomain, colDomain, cMatrix )



#_______________________________________________________________________________
# 1R Related Functions
#_______________________________________________________________________________
# error matrix for a given feature and considering the datatset class
def get_errorMatrix( dataset, feature ):
   if( isinstance( feature, str ) ): feature = get_variableFrom_str( dataset, feature )
   the_class = dataset.domain.class_var
   ( rowDomain, colDomain, cMatrix ) = get_conditionalProbability( dataset, the_class, feature )
   if( not (rowDomain or colDomain) ): return ( [], [], None )

   errorMatrix = 1 - cMatrix
   return ( rowDomain, colDomain, errorMatrix )
  


#_______________________________________________________________________________
# Data Presentation Functions
#_______________________________________________________________________________
# show contingency matrix
def show_contingencyMatrix( dataset, rowVar, colVar ):
   my_print( "{} & {} :".format( rowVar.name, colVar.name ) )
   ( rowDomain, colDomain, cMatrix ) = get_contingencyMatrix( dataset, rowVar, colVar )  
   print( cMatrix )
   for row in rowDomain:
      showStr = "<" + row + "> "
      for col in colDomain:
         showStr += col + ": " + str(cMatrix[rowDomain.index(row), colDomain.index( col )]) + " | "
      print( showStr )
   return ( rowDomain, colDomain, cMatrix )



#_______________________________________________________________________________
# show all contingency matrix
# - for each feature and the class
def showAll_contingencyMatrix( dataset ):
   feature_list, the_class = dataset.domain.attributes, dataset.domain.class_var
   for feature in feature_list:
      show_contingencyMatrix( dataset, feature, the_class )
      print()



#_______________________________________________________________________________
# P( H | E )
# show matriz and textual description
def show_conditionalProbability( dataset, H, E ):
   ( rowDomain, colDomain, cMatrix ) = get_conditionalProbability( dataset, H, E )
   print( cMatrix )
   print()

   for h in rowDomain:
      for e in colDomain:
         rowIndex, colIndex = rowDomain.index( h ), colDomain.index( e )
         P_h_e = cMatrix[ rowIndex, colIndex ]
         print( "  P({} | {}) = {:.3f}".format( h, e, P_h_e ) )




#_______________________________________________________________________________
# implementation of some test cases
def test():
   #fileName = "./_dataset/adult_sample"
   # fileName = "./_dataset/lenses"
   #fileName = "./_dataset/lenses_copy"
   # fileName = "./_dataset/lenses_with_missingValues"
   fileName = "./_dataset/lenses_fromLecture"
   dataset = load( fileName )

   # print()
   # aStr = ">> Percentage of missing values per variable <<"
   # my_print( aStr )
   # ( variable_list, missingValue_list ) = get_missingValuePercentage( dataset )
   # for i in range( len( variable_list ) ):
   #     print( "%4.1f%s %s" % ( missingValue_list[ i ], '%', variable_list[ i ].name ) )

   # print()
   # aStr = ">> Contingency Matrix <<"
   # my_print( aStr )
   # showAll_contingencyMatrix( dataset )

   # print()
   # H = "lenses"  #"capital-gain"
   # E = "age" #"y"
   # aStr = ">> P( %s | %s ) <<" % ( H, E )
   # my_print( aStr )
   # show_conditionalProbability( dataset, H, E )

   # print()
   # H = "lenses" #"native-country"
   # E =    "prescription" #"y"
   # aStr = ">> P( %s | %s ) <<" % ( H, E )
   # my_print( aStr )
   # show_conditionalProbability( dataset, H, E )

   print()
   print("1R Here")
   the_feature = "prescription" #"age"
   the_features = ["prescription", "age", "tear_rate"]

   contasFinais = []

   for feat in the_features:
      aStr = "(1R-approach) >>Error Matrix>> %s & %s <<" % ( feat, dataset.domain.class_var )
      my_print( aStr )
      ( classDomain, featureDomain, errorMatrix ) = get_errorMatrix( dataset, feat )
      if( not (classDomain or featureDomain) ): return
       
      print("___________________________________")
       
      print("___________________________________")
       
      print( classDomain )
      print( featureDomain )
      print( errorMatrix )
      print()
       
      attrib = ""
      minErrorAttrib = 1000
      minErrors = []
       
      feature_list, the_class = dataset.domain.attributes, dataset.domain.class_var
      featForContingency = None
      for i in feature_list:
         if (i.name == feat):
            featForContingency = i
      print("-->> Contingency Matrix")
      ( rowDomain, colDomain, cMatrix ) = get_contingencyMatrix( dataset, featForContingency, the_class )
       
      print(rowDomain)
      print(cMatrix)
      total = np.sum(cMatrix)
      print("total: ", total)
      
       
       
      print( "-->> so, the rule, and error, for the {} feature are:".format(feat) )
      print()
      for feature in range(len(featureDomain)):
         errorFeature = errorMatrix[:, feature]
         errorMin = min( errorFeature )
         errorMinIndex = errorFeature.tolist().index( errorMin )
         featureValue = featureDomain[feature]
         classValue = classDomain[errorMinIndex]
         showStr = "(" + feat + ", " + featureValue + ", " + classValue + ") : "
         print( showStr + "{:.3f}".format( errorMin ) )
         minErrors.append([featureValue ,errorMin, classValue])

      count = 0
      for i in range(len(rowDomain)):
         soma = np.sum(cMatrix[i])
         erro = minErrors[i][1]
         num = erro * soma
         # print(str(rowDomain[i]) + ", " + str(soma) + ", " + str(erro))
         # print(str(num))
         count += num
         # print("_________________")
      print()
      minErrors = np.array(minErrors)
      
      contaFinal = count/total
      contasFinais.append([contaFinal, minErrors, feat])
      print("Conta final: " + str(contaFinal))
      print(minErrors)
   
   print("________________________")
   print("Resultado 1R: ")
   contasFinais = np.array(contasFinais, dtype=object)
   
   contador = 1000
   idx = 0
   for i in range(len(contasFinais)):
       if contasFinais[i][0] < contador:
           contador = contasFinais[i][0]
           idx = i
   contasFinais = contasFinais[idx]
   
   print("Melhor Atributo =", contasFinais[2])
   print("Erro =", contasFinais[0])
   print("Classificao :")
   
   for i in contasFinais[1]:
       print("-> ", i[0], ":", i[2])
   #print("Classificacao =", contasFinais[1])       


#_______________________________________________________________________________
# the main of this module (in case this module is imported from another module)
if __name__=="__main__":
   test()












