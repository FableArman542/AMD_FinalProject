from pandas import read_csv, DataFrame
from numpy import array, set_printoptions
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, \
                                    KFold, StratifiedKFold, \
                                    RepeatedKFold, RepeatedStratifiedKFold, \
                                    LeaveOneOut, LeavePOut, \
                                    cross_val_score
from my_split_and_eval import *
from pandas import DataFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from oneRClassifier import oneRClassifier
import numpy as np
import Orange as DM
from sklearn.exceptions import UndefinedMetricWarning
from u01_util import *
import warnings
warnings.filterwarnings( "ignore", category=UndefinedMetricWarning ) #use "always" or "ignore"


def main():

    fileName = "dataset_long_name_EXPORTED.tab"
    dataset1 = load(fileName)
    #print(dataset)
    #show_data(dataset)
    print("classifier:", "oneRClassifier")
    classifier = oneRClassifier()
    a,y_test, y_predict =score_recipe2(classifier, dataset1, list_score_metric,[ var.name for var in dataset1.domain.attributes ])
    plot_my_graph(y_test, y_predict,['EDIBLE', 'POISONOUS'],"oneRClassifier")
    print(2 * "\n" + "<<< ----- >>>" + 2 * "\n")

    results=classifier.fit(dataset1,[ var.name for var in dataset1.domain.attributes ])

    print("Resultados:")
    for result in results:
        print(result)

#______________
# score metrics
# <your-code-here>
list_score_metric = \
  [
    (accuracy_score, {}),
    (precision_score, {"average":"weighted"}), #macro #micro #weighted
    (recall_score, {"average":"weighted"}), #macro #micro #weighted
    (f1_score, {"average":"weighted"}), #macro #micro #weighted
    #(cohen_kappa_score, {}),
  ]
#______________________________________________________________________________
# The "main" of this module (in case it was not loaded from another module)
if __name__ == "__main__": main()


