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

    fileName = "teste1.csv"
    dataset1 = load(fileName)
    #oneRClassifier().oneRClassifierTest(dataset1)
    dataset=dataset1[:,4:]

    print(dataset)
    dataset = DataFrame(dataset)


    dataset = DataFrame(dataset)
    show_data(dataset)


    for (f_tt_split, args_tt_split) in list_func_tt_split:
        (X, y, tt_split_indexes) = train_test_split_recipe(dataset,f_tt_split, *args_tt_split)
        show_function_name("train_test_split:", f_tt_split)
        show_train_test_split(X, y, tt_split_indexes, numFirstRows=10)

        for (f_classifier, args_classifier) in list_func_classifier:
            if(f_classifier.__name__=="oneRClassifier"):
                print("something")
                show_function_name("classifier:", f_classifier)
                classifier = oneRClassifier()
                score_recipe2(classifier, dataset1, list_score_metric)

            else:
                classifier = f_classifier(*args_classifier)
                show_function_name("classifier:", f_classifier)

                for (f_score, keyword_args_score) in list_score_metric:

                    score_all = score_recipe(classifier, X, y, tt_split_indexes,f_score, **keyword_args_score)
                    show_function_name("score_method:", f_score)
                    show_score(score_all)

        print(2 * "\n" + "<<< ----- >>>" + 2 * "\n")

list_func_tt_split = \
  [
    (holdout, (1.0/7.0, seed)),
    (stratified_holdout, (1.0/7.0, seed)),
    #(repeated_holdout, (1.0/3.0, 2, seed)),
    #(repeated_stratified_holdout, (1.0/3.0, 2, seed)),
    #(fold_split, (3, seed)),
    #(stratified_fold_split, (3, seed)),
    #(repeated_fold_split, (3, 2, seed)),
    #(repeated_stratified_fold_split, (3, 2, seed)),
    #(leave_one_out, ()),
    #(leave_p_out, (2, )),
    #(bootstrap_split_once, (seed, )),
    #(bootstrap_split_repeated, (2, seed))
  ]



#__________________________
# classification techniques
# <your-code-here>
list_func_classifier = \
  [
    (GaussianNB, ()),
    (DecisionTreeClassifier, ()),
    (oneRClassifier,())
  ]



#______________
# score metrics
# <your-code-here>
list_score_metric = \
  [
    (accuracy_score, {}),
    (precision_score, {"average":"weighted"}), #macro #micro #weighted
    (recall_score, {"average":"weighted"}), #macro #micro #weighted
    (f1_score, {"average":"weighted"}), #macro #micro #weighted
    (cohen_kappa_score, {}),
  ]
#______________________________________________________________________________
# The "main" of this module (in case it was not loaded from another module)
if __name__ == "__main__": main()


