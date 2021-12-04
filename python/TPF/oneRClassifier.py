import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings( "ignore", category=UndefinedMetricWarning ) #use "always" or "ignore"
class oneRClassifier:

    def __init__(self):
        print("oneRClassifierInit")


    def predict(self, X_test ):
        print("oneRClassifierpredict")
        my_result=[]
        indexSelected=-1
        for i in range(len(X_test[0].domain.variables)):
            if (str(X_test[0].domain[i]) == self.atributo):
                indexSelected=i
                break
        for consulta in X_test:
            for classe in self.result:
                if(str(classe[0])==str(consulta[indexSelected])):
                    my_result.append(str(classe[2]))
                    break


        return my_result

    def fit(self,dataset):

        the_features = ["tear_rate_name", "myope", "astigmatic", "hypermetrope", "age"]

        contasFinais = []

        for feat in the_features:

            (classDomain, featureDomain, errorMatrix) = self.get_errorMatrix(dataset, feat)
            if (not (classDomain or featureDomain)): return


            attrib = ""
            minErrorAttrib = 1000
            minErrors = []

            feature_list, the_class = dataset.domain.attributes, dataset.domain.class_var
            featForContingency = None
            for i in feature_list:
                if (i.name == feat):
                    featForContingency = i

            (rowDomain, colDomain, cMatrix) = self.get_contingencyMatrix(dataset, featForContingency, the_class)


            total = np.sum(cMatrix)

            for feature in range(len(featureDomain)):
                errorFeature = errorMatrix[:, feature]
                errorMin = min(errorFeature)
                errorMinIndex = errorFeature.tolist().index(errorMin)
                featureValue = featureDomain[feature]
                classValue = classDomain[errorMinIndex]
                showStr = "(" + feat + ", " + featureValue + ", " + classValue + ") : "

                minErrors.append([featureValue, errorMin, classValue])

            count = 0
            for i in range(len(rowDomain)):
                soma = np.sum(cMatrix[i])
                erro = minErrors[i][1]
                num = erro * soma
                # print(str(rowDomain[i]) + ", " + str(soma) + ", " + str(erro))
                # print(str(num))
                count += num
                # print("_________________")

            minErrors = np.array(minErrors)

            contaFinal = count / total
            contasFinais.append([contaFinal, minErrors, feat])



        contasFinais = np.array(contasFinais, dtype=object)

        contador = 1000
        idx = 0
        for i in range(len(contasFinais)):
            if contasFinais[i][0] < contador:
                contador = contasFinais[i][0]
                idx = i
        contasFinais = contasFinais[idx]

        self.atributo=contasFinais[2]
        self.result=contasFinais[1]
        # print("Classificacao =", contasFinais[1])

    def fit2(self,dataset):
        print("oneRClassifier")
        print("Created1R")
        print()
        print("1R Here")
        the_features = ["tear_rate_name", "myope", "astigmatic", "hypermetrope", "age"]

        contasFinais = []

        for feat in the_features:
            aStr = "(1R-approach) >>Error Matrix>> %s & %s <<" % (feat, dataset.domain.class_var)
            self.my_print(aStr)
            (classDomain, featureDomain, errorMatrix) = self.get_errorMatrix(dataset, feat)
            if (not (classDomain or featureDomain)): return

            print("___________________________________")

            print("___________________________________")

            print(classDomain)
            print(featureDomain)
            print(errorMatrix)
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
            (rowDomain, colDomain, cMatrix) = self.get_contingencyMatrix(dataset, featForContingency, the_class)

            print(rowDomain)
            print(cMatrix)
            total = np.sum(cMatrix)
            print("total: ", total)

            print("-->> so, the rule, and error, for the {} feature are:".format(feat))
            print()
            for feature in range(len(featureDomain)):
                errorFeature = errorMatrix[:, feature]
                errorMin = min(errorFeature)
                errorMinIndex = errorFeature.tolist().index(errorMin)
                featureValue = featureDomain[feature]
                classValue = classDomain[errorMinIndex]
                showStr = "(" + feat + ", " + featureValue + ", " + classValue + ") : "
                print(showStr + "{:.3f}".format(errorMin))
                minErrors.append([featureValue, errorMin, classValue])

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

            contaFinal = count / total
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

        self.atributo=contasFinais[2]
        self.result=contasFinais[1]
        # print("Classificacao =", contasFinais[1])
    def my_print(self,aStr):
        separator = lambda x: "_" * len(x)
        print(separator(aStr))
        print(aStr)


    def get_errorMatrix( self,dataset, feature ):
       if( isinstance( feature, str ) ): feature = self.get_variableFrom_str( dataset, feature )
       the_class = dataset.domain.class_var
       ( rowDomain, colDomain, cMatrix ) = self.get_conditionalProbability( dataset, the_class, feature )
       if( not (rowDomain or colDomain) ): return ( [], [], None )

       errorMatrix = 1 - cMatrix
       return ( rowDomain, colDomain, errorMatrix )

    def get_variableFrom_str(self, dataset, str_name ):
       variable_list = dataset.domain.variables
       for variable in variable_list:
          if( variable.name == str_name ): return variable
       self.my_print( ">>error>> \"{}\" is not a variable name in dataset!".format( str_name ) )
       return None

    def get_contingencyMatrix(self,dataset, rowVar, colVar):
        if (isinstance(rowVar, str)): rowVar = self.get_variableFrom_str(dataset, rowVar)
        if (isinstance(colVar, str)): colVar = self.get_variableFrom_str(dataset, colVar)
        if (not (rowVar and colVar)): return ([], [], None)
        if (not (rowVar.is_discrete and colVar.is_discrete)):
            self.my_print(">>error>> variables are expected to be discrete")
            return ([], [], None)

        rowDomain, colDomain = rowVar.values, colVar.values
        len_rowDomain, len_colDomain = len(rowDomain), len(colDomain)
        contingencyMatrix = np.zeros((len_rowDomain, len_colDomain))
        for instance in dataset:
            rowValue, colValue = instance[rowVar], instance[colVar]
            if (np.isnan(rowValue) or np.isnan(colValue)): continue

            rowIndex, colIndex = rowDomain.index(rowValue), colDomain.index(colValue)
            contingencyMatrix[rowIndex, colIndex] += 1
        return (rowDomain, colDomain, contingencyMatrix)

    def get_conditionalProbability(self,dataset, H, E):
        if (isinstance(H, str)): H = self.get_variableFrom_str(dataset, H)
        if (isinstance(E, str)): E = self.get_variableFrom_str(dataset, E)
        if (not (H and E)): return ([], [], None)
        (rowDomain, colDomain, cMatrix) = self.get_contingencyMatrix(dataset, H, E)

        len_rowDomain, len_colDomain = len(rowDomain), len(colDomain)
        E_marginal = np.zeros(len_colDomain)
        for col in range(len_colDomain): E_marginal[col] = sum(cMatrix[:, col])

        for row in range(len_rowDomain):
            for col in range(len_colDomain):
                cMatrix[row, col] = cMatrix[row, col] / E_marginal[col]
        return (rowDomain, colDomain, cMatrix)