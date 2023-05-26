###En este archivo encontraras todas las funciones desarrolladas para la creación del modelo

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from IPython.core.display import display, HTML,display_html
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.model_selection import cross_val_score


class analytics:

    def __init__(self, df): #Parametros de la clase
        self.df = df
        
    # La función caract_num se encarga de generar un histograma con su correspondiente box plot con el fin de tener una visión rapida de cada una de las variables númericas

    def caract_num(self,col_list):
    # Calcula los histogramas de cada variable

        display(HTML("<h2>Validación de distribución " + str(list(col_list)) + "<h2/>"))

        plt.figure(figsize=(30,6))

        for i in range(len(col_list)):
            
            plt.subplot(1, len(col_list), i + 1)
            self.df[col_list[i]].plot(kind='hist' ,title = col_list[i],bins=30)
            #plt.show()

        # Calcula los boxplot de cada variable
        plt.figure(figsize=(30,6))

        for i in range(len(col_list)):
            
            plt.subplot(1, len(col_list), i + 1)
            self.df.boxplot(column=[col_list[i]])

        plt.show()

    
    # La función caract_num se encarga de generar un diagrama de barrascon el fin de tener una visión rapida de cada una de las variables string

    def caract_obj(self,col_list):

        # Calculo de frecuencias de cada variable
        display(HTML("<h2>Validación de frecuencias " + str(list(col_list)) + "<h2/>"))
        plt.figure(figsize=(20,6))

        for i in range(len(col_list)):
            plt.subplot(1, len(col_list), i + 1)
            self.df[col_list[i]].value_counts().plot(kind='bar',title = col_list[i])
        plt.show()



class modeling:

    def __init__(self, df,balanceo = False, hiperparametros = False): #Parametros de la clase
        self.df = df
        self.balanceo = balanceo
        self.hiperparametros = hiperparametros
        


    def train_test(self,var_obj,id_var,random_state = 42, var_x = False): #Se deja la semilla fija

        if var_x != False:
            
            # Definimos los datos entre variables y target
            x = self.df.drop([var_obj,id_var],axis=1)
            x = x[var_x]
            y = self.df[var_obj]

        else:
            
            # Definimos los datos entre variables y target
            x = self.df.drop([var_obj,id_var],axis=1)
            y = self.df[var_obj]

        # Dividimos los datos en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=random_state)

#Esta función se encarga de estimar los modelos de regresión logística según los parametros establecidos en el objeto inicial
    
    def logistic_regression(self): #Entrena un modelo de regresión logistica

        if self.balanceo == True:

            print("El modelo realizo el balanceo de la muestra")
            print("La porporción inicial es de:", '{:.1%}'.format(self.y_train.sum()/self.X_train.shape[0]))
            
            # Realizar oversampling con SMOTE
            smote = SMOTE(random_state=0)
            X_train_resampled, y_train_resampled = smote.fit_resample(self.X_train, self.y_train)

            self.X_train = X_train_resampled
            self.y_train = y_train_resampled

            print("La porporción posterior al OV es de:", '{:.1%}'.format(y_train_resampled.sum()/X_train_resampled.shape[0]))


        if self.hiperparametros == False:
        
            lr = LogisticRegression()
            lr.fit(self.X_train, self.y_train)    
        
            return lr
        
        else:          

            # Se definen los hiperparametros a mejorar
            params = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['saga', 'liblinear']}
            
            # Se crea un objeto de regresión logística y objeto GridSearchCV
            lr = LogisticRegression()
            lr_grid_search = GridSearchCV(lr, params, cv=5)

            # Se optimizan los hiperparametros por medio de GridSearchCV
            lr_grid_search.fit(self.X_train, self.y_train)

            # Se imprimen los mejores parámetros encontrados
            print("\n ======== Hiperaparametros encontrados para el modelo ==============")
            print(lr_grid_search.best_params_)

            return lr_grid_search


#Esta función es la encargada de seleccionar las mejores variables para una regresión logistica

    def logit_select_var(self,n_features = 10):

        # Crear el modelo de regresión logística
        log_reg = LogisticRegression()

        # Selección recursiva de características (RFE) con validación cruzada (CV)
        rfe = RFE(log_reg, n_features_to_select = n_features)
        X_train_rfe = rfe.fit_transform(self.X_train, self.y_train)

        selected_vars = self.X_train.columns[rfe.support_]

        print("Las mejores variables según RFE para el modelo son:\n",str(selected_vars))

        return list(selected_vars)
        


#Esta función se encarga de evaluar todas las métricas que utilizara el modelo

    def calculate_metrics(self,model,type = 'test',return_predict = False):
        
        if type == 'test':

            y_pred = model.predict(self.X_test) #Hace la estimación del modelo en 1 y 0
            y_prob = model.predict_proba(self.X_test) #Hace la estimación del modelo en terminos de probabilidad
            y_prob = list(map(lambda x: x[1],y_prob)) #Captura la segunda probabilidad que es equivalente a que acepten o no una tarjeta de crédito

            auc = roc_auc_score(self.y_test, y_prob) #Estima el AUC
           
            precision = precision_score(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)

            print("\n ======== Métricas Modelo ==============")

            print("El AUC del modelo es:",'{:.1%}'.format(auc))
            
            print("La precision del modelo es:",'{:.1%}'.format(precision))
            print("El accuracy del modelo es:",'{:.1%}'.format(accuracy))
            print("El recall del modelo es:",'{:.1%}'.format(recall))

            if return_predict == True:
                
                df_test = self.X_test.copy()
                df_test['Win'] = self.y_test
                df_test['Predict'] = y_pred
                df_test['Probabilidad'] = y_prob

                
                return df_test
            
            else:
                pass

        else:

            y_pred = model.predict(self.X_train) #Hace la estimación del modelo en 1 y 0
            y_prob = model.predict_proba(self.X_train) #Hace la estimación del modelo en terminos de probabilidad
            y_prob = list(map(lambda x: x[1],y_prob)) #Captura la segunda probabilidad que es equivalente a que acepten o no una tarjeta de crédito

            auc = roc_auc_score(self.y_train, y_prob) #Estima el AUC
            fpr, tpr, thresholds = roc_curve(self.y_train, y_prob) #Calcula la curva ROC 
            ks = max(tpr-fpr)

            precision = precision_score(self.y_train, y_pred)
            accuracy = accuracy_score(self.y_train, y_pred)
            recall = recall_score(self.y_train, y_pred)

            print("\n ======== Métricas Modelo ==============")

            print("El AUC del modelo es:",'{:.1%}'.format(auc))
            print("El KS del modelo es:",'{:.1%}'.format(ks))

            print("La precision del modelo es:",'{:.1%}'.format(precision))
            print("El accuracy del modelo es:",'{:.1%}'.format(accuracy))
            print("El recall del modelo es:",'{:.1%}'.format(recall))


#Esta función es la encarga de correr el cross validation para cada modelo

    def cross_validation_model(self,model):

        # Cross Validation con ROC-AUC y Accuracy para Regresión Logística
        roc_auc = np.mean(cross_val_score(model, self.X_test, self.y_test, cv=5, scoring='roc_auc'))
        accuracy = np.mean(cross_val_score(model, self.X_test, self.y_test, cv=5, scoring='accuracy'))

        print("ROC-AUC: {:.1%}, Accuracy: {:.1%}".format(roc_auc, accuracy))

#Calculo de la curva roc y distancia al punto perfecto

    def auc_distancia(self,model):

        paraumbral=pd.DataFrame(np.transpose(np.round(roc_curve(self.y_train, model.predict_proba(self.X_train)[:,1]),4)), 
                                columns=["FPR", "TPR", "Umbral"])
        
        paraumbral["Distancia"] = np.sqrt(paraumbral["FPR"]**2 +(1-paraumbral["TPR"])**2) #Calculo de la distancia al punto perfecto
        paraumbral['Ks'] = paraumbral.TPR - paraumbral.FPR

        display(paraumbral.sort_values("Distancia"))
        print("El KS del modelo es:", str(paraumbral['Ks'].max()))


    def curva_roc(self,model, thres = 0.5,ajuste = False):

        if ajuste == False:
        
            logit_roc_auc = roc_auc_score(self.y_train, model.predict(self.X_train))  ## Calculo el area bajo la curva
            fpr, tpr, thresholds = roc_curve(self.y_train, model.predict_proba(self.X_train)[:,1]) ## Calculo la fpr(false positive rate),
            ## la tpr(true positive rate) y thresholds (los umbrales)
            plt.figure(figsize=(5,5))
            plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Curva ROC sin ajuste de Umbral')
            plt.legend(loc="lower right")
            plt.savefig('Log_ROC')
            plt.show()

        elif ajuste == True:

            logit_roc_auc = roc_auc_score(self.y_train, model.predict(self.X_train))  ## Calculo el area bajo la curva

            y_pred= np.where(model.predict_proba(self.X_train)[:,1] > thres, 1, 0)

            fpr, tpr, thresholds = roc_curve(self.y_train, y_pred) ## Calculo la fpr(false positive rate),
            ## la tpr(true positive rate) y thresholds (los umbrales)
            plt.figure(figsize=(5,5))
            plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Curva ROC con ajuste de Umbral')
            plt.legend(loc="lower right")
            plt.savefig('Log_ROC')
            plt.show()