# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 18:49:26 2022

@author: Najia
"""

## ----- importation des packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



##----------------------------importation du csv 

dataset=pd.read_csv('Scenario1.csv',sep=';')

##------------------------------qlq petites manip sur le jeu de données :

type(dataset)   #double check :)
 
dataset.isna().sum()  # pas de NA  :)

dataset['ME'] = [x.replace(',', '.') for x in dataset['ME']] 
dataset['Pdig'] = [x.replace(',', '.') for x in dataset['Pdig']]
dataset['SIDLys'] = [x.replace(',', '.') for x in dataset['SIDLys']]


for x in dataset.columns:
    
    dataset[x]=dataset[x].astype(float) 
    

dataset.pop('Sow')  #j'enleve car j'en ai pas besoin (on s'en fou de quelle truie c'est)

dataset.info() 
dataset.dtypes



##------------------------ Fonctions qui seront utilisées plusieurs fois later : 

#normalisation données:
def norm(x):
    return (x-train_stats['mean'])/train_stats['std']

#build modele de reseau de neurones
    
def build_model():
    model=keras.Sequential([
            layers.Dense(64,activation=tf.nn.relu,input_shape=[len(train_dataset.keys())]),
            layers.Dense(64,activation=tf.nn.relu),
            layers.Dense(1)
            ])
    optimizer=tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])
    return model

#suivi du 'fitting' du model:
    
class PrintDot(keras.callbacks.Callback):   #ca sert a suivre le temps d'execution du model
    def on_epoch_end(self,epoch,logs):
        if epoch%100==0 : print('')
        print('.',end='')

#Tracer le graphe de l'erreur (ici mae) sur l'ensemble dentraimenet et de validation

def plot_history(history):   #tracer l'erreur (ici mae), mais j'aurai pu prendre un autre metric
  hist=pd.DataFrame(history.history) 
  hist['epoch']=history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('MAE Error [ME]')
  plt.plot(hist['epoch'],hist['mae'], label='train error')
  plt.plot(hist['epoch'], hist['val_mae'], label='val error')
  plt.legend()
  plt.ylim([0, 10])        
        
  
  
  










######------------------------------- ETUDIER VARIABLE REPONSE 'ME'------------
dataset_ME=dataset.copy()
dataset_ME.pop('Pdig')
dataset_ME.pop('SIDLys')
train_dataset = dataset_ME.sample(frac=0.8, random_state=0)
test_dataset = dataset_ME.drop(train_dataset.index)

##---exploration des données (voir si y a lien...)
sns.pairplot(train_dataset)  #on peut rajouter d'autres var...
plt.show()

#--- qlq stat : moyenne, sd...
train_stats=train_dataset.describe()
train_stats.tail()  

train_labels = train_dataset.pop('ME')
test_labels = test_dataset.pop('ME')

train_stats=train_stats.transpose()
train_stats

train_dataset.describe().transpose()[['mean', 'std']]

##Normalisation des données : 
normed_train_data=norm(train_dataset)
normed_train_data.pop('ME')
normed_test_data=norm(test_dataset)
normed_test_data.pop('ME')

## construction du modele

model=build_model()

model.summary()

#verification : voir si le modele predit qlq chose
example_batch=normed_train_data[:10]
example_batch = np.asarray(example_batch).astype('float32')
example_result=model.predict(example_batch)

##--------entrainer le model: 


EPOCHS=1000
history=model.fit(normed_train_data,train_labels,epochs=EPOCHS,validation_split=0.2,verbose=0,callbacks=[PrintDot()])

hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
hist.tail()

# tracer l'erreur de l'ensemble de validation/entrainement pour detecter eventuel overfitting

plot_history(history)

###----prédiction :

test_predictions=model.predict(normed_test_data).flatten()

def pred():
    
    plt.scatter(test_labels,test_predictions,color='purple')
    plt.xlabel('vraies valeurs de ME ')
    plt.ylabel('valeurs prédites de ME ')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-100,100],[-100,100],color='purple') #la droite y=x 

pred()

###---- vérifier la normalité de l'erreur
error = test_predictions - test_labels
plt.hist(error, bins=25,color='purple')
plt.xlabel('Prediction Error ME')
_ = plt.ylabel('fréquence')









####
#####--------------------------------- ETUDIER LA VARIABLE REPONSE 'Pdig' ------


dataset_Pdig=dataset.copy()
dataset_Pdig.pop('ME')
dataset_Pdig.pop('SIDLys')
train_dataset = dataset_Pdig.sample(frac=0.8, random_state=0)
test_dataset = dataset_Pdig.drop(train_dataset.index)

##---exploration des données (voir si y a lien...)
sns.pairplot(train_dataset) #on peut rajouter d'autres var...
plt.show()

#--- qlq stat : moyenne, sd...
train_stats=train_dataset.describe()

train_stats.tail()  

train_labels = train_dataset.pop('Pdig')
test_labels = test_dataset.pop('Pdig')

train_stats=train_stats.transpose()
train_stats

train_dataset.describe().transpose()[['mean', 'std']]

##Normalisation des données : 
normed_train_data=norm(train_dataset)
normed_train_data.pop('Pdig')
normed_test_data=norm(test_dataset)
normed_test_data.pop('Pdig')

## construction du modele

model=build_model()

model.summary()

#verification : voir si le modele predit qlq chose
example_batch=normed_train_data[:10]
example_batch = np.asarray(example_batch).astype('float32')
example_result=model.predict(example_batch)

##--------entrainer le model: 


EPOCHS=1000
history=model.fit(normed_train_data,train_labels,epochs=EPOCHS,validation_split=0.2,verbose=0,callbacks=[PrintDot()])

hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
hist.tail()

# tracer l'erreur de l'ensemble de validation/entrainement pour detecter eventuel overfitting

plot_history(history)

###----prédiction :

test_predictions=model.predict(normed_test_data).flatten()

def pred():
    
    plt.scatter(test_labels,test_predictions,color='purple')
    plt.xlabel('vraies valeurs de Pdig ')
    plt.ylabel('valeurs prédites de Pdig ')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-100,100],[-100,100],color='red')#la droite y=x 
    
pred()

###---- vérifier la normalité de l'erreur
error = test_predictions - test_labels
plt.hist(error, bins=25,color='purple')
plt.xlabel('Prediction Error Pdig')
_ = plt.ylabel('fréquence')








##### ----------------------------------ETUDIER LA VARIABLE REPONSE SIDLys---------
dataset_SIDL=dataset.copy()
dataset_SIDL.pop('ME')
dataset_SIDL.pop('Pdig')
train_dataset = dataset_SIDL.sample(frac=0.8, random_state=0)
test_dataset = dataset_SIDL.drop(train_dataset.index)

##---exploration des données (voir si y a lien...)
sns.pairplot(train_dataset) 
plt.show()

#--- qlq stat : moyenne, sd...
train_stats=train_dataset.describe()

train_stats.tail()  

train_labels = train_dataset.pop('SIDLys')
test_labels = test_dataset.pop('SIDLys')

train_stats=train_stats.transpose()
train_stats

train_dataset.describe().transpose()[['mean', 'std']]

##Normalisation des données : 
normed_train_data=norm(train_dataset)
normed_train_data.pop('SIDLys')
normed_test_data=norm(test_dataset)
normed_test_data.pop('SIDLys')

## construction du modele

model=build_model()

model.summary()

#verification : voir si le modele predit qlq chose
example_batch=normed_train_data[:10]
example_batch = np.asarray(example_batch).astype('float32')
example_result=model.predict(example_batch)

##--------entrainer le model: 


EPOCHS=1000
history=model.fit(normed_train_data,train_labels,epochs=EPOCHS,validation_split=0.2,verbose=0,callbacks=[PrintDot()])

hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
hist.tail()

# tracer l'erreur de l'ensemble de validation/entrainement pour detecter eventuel overfitting

plot_history(history)

###----prédiction :

test_predictions=model.predict(normed_test_data).flatten()

def pred():
    
    plt.scatter(test_labels,test_predictions,color='purple')
    plt.xlabel('vraies valeurs de SIDLys ')
    plt.ylabel('valeurs prédites de SIDLys ')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-100,100],[-100,100],color='red')#la droite y=x 
    
pred()


###---- vérifier la normalité de l'erreur
error = test_predictions - test_labels
plt.hist(error, bins=25,color='purple')
plt.xlabel('Prediction Error SIDLys')
_ = plt.ylabel('fréquence')