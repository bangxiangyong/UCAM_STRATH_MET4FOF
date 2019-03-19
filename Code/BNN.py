# -*- coding: utf-8 -*-
#without truncating sequence length
import pickle
from torch.autograd import Variable
import numpy as np
import torch
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from scipy import stats


from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler

import pyro
from pyro.distributions import Bernoulli, Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import random
from sklearn.metrics import mean_squared_error
torch.random.manual_seed(100)
np.random.seed(100)
#load data
pickle_path = "../pickles/"
strath_preproc_data = pickle.load( open( pickle_path+"strath_preproc_data.p", "rb" ) )

#select one target output column 
target_output_column = 0
X_data = strath_preproc_data.ix[:,:-16]
Y_data = strath_preproc_data['output_'+str(target_output_column)]

#define model
class Neural(torch.nn.Module):
    def __init__(self):
        super(Neural,self).__init__()
#        self.fc1 = torch.nn.Linear(318,400)
#        self.fc2 = torch.nn.Linear(400,400)
#        self.fc3 = torch.nn.Linear(400,200)
#        self.fc4 = torch.nn.Linear(200,1)
        
        self.fc1 = torch.nn.Linear(318,150)
        self.fc2 = torch.nn.Linear(150,50)
        self.fc3 = torch.nn.Linear(50,1)
#        self.fc4 = torch.nn.Linear(50,1)
        dropout_p = 0.2
        self.dropout_layer = torch.nn.Dropout(p=dropout_p)
    def forward(self,x):
        x_temp = self.fc1(self.dropout_layer(x))
        x_temp = self.fc2(self.dropout_layer(x_temp))
        x_temp = self.fc3(self.dropout_layer(x_temp))
#        x_temp = self.fc4(self.dropout_layer(x_temp))
        
        return x_temp

def predict(regression_model, x_test_tensor,num_samples=20):
    y_preds_list = [regression_model(x_test_tensor).cpu().detach().numpy().reshape(-1) for i in range(num_samples)]
    y_preds_np = np.array(y_preds_list)
    y_pred_mean=np.mean(y_preds_np,axis=0)
    y_pred_std=np.std(y_preds_np,axis=0)
    
    return (y_preds_np,y_pred_mean,y_pred_std)
    

regression_model = Neural()
scaler = StandardScaler()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(regression_model.parameters(), lr= 0.0001)


kf = KFold(n_splits=5)
y_pred_kfold =[]
y_pred_mean_kfold =[]
y_pred_std_kfold =[]

mse_kfold=[]

#Y_data=Y_data.drop(index_18k)
#X_data=X_data.drop(index_18k)
#
#Y_data=Y_data.drop(index_23k)
#X_data=X_data.drop(index_23k)

Y_data=Y_data.drop([0,69])
X_data=X_data.drop([0,69])

index_list = X_data.index.tolist()
random.shuffle(index_list)

Y_data=Y_data.loc[index_list]
X_data=X_data.loc[index_list]

Y_data= Y_data.reset_index(drop=True)
X_data= X_data.reset_index(drop=True)

y_actual=Y_data.values

num_samples = 30
for train, test in kf.split(X_data,y=Y_data):
    #print("%s %s" % (train, test))
    #test=np.array([x for x in test if x not in index_23k] )
    #train=np.array([x for x in train if x not in index_23k])
    
    x_train=X_data.iloc[train]
    y_train=Y_data.iloc[train]
    x_test=X_data.iloc[test]
    y_test=Y_data.iloc[test]
    
    #preprocess
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)

    x_train_tensor = Variable(torch.from_numpy(x_train).float())
    y_train_tensor = Variable(torch.from_numpy(y_train.values).float())
    x_test_tensor = Variable(torch.from_numpy(x_test).float())
    y_test_tensor = Variable(torch.from_numpy(y_test.values).float())
    
    #training
    num_epochs = 1000
    for epoch in range(num_epochs):
        y_pred = regression_model(x_train_tensor).view(-1)
        loss = criterion(y_pred,y_train_tensor)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("LOSS: {}".format(loss.item()))
    y_preds_np,y_pred_mean,y_pred_std = predict(regression_model, x_test_tensor,num_samples=num_samples)
    #testing
    #y_pred_test = regression_model(x_test_tensor)
    y_pred_kfold = y_pred_kfold+y_preds_np.transpose(1,0).tolist()
    mse = mean_squared_error(y_test_tensor,y_pred_mean)
    y_pred_mean_kfold.append(y_pred_mean)
    y_pred_std_kfold.append(y_pred_std)
    
    mse_kfold.append(mse)
#y_pred_kfold.reverse()
y_pred_mean_kfold_np=[]
for k in range(len(y_pred_mean_kfold)):
    for i in range(y_pred_mean_kfold[k].shape[0]):
        y_pred_mean_kfold_np.append(y_pred_mean_kfold[k][i])
y_pred_mean_kfold_np = np.array(y_pred_mean_kfold_np)        

y_pred_kfold_std_np=[]
for k in range(len(y_pred_std_kfold)):
    for i in range(y_pred_std_kfold[k].shape[0]):
        y_pred_kfold_std_np.append(y_pred_std_kfold[k][i])
y_pred_kfold_std_np = np.array(y_pred_kfold_std_np)  

    
mse_mean =np.mean(np.array(mse_kfold))
mse_std =np.std(np.array(mse_kfold))

print("MSE MEAN: "+str(mse_mean)+"+-"+str(mse_std))
print(mse_kfold)
#y_pred_mean_kfold_np = np.delete(y_pred_mean_kfold_np,index_23k)
#y_actual = np.delete(y_actual,index_23k)
#fig = plt.figure(2,dpi=120,figsize=(20,10))
#plt.plot(y_actual)
#plt.plot(y_pred_mean_kfold_np)
#plt.legend(['True','Predicted'])
#plt.xlabel("Part Number")
#plt.ylabel("CMM Dimension #"+str(target_output_column))
#plt.title("Prediction / Actual: Dimension #"+str(target_output_column))
#plt.suptitle("MSE: "+ str(mse_mean))       
#plt.show()
#plt.clf()

fig, (ax1, ax2,ax3) = plt.subplots(3, 1, constrained_layout=True,dpi=120,figsize=(20,30))
ax1.plot(Y_data.values)
ax1.plot(y_pred_mean_kfold_np)
ax1.legend(['True','Predicted'])
ax1.set_xlabel("Part Number")
ax1.set_title("Prediction / Actual: Dimension #"+str(target_output_column))
ax2.plot(error,color='r')
ax2.set_title("Error Prediction / Actual: Dimension #"+str(target_output_column))
ax3.hist(error)
ax3.set_title("Histogram of Error Prediction")
fig.suptitle("MSE : "+str(mse_mean)+"+-"+str(mse_std))       
plt.show()
plt.clf()



#box plot
# Create a figure instance
fig = plt.figure(2,dpi=120,figsize=(20,10))
ax = fig.add_subplot(111)
ax.boxplot(y_pred_kfold,showmeans=True, showfliers=False)
ax.plot(np.arange(1,len(y_actual)+1),y_actual,'r.')
red_patch = mpatches.Patch(color='red', label='True Value')
ax.legend(handles=[red_patch])
plt.title("Boxplots: Uncertainty of BNN model ")
plt.ylabel("Model Prediction")
plt.xlabel("Part Number")
plt.show()
plt.clf()

error_v1 = np.abs(y_actual-y_pred_mean_kfold_np)
y_pred_kfold_std_np = np.abs(y_pred_kfold_std_np)

fig = plt.figure(2,dpi=120,figsize=(20,10))
#error_v1 = y_actual-y_pred_mean_kfold_np
plt.plot(y_pred_kfold_std_np)
plt.plot(error_v1)
plt.title("Std.Deviation of Model Prediction, Error: Y_actual - Y_pred")
plt.ylabel("Std.Deviation")
plt.xlabel("Part Number")
plt.show()
plt.clf()

#fig = plt.figure(2,dpi=120,figsize=(20,10))
#error_v2 = y_pred_mean_kfold_np-y_actual
#error_v2 = error_v1
#plt.plot(y_pred_kfold_std_np)
#plt.plot(error_v2)
#plt.title("Std.Deviation of Model Prediction, -Error: -(Y_actual - Y_pred)")
#plt.ylabel("Std.Deviation")
#plt.xlabel("Part Number")
#plt.show()
#plt.clf()


#plot correlation
fig = plt.figure(2,dpi=120,figsize=(5,5))
correlation = np.corrcoef(y_pred_kfold_std_np, error_v1)[0,1]
plt.scatter(y_pred_kfold_std_np,error_v1)
plt.xlabel("Std. Deviation of Model Prediction")
plt.ylabel("Error")
plt.title("Scatterplot: Std. Deviation of Model vs Error: Y_actual - Y_pred")
print(correlation)
plt.show()
plt.clf()


#
#fig = plt.figure(2,dpi=120,figsize=(10,10))
#plt.plot(Y_data.values,color='r')
#plt.plot(y_pred_mean_kfold_np+y_pred_kfold_std_np,color='b')
#plt.plot(y_pred_mean_kfold_np-y_pred_kfold_std_np,color='b')
#plt.plot(y_pred_mean_kfold_np,color='b')
#plt.show()
#plt.clf()
#









