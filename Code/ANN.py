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
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
#        self.fc3 = torch.nn.Linear(100,50)
#        self.fc4 = torch.nn.Linear(50,1)
    def forward(self,x):
        x_temp = self.fc1(x)
        x_temp = self.fc2(x_temp)
        x_temp = self.fc3(x_temp)
#        x_temp = self.fc4(x_temp)
        
        return x_temp
    
regression_model = Neural()
scaler = StandardScaler()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(regression_model.parameters(), lr= 0.0001)


kf = KFold(n_splits=5)
y_pred_kfold =[]
mse_kfold=[]

#Y_data=Y_data.drop(index_18k)
#X_data=X_data.drop(index_18k)

#Y_data=Y_data.drop(index_23k)
#X_data=X_data.drop(index_23k)

#Y_data=Y_data.drop([0,69])
#X_data=X_data.drop([0,69])

#randomShuffling
#index_list = X_data.index.tolist()
#random.shuffle(index_list)
#
#Y_data=Y_data.loc[index_list]
#X_data=X_data.loc[index_list]

Y_data= Y_data.reset_index(drop=True)
X_data= X_data.reset_index(drop=True)

y_actual=Y_data.values


for train, test in kf.split(X_data,y=Y_data):
    print("%s %s" % (train, test))
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
    y_pred_test = regression_model(x_test_tensor)
    mse = criterion(y_test_tensor,y_pred_test).item()
    y_pred_kfold.append(y_pred_test.cpu().detach().numpy())
    mse_kfold.append(mse)
#y_pred_kfold.reverse()
y_pred_kfold_np=[]
for k in range(len(y_pred_kfold)):
    for i in range(y_pred_kfold[k].shape[0]):
        y_pred_kfold_np.append(y_pred_kfold[k][i][0])
y_pred_kfold_np = np.array(y_pred_kfold_np)        
    
mse_mean =np.mean(np.array(mse_kfold))
mse_std =np.std(np.array(mse_kfold))

print("MSE MEAN: "+str(mse_mean)+"+-"+str(mse_std))
print(mse_kfold)

fig = plt.figure(2,dpi=120,figsize=(20,10))
plt.plot(Y_data.values)
plt.plot(y_pred_kfold_np)
plt.legend(['True','Predicted'])
plt.xlabel("Part Number")
plt.title("Prediction / Actual: Dimension #"+str(target_output_column))
plt.suptitle("MSE: "+ str(mse_mean))       
plt.show()
plt.clf()

error = Y_data.values-y_pred_kfold_np
fig, (ax1, ax2,ax3) = plt.subplots(3, 1, constrained_layout=True,dpi=120,figsize=(20,30))
ax1.plot(Y_data.values)
ax1.plot(y_pred_kfold_np)
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


















