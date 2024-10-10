# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021

@author: efthi
"""

from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\datasets_smc_mcmc_CART\Ionosphere.csv")
# df['Target'] = pd.Categorical(df['Target']).codes
# df['A'] = pd.Categorical(df['A']).codes

#df=df.drop(["Date"], axis = 1)
#df=df.drop(["month"], axis = 1)
#df=df.drop(["day"], axis = 1)
df = df.dropna()
y = df.Target
X = df.drop(['Target'], axis=1)
X = X.to_numpy()
y = y.to_numpy()



acc = []
# dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
# try:
#     treeSamples = dtMCMC.sample(500)

#     mcmcLabels = dt.stats(treeSamples, X_test).predict(X_test, use_majority=True)
#     mcmcAccuracy = [dt.accuracy(y_test, mcmcLabels)]
#     print("MCMC mean accuracy: ", (mcmcAccuracy))
# except ZeroDivisionError:
#     print("MCMC sampling failed due to division by zero")

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
    a = 100
    #b = 5
    target = dt.TreeTarget(a)
    initialProposal = dt.TreeInitialProposal(X_train, y_train)
    dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
    try:
        treeSMCSamples = dtSMC.sample(100, 100, resampling= "systematic")#systematic, knapsack, min_error, variational, min_error_imp
    
        smcLabels = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
        smcAccuracy = [dt.accuracy(y_test, smcLabels)]
        print("SMC mean accuracy: ", np.mean(smcAccuracy))
        acc.append(smcAccuracy)
    
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")
    
print("overall acc for 10 mc runs is: ", np.mean(acc))
