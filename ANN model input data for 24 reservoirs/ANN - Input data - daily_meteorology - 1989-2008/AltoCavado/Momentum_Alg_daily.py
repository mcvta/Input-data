import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import KFold
from scipy.interpolate import Rbf
from scipy import stats
from neupy import layers, algorithms
from neupy import plots
from neupy import algorithms
from neupy.layers import *
import dill

# Import data
data = pd.read_excel('InputAC.xlsx', index_col=0, header=0)
data.columns = ['Temp(alb)','Temp(In)','Tair','Tdew', 'Wind','HR']

data['Month'] = data.index.month
data['MonthCos'] = np.cos(2 * np.pi * data.index.month / 12)
data['MonthSin'] = np.sin(2 * np.pi * data.index.month / 12)

data['DayOfYear'] = data.index.dayofyear
data['DayOfYearCos'] = np.cos(2 * np.pi * data.index.dayofyear / 365)
data['DayOfYearSin'] = np.sin(2 * np.pi * data.index.dayofyear / 365)

data['WeekOfYear'] = data.index.weekofyear
data['WeekOfYearCos'] = np.cos(2 * np.pi * data.index.weekofyear / 52)
data['WeekOfYearSin'] = np.sin(2 * np.pi * data.index.weekofyear / 52)

data['WeekOfHalfYearCos'] = np.cos(2 * np.pi * data.index.weekofyear / 26)
data['WeekOfHalfYearSin'] = np.sin(2 * np.pi * data.index.weekofyear / 26)

# Add filtered data
tmp2 = data.loc[:,['Temp(In)','Tair','Tdew', 'Wind','HR']].rolling(31, center=False, axis=0, min_periods=1).mean()
tmp2.columns = ['TempinMeanFilter_31', 'TairMeanFilter_31', 'TdewMeanFilter_31','TWindMeanFilter_31','HRMeanFilter_31']

tmp3 = data.loc[:,['Temp(In)','Tair','Tdew', 'Wind','HR']].rolling(31, center=False, axis=0, min_periods=1).std()
tmp3.columns = ['TempinStdFilter_31', 'TairStdFilter_31', 'TdewStdFilter_31','TWindStdFilter_31','HRStdFilter_31']

data = pd.concat((data, tmp2, tmp3 ** 2), axis=1)

# Drop empty records
data = data.dropna()

#Define data (Temp)

X = data.iloc[:, 1:]
y = data.loc[:, ['Temp(alb)']]

years = data.index.year
yearsTrain, yearsTest = train_test_split(np.unique(years), test_size=0.3, train_size=0.7, random_state=42)


XTrain = X.query('@years in @yearsTrain')
yTrain = y.query('@years in @yearsTrain').values.ravel()
XTest = X.query('@years in @yearsTest')
yTest = y.query('@years in @yearsTest').values.ravel()
results = y.query('@years in @yearsTest')


#===============================================================================
# Neural network
#===============================================================================

# Define neural network

cgnet = algorithms.Momentum(
    network=[
        layers.Input(XTrain.shape[1]),
        layers.Relu(24),
        layers.Linear(1),
    ],
    step=algorithms.step_decay(
        initial_value=0.05,
        reduction_freq=750,
    ),
    loss='mse',
    batch_size=None,
    regularizer=algorithms.l2(0.002),
    shuffle_data=False,
    verbose=True,
    show_epoch=100,
)

XScaler = StandardScaler()
XScaler.fit(XTrain)
XTrainScaled = XScaler.transform(XTrain)
XTestScaled = XScaler.transform(XTest)

yScaler = StandardScaler()
yScaler.fit(yTrain.reshape(-1, 1))
yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)).ravel()
yTestScaled = yScaler.transform(yTest.reshape(-1, 1)).ravel()

# Train
cgnet.train(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, epochs=5000)
# cgnet.plot_errors()

yEstTrain = yScaler.inverse_transform(cgnet.predict(XTrainScaled).reshape(-1, 1)).ravel()
mae = np.mean(np.abs(yTrain-yEstTrain))
results['ANN'] = yScaler.inverse_transform(cgnet.predict(XTestScaled).reshape(-1, 1)).ravel()



# Metrics
mse  = np.mean((yTrain-yEstTrain)**2)
mseTes = np.mean((yTest-results['ANN'])**2)
maeTes = np.mean(np.abs(yTest-results['ANN']))
meantrain = np.mean(yTrain)
ssTest = (yTrain-meantrain)**2
r2=(1-(mse/(np.mean(ssTest))))
meantest = np.mean(yTest)
ssTrain = (yTest-meantest)**2
r2Tes=(1-(mseTes/(np.mean(ssTrain))))


# Plot results
print("NN MAE: %f (All), %f (Test) " % (mae, maeTes))
print ("NN MSE: %f (All), %f (Test) " % (mse, mseTes))
print ("NN R2: %f (All), %f (Test) " % (r2, r2Tes))

results.plot()
plt.show()


"""
plt.scatter(yTest,results['ANN'])
plt.plot([7, 27], [7, 27], color='red')
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.show(block=True)

"""
#===============================================================================
# Predict with unseen data
#===============================================================================

# Import data
dataR = pd.read_excel('RealAC.xlsx', index_col=0, header=0)
dataR.columns = ['Temp(alb)','Temp(In)','Tair','Tdew', 'Wind','HR']

dataR['Month'] = dataR.index.month
dataR['MonthCos'] = np.cos(2 * np.pi * dataR.index.month / 12)
dataR['MonthSin'] = np.sin(2 * np.pi * dataR.index.month / 12)

dataR['DayOfYear'] = dataR.index.dayofyear
dataR['DayOfYearCos'] = np.cos(2 * np.pi * dataR.index.dayofyear / 365)
dataR['DayOfYearSin'] = np.sin(2 * np.pi * dataR.index.dayofyear / 365)

dataR['WeekOfYear'] = dataR.index.weekofyear
dataR['WeekOfYearCos'] = np.cos(2 * np.pi * dataR.index.weekofyear / 52)
dataR['WeekOfYearSin'] = np.sin(2 * np.pi * dataR.index.weekofyear / 52)

dataR['WeekOfHalfYearCos'] = np.cos(2 * np.pi * dataR.index.weekofyear / 26)
dataR['WeekOfHalfYearSin'] = np.sin(2 * np.pi * dataR.index.weekofyear / 26)

# Add filtered data
tmp2 = dataR.loc[:,['Temp(In)','Tair','Tdew', 'Wind','HR']].rolling(31, center=False, axis=0, min_periods=1).mean()
tmp2.columns = ['TempinMeanFilter_31', 'TairMeanFilter_31', 'TdewMeanFilter_31','TWindMeanFilter_31','HRMeanFilter_31']

tmp3 = dataR.loc[:,['Temp(In)','Tair','Tdew', 'Wind','HR']].rolling(31, center=False, axis=0, min_periods=1).std()
tmp3.columns = ['TempinStdFilter_31', 'TairStdFilter_31', 'TdewStdFilter_31','TWindStdFilter_31','HRStdFilter_31']

dataR = pd.concat((dataR, tmp2, tmp3 ** 2), axis=1)

# Drop empty records
dataR = dataR.dropna()

#Define data (Temp)

Xr = dataR.iloc[:, 1:]
yr = dataR.loc[:, ['Temp(alb)']]

XTestR = Xr
yTestR = yr.values.ravel()
XTestScaled = XScaler.transform(XTestR)
yTestScaled = yScaler.transform(yTestR.reshape(-1, 1)).ravel()
yTestPredicted = yScaler.inverse_transform(cgnet.predict(XTestScaled).reshape(-1, 1)).ravel()

results2 = pd.DataFrame({'obs': yTestR, 'est': yTestPredicted}, index = XTestR.index)


#===============================================================================
#Export results to Excel
writer = pd.ExcelWriter('ResultsNNTemp.xlsx')
results2.to_excel(writer,'results')
writer.save()
#===============================================================================

print(results2)



