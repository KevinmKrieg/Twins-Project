#Import neccessary packages
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

#Column names for data
pitchers = pd.read_csv('pitchers.csv')
variables = ['MPH','Stress','Arm Speed','Arm Slot','Shoulder Rotation','Pitcher']
train_data = pd.DataFrame(columns = variables)
#Loop through csv sheets and read in Motus data provided by Twins
for filename in os.listdir(os.getcwd()):
    if filename.endswith('.csv') and filename.startswith('Public Master Plyo Sheet'):
        motus_data = pd.read_csv(filename)
        #Append data from each ball type to training data set     
        data_to_append = motus_data.iloc[5:54,2:7].reset_index(drop=True)
        data_to_append['Pitcher'] = pitchers
        data_to_append.columns = variables
        train_data = train_data.append(data_to_append)
    
        data_to_append = motus_data.iloc[5:54,8:13].reset_index(drop=True)
        data_to_append['Pitcher'] = pitchers
        data_to_append.columns = variables
        train_data = train_data.append(data_to_append)
    
        data_to_append = motus_data.iloc[5:54,15:20].reset_index(drop=True)
        data_to_append['Pitcher'] = pitchers
        data_to_append.columns = variables
        train_data = train_data.append(data_to_append)
    
        data_to_append = motus_data.iloc[5:54,21:26].reset_index(drop=True)
        data_to_append['Pitcher'] = pitchers
        data_to_append.columns = variables
        train_data = train_data.append(data_to_append)

#Remove bad values    
train_data = train_data.dropna(how = 'any', axis = 0)
#Pull out pylo velo green data as the test data set
test_data = train_data.iloc[392:441]

#Create seperate predictor and target data sets for both training and testing
predictors = ['MPH','Arm Speed','Arm Slot','Shoulder Rotation','Pitcher']
target = ['Stress']
train_predictors = train_data[predictors]
train_target = train_data[target]
test_predictors = test_data[predictors]
test_target = test_data[target]

#Create random forest object
#random_forest = RandomForestRegressor(n_estimators=500,n_jobs=-1)
lr = LinearRegression()
#Train random forest of decision trees on taining data set
#random_forest.fit(train_predictors,train_target)
lr.fit(train_predictors,train_target)
#Make Stress predictions on training, and testing data sets
#training_predictions = random_forest.predict(train_predictors)
#testing_predictions = random_forest.predict(test_predictors)
training_predictions = lr.predict(train_predictors)
testing_predictions = lr.predict(test_predictors)
#Calculate mean squared error to assess performance
mse_train = mean_squared_error(training_predictions,train_target)
mse_test = mean_squared_error(testing_predictions,test_target)

#Plot prediction results ad performance
plt.figure(0)
plt.scatter(testing_predictions, test_target)
plt.plot([40,90],[0,25],'r-')
plt.xlabel('Predicted Stress')
plt.ylabel('Actual Stress')
plt.legend([ 'perfect', 'prediction'])
plt.savefig('Figure 1.tiff')
 
#Plot Correlation Matrix
plt.figure(1)
correlations = train_data.astype(float).corr()
ax = plt.axes()
sns.heatmap(correlations, xticklabels=correlations.columns.values, yticklabels=correlations.columns.values,center=.25,cmap="RdBu",ax = ax,annot=True)
ax.set_title('Motus Data Correlation Matrix')
plt.savefig('Figure 2.tiff')