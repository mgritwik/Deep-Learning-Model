import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing,cross_validation
from sklearn.pipeline import Pipeline
 
df=pd.read_csv('ClassificationData.csv')
#load data
# dataframe=pd.read_csv("housing.csv",delim_whitespace=True,header=None)
# # dataset=dataframe.values
# ID,ATM Name,Transaction Date,No Of Withdrawals,No Of CUB Card Withdrawals,No Of Other Card Withdrawals,
# Total amount Withdrawn,Amount withdrawn CUB Card,Amount withdrawn Other Card,Sunday,Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,
# WorkingDay,H,N,C,M,NH,HWH,HHW,WWH,WHH,HWW,WWW,WHW,HHH,averageWithdrawals,AvgAmountPerWithdrawal,Rounded Amount Withdrawn,class
# df=pd.read_csv('ClassificationData.csv')
df=df[df['ATM Name']=='Christ College ATM']
df.drop(['ID','ATM Name','Transaction Date','No Of Withdrawals','No Of CUB Card Withdrawals','No Of Other Card Withdrawals',
		  'class','Amount withdrawn CUB Card','Amount withdrawn Other Card','Rounded Amount Withdrawn','AvgAmountPerWithdrawal'],1,inplace=True)
X=np.array(df.drop(['Total amount Withdrawn'],1))
print('Shape of X; ',X.shape)
# X=preprocessing.scale(X)
Y=np.array(df['Total amount Withdrawn'])
print('Shape of y; ',Y.shape)


# comment out
# X=np.array(df.drop(['i'],1))
# X=preprocessing.scale(X)
# y=np.array(df['i'])
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.2)
#split into input(X) adn output (Y) variables
# X=dataset[:,0:13]
# Y=dataset[:,13]

# X=preprocessing.scale(X)
#define base_model

def baseline_model():
	#create model
	model=Sequential()
	model.add(Dense(8,input_dim=22,kernel_initializer='normal',activation='relu'))
	# model.add(Dense(22,input_dim=22,activation='relu'))
	model.add(Dense(1,kernel_initializer='normal'))
	#compile model
	model.compile(loss='mean_squared_error',optimizer='adam')
	return model

# #fix random seeed for reproductibility
seed=7
np.random.seed(seed)
#evalurate model with standardized dataset
# estimators=[]
# estimators.append=[]
# estimators.append(('standardize',StandardScaler()))
# estimators.append(('mlp',KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=0)))
# # pipeline=Pipeline(estimators)
# estimator=KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=0)
# kfold=KFold(n_splits=10,random_state=seed)
# results=cross_val_score(estimator,X,Y,cv=kfold)
# print("Results: %.2f (%.2f) MSE"%(results.mean(),results.std()))


np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasRegressor(build_fn=baseline_model,epochs=50,batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=10,random_state=seed)
# results=cross_validate(pipeline,X,Y,cv=kfold,n_jobs=1)
results=cross_val_score(pipeline,X,Y,cv=kfold,n_jobs=1)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(),results.std()))
# print('testscore',results['test_score'])












































# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn import preprocessing
# from sklearn.pipeline import Pipeline
 

# #load data
# dataframe=pd.read_csv("housing.csv",delim_whitespace=True,header=None)
# dataset=dataframe.values

# #split into input(X) adn output (Y) variables
# X=dataset[:,0:13]
# Y=dataset[:,13]

# X=preprocessing.scale(X)
# #define base_model

# def baseline_model():
# 	#create model
# 	model=Sequential()
# 	model.add(Dense(13,input_dim=13,kernel_initializer='normal',activation='relu'))
# 	model.add(Dense(13,input_dim=13,activation='relu'))
# 	model.add(Dense(1,kernel_initializer='normal'))
# 	#compile model
# 	model.compile(loss='mean_squared_error',optimizer='adam')
# 	return model

# #fix random seeed for reproductibility
# seed=7
# np.random.seed(seed)
# #evalurate model with standardized dataset
# # estimators=[]
# # estimators.append=[]
# # estimators.append(('standardize',StandardScaler()))
# # estimators.append(('mlp',KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=0)))
# # pipeline=Pipeline(estimators)
# estimator=KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=0)
# kfold=KFold(n_splits=10,random_state=seed)
# results=cross_val_score(estimator,X,Y,cv=kfold)
# print("Results: %.2f (%.2f) MSE"%(abs(results.mean()),results.std()))