# Train model and make predictions
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing,cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
df=pd.read_csv('ClassificationData.csv')
df=df[df['ATM Name']=='Christ College ATM']
df.drop(['ID','ATM Name','Transaction Date','No Of Withdrawals','No Of CUB Card Withdrawals','No Of Other Card Withdrawals',
		  'Rounded Amount Withdrawn','Amount withdrawn CUB Card','Amount withdrawn Other Card','AvgAmountPerWithdrawal',
		  'Total amount Withdrawn'],1,inplace=True)
print(df.head(1))
# y=np.array(df['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23',
# 			  '24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43',
# 		  	  '44','45','46','47','48','49','50'])
# X=np.array(df.drop(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22',
# 				'23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43',
# 		  		'44','45','46','47','48','49','50'],1))

X=np.array(df.drop(['class'],1))
Y=np.array(df['class'])
X=preprocessing.scale(X)

# dataframe = pandas.read_csv("iris-data.csv", header=None)

# print('X[0] is: ', X[0])
# print('y[0] is: ', y[0])
# # print('###########'*10,'Dataset[0] is:  ',dataset[0].shape,'###########'*10)
# X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,Y,test_size=0.1)

# X = dataset[:,0:4].astype(float)
# Y = dataset[:,4]
# encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
df1=pd.read_csv('ClassificationDataOneHot.csv')
df1=df1[df1['ATM Name']=='Christ College ATM']
# print(df1.shape)
# y_50=np.zeros((len(df1.index),50))
# # print(y[0],y[0].shape)
# for i in range(0,50):
# 	j=str(i+1)
# 	y_50[:,i]=np.array(df1[str(j)])
dummy_y = np_utils.to_categorical(Y)	#numpy array
# df1=pd.read_csv('ClassificationDataOneHot.csv')



print('###########'*10,'dummy_y is:  ',dummy_y[0].shape,'###########'*10)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(500, input_dim=22, init='normal', activation='relu'))
	# model.add(Dense(500, activation='relu'))
	# model.add(Dense(500, activation='relu'))
	# model.add(Dense(1500, activation='relu'))
	# model.add(Dense(1500, activation='relu'))
	# model.add(Dense(1500, activation='relu'))
	# model.add(Dense(1500, activation='relu'))
	# model.add(Dense(1500, activation='relu'))
	# model.add(Dense(1500, activation='relu'))
	# model.add(Dense(1500, activation='relu'))
	model.add(Dense(40, init='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
# X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
# estimator.fit(X_train, Y_train)
# # scores=estimator.evaluate(X_test,y_test)
# # print("\n%s: %.2f%%" % (estimator.metrics_names[1],scores[1]*100))
# # for predictions
# predictions = estimator.predict(X_test)
# print(predictions)
# # print(encoder.inverse_transform(predictions))
# #for accuracy

kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,X,dummy_y,cv=kfold)
print("Baseline: %.2f%% (%.2f%%)"% 
	(results.mean()*100,results.std()*100))