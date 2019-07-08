import pandas as pd
import numpy as np
from scipy import stats
#Read data
train = pd.read_csv('adult.data.csv',header=None)
test = pd.read_csv('adult.test.csv',header=None)

print("Finish data loading.")

#total num of <=50K and >50K
num_y1 = 0
num_y2 = 0
for i in range(len(train)):
	if(train.iloc[i,14] == ' <=50K'):
		num_y1 += 1
	else:
		num_y2 += 1

#probability of <=50K and >50K
prob_y1 = num_y1/(num_y1+num_y2)
prob_y2 = num_y2/(num_y1+num_y2)

#probability for all values of num-th feature.
def workOnProb(train, num, isConti, num_y1, num_y2):
	"""
	train: the train data
	num: the feature working on
	isConti: is the feature continuous
	num_y1, num_y2: as above
	return type: a list of two dictionary, one is probability of ' <=50K'
				one is ' >50K'. If the feature is continuous, the dictionary 
				have two keys: 'mu' for the data's mean, 'sigma' for the
				standard deviation of the data. If the feature is not
				continuous, the dictionary's keys are corresponding to
				the value of the data, and the value of the keys are
				corresponding to their probability.
	"""

	res1 = {}
	res2 = {}
	if(isConti):
		#the list of corresponding data
		list1 = [] 
		list2 = []
		for i in range(len(train)):
			if(train.iloc[i,14] == ' <=50K'):
				list1.append(int(train.iloc[i,num]))
			else:
				list2.append(int(train.iloc[i,num]))
		res1['mu'] = np.mean(list1)
		res2['mu'] = np.mean(list2)
		res1['sigma'] = np.std(list1)
		res2['sigma'] = np.std(list2)
	else:
		for i in range(len(train)):
			if (train.iloc[i,14] == ' <=50K'):
				res1[train.iloc[i,num]] = res1.get(train.iloc[i,num], 0.) + 1.
			else:
				res2[train.iloc[i,num]] = res2.get(train.iloc[i,num], 0.) + 1.
		for it in res1.keys():
			res1[it] /= num_y1
		for it in res2.keys():
			res2[it] /= num_y2
	res = [res1, res2]
	return res

def isContinuous(x):
	"""
	x: the index to determine.
	return: if the x-th feature is continuous
	"""
	if(x in [0,2,4,10,11,12]):
		return True
	else:
		return False

Prob = [] #all features' probability
for i in range(14):
	Prob.append(workOnProb(train, i, isContinuous(i), num_y1, num_y2))

print("Finish training.")

acc = 0 #accuracy 
for i in range(len(test)):
	y1, y2 = prob_y1, prob_y2
	for j in range(14):
		if(isContinuous(j)):
			y1 *= stats.norm.pdf(int(test.iloc[i,j]), Prob[j][0]['mu'], Prob[j][0]['sigma'])
			y2 *= stats.norm.pdf(int(test.iloc[i,j]), Prob[j][1]['mu'], Prob[j][1]['sigma'])
		else:
			y1 *= Prob[j][0].get(test.iloc[i,j],1)
			y2 *= Prob[j][1].get(test.iloc[i,j],1)
	if((y1>y2 and test.iloc[i,14] == ' <=50K.') or (y1<y2 and test.iloc[i,14] == ' >50K.')):
		acc += 1

acc /= len(test)

print("the accuracy: {}%".format(acc*100))