"""
  author: Sierkinhane
  since: 2018-10-10 13:53:34
  description: spam-filter 
  needed improvement: add words out of my dictnary to generate a new comprehensive dictionary
"""
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from dictionary import to_dict, create_a_dict, combined
from  email_content_filter import email_content_filter

## 1. Load train email content
email = pd.read_csv('./data/spam-utf8.csv')

ins = {'ham': '0', 'spam': '1'}
email = email.replace({'v1': ins})
email['v1'] = email['v1'].astype('int32') # change into type of integer



#  1.1 email content filtering
X = email_content_filter(email['v2']) # return a list of words of each row in the mail
#  1.2 split into train and dev set
y = email['v1'].values

# num of positive and negative examples
print(email['v1'].value_counts())
amount_y_0 = email['v1'].value_counts()[0]
amount_y_1 = email['v1'].value_counts()[1]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)


## 2. Load dictionary
dictionary1 = to_dict('./words/words_alpha.txt')
dictionary2 = create_a_dict('./data/spam-utf8.csv')
dictionary = combined(dictionary1, dictionary2)

## 3. Compute parameters
#  3.1 parameters definition
p_y = 0  # p(y=1)
fi_y_0 = np.zeros((len(dictionary), 1), dtype=np.float32)  # φi|y=0
fi_y_1 = np.zeros((len(dictionary), 1), dtype=np.float32)  # φi|y=1

#  3.2 function
def feature_handle(row_x):

	x_temp = np.zeros((len(dictionary), 1))
	for col in row_x:
		try:
			dictionary[col]
		except Exception as e:
			pass
		else:
			row_in_vec = dictionary[col]
			x_temp [row_in_vec] = 1

	return x_temp 

def derive_parameters(dictionary, X_train, Y_train, p_y, fi_y_0, fi_y_1):
	
	error = []
	amount_y_1 = 0
	# x feature vector
	x0 = np.zeros((len(dictionary), 1))
	x1 = np.zeros((len(dictionary), 1))
	# m training examples
	i = 0
	for row_x, row_y in zip(X_train, Y_train):
		print(i)
		# the ith trainging example
		x_temp = feature_handle(row_x)
		if row_y:
			amount_y_1 += 1
			x1 = np.add(x1, x_temp)
		else:
			x0 = np.add(x0, x_temp)
		i+=1


	p_y = amount_y_1 / Y_train.shape[0]
	# laplace smoothing
	fi_y_0 = (x0 + 1) / (Y_train.shape[0] + 2)
	fi_y_1 = (x1 + 1) / (Y_train.shape[0] + 2) 

	return p_y, fi_y_0, fi_y_1

def prediction(dictionary, example, p_y, fi_y_0, fi_y_1):
	
	p_1 = 1
	p_0 = 1

	x_input = feature_handle(example)
	index = []
	for _ in example:
		try:
			dictionary[_]
		except Exception as e:
			pass
		else:
			index.append(dictionary[_])

	for i in index:
		p_1 *= (fi_y_1[i] ** x_input[i])
		p_0 *= (fi_y_0[i] ** x_input[i])

	prediction = (p_1 * p_y) / (p_1 * p_y + p_0 * (1-p_y))

	return prediction

def test(dictionary, X_test, Y_test, p_y, fi_y_0, fi_y_1):
	
	amount_of_spam_email = 0
	amount_y_1 = 0

	for x, y in zip(X_test, Y_test):
		p_y_1 = prediction(dictionary, x, p_y, fi_y_0, fi_y_1)
		if p_y_1 > 0.9:
			amount_of_spam_email += 1
		if y:
			amount_y_1 += 1

	print("prediction: [ham/spam][{0}/{1}]".format(Y_test.shape[0] - amount_of_spam_email, amount_of_spam_email))
	print("    real  : [ham/spam][{0}/{1}]".format(Y_test.shape[0] - amount_y_1, amount_y_1))

	print("prediction spam accuracy: {0}".format(amount_of_spam_email/amount_y_1, amount_y_1))


if __name__ == '__main__':

	p_y, fi_y_0, fi_y_1 = \
		derive_parameters(dictionary, X_train, Y_train, p_y, fi_y_0, fi_y_1)
	
	np.savetxt('./fi_y_0.csv', fi_y_0)
	np.savetxt('./fi_y_1.csv', fi_y_1)

	test(dictionary, X_test, Y_test, p_y, fi_y_0, fi_y_1)

	example = X_test[13]
	result = prediction(dictionary, example, p_y, fi_y_0, fi_y_1)
	print("real label:{0}".format(Y_test[13]))
	print("prediction:{0}".format(result))



