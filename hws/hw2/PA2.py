import numpy as np
import pandas
import csv

def load(path):
	df = None
	'''YOUR CODE HERE'''
	
	df = pandas.read_csv(path)

	print(df)
	'''END'''
	return df


def prior(df):
	ham_prior = 0
	spam_prior =  0
	'''YOUR CODE HERE'''
	for index, row in df.iterrows():
		if row[4] == 1:
			spam_prior += 1
		if row[4] == 0:
			ham_prior += 1

	# print(ones.shape[0])
	# print(df.shape[0])
	total = df.shape[0]
	spam_prior = spam_prior/ total
	ham_prior = ham_prior/total

	print('hp',ham_prior)

	'''END'''
	return ham_prior, spam_prior

def likelihood(df):
	

		
	ham_like_dict = {}
	spam_like_dict = {}
	ham_prior = 0
	spam_prior =  0
		
	for index, row in df.iterrows():
		
		content = row[3]
		spam = row[4]
		counted = []
		if spam:
			spam_prior += 1
		else:
			ham_prior += 1
		for word in content.split(' '):
			
			if spam:
				
				if word not in counted:
					counted.append(word)
					if word in spam_like_dict.keys():
					
						spam_like_dict[word] += 1
					else:
						spam_like_dict[word] = 1

			if spam == 0:
				
				if word not in counted:
					
					counted.append(word)

					if word in ham_like_dict.keys():
						
						ham_like_dict[word] += 1
					else:
						ham_like_dict[word] = 1
			


		# split email on spaces
		# update corresponding dict 

	for key in spam_like_dict.keys():
		spam_like_dict[key] = spam_like_dict[key]/ spam_prior # (df['label_num'] == 1).shape[0]
	for key in ham_like_dict.keys():
		ham_like_dict[key] = ham_like_dict[key]/ ham_prior # (df['label_num'] == 0).shape[0]
	'''END'''



	return ham_like_dict, spam_like_dict

def predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, text):
	'''
	prediction function that uses prior and likelihood structure to compute proportional posterior for a single line of text
	'''
	#ham_spam_decision = 1 if classified as spam, 0 if classified as normal/ham
	ham_spam_decision = None




	'''YOUR CODE HERE'''
	#ham_posterior = posterior probability that the email is normal/ham
	ham_posterior = None

	#spam_posterior = posterior probability that the email is spam
	spam_posterior = None

	for word in text.split(" "):
		if word in ham_like_dict:
			ham_prior*=ham_like_dict[word]
		else:
			ham_prior*=(10**-5)

		if word in spam_like_dict:
			spam_prior*=spam_like_dict[word]
		else:
			spam_prior*=(10**-5)
	return(1 - (ham_prior > spam_prior))





def metrics(ham_prior, spam_prior, ham_dict, spam_dict, df):
	'''
	Calls "predict"
	'''
	hh = 0 #true negatives, truth = ham, predicted = ham
	hs = 0 #false positives, truth = ham, pred = spam
	sh = 0 #false negatives, truth = spam, pred = ham
	ss = 0 #true positives, truth = spam, pred = spam
	num_rows = df.shape[0]
	for i in range(num_rows):
		roi = df.iloc[i,:]
		roi_text = roi.text
		roi_label = roi.label_num
		guess = predict(ham_prior, spam_prior, ham_dict, spam_dict, roi_text)
		if roi_label == 0 and guess == 0:
			hh += 1
		elif roi_label == 0 and guess == 1:
			hs += 1
		elif roi_label == 1 and guess == 0:
			sh += 1
		elif roi_label == 1 and guess == 1:
			ss += 1

	acc = (ss + hh)/(ss+hh+sh+hs)
	precision = (ss)/(ss + hs)
	recall = (ss)/(ss + sh)
	return acc, precision, recall
    
if __name__ == "__main__":
	df = load('C:\\Users\\jcbol\\OneDrive\\Documents\\@Spring2022\\CS365\\cs365-spring22\\hws\\hw2\\TRAIN_balanced_ham_spam.csv')
	text = load('C:\\Users\\jcbol\\OneDrive\\Documents\\@Spring2022\\CS365\\cs365-spring22\\hws\\hw2\\TEST_balanced_ham_spam.csv')
	ham_prior, spam_prior = prior(df)
	ham_like_dict, spam_like_dict = likelihood(df)

	print(metrics(ham_prior, spam_prior, ham_like_dict, spam_like_dict, text))

	'''YOUR CODE HERE'''
	#this cell is for your own testing of the functions above