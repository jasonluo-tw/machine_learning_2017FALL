from __future__ import print_function
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Input, Embedding, merge
from keras.models import Model
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys
def load_data(file_name, file):

	if file == 'train':
		train = pd.read_fwf(file_name, header=None, encoding='utf_8', widths=[1, 8, 200])
		y = np.asarray(train[0])
		sentences =train[2].apply(lambda x:''.join([i for i in x if i in 'abcdefghijklmnopqrstuvwxyz ']))
	elif file == 'test':
		test = pd.read_fwf(file_name, encoding='utf_8', header=None, skiprows=1)
		sentences = test[0].apply(lambda x:''.join([i for i in x if i in 'abcdefghijklmnopqrstuvwxyz ']))
	elif file == 'nolabel':
		train_nolabel = pd.read_fwf(file_name, encoding='utf_8', header=None, widths=[200])
		sentences = train_nolabel[0].apply(lambda x:''.join([i for i in x if i in 'abcdefghijklmnopqrstuvwxyz ']))
	#f.close()
	print('load_data done!')
	if file == 'train':
		return sentences, y
	else:
		return sentences

def predict_data(test_data):
	result = model.predict(test_data,batch_size=256)
	rounded = [round(x[0]) for x in result]
	import csv
	out = []
	out.append(["id", "label"])
	for i in range(len(rounded)):
	     out.append([i,int(rounded[i])])
	filename = './result.csv'
	with open(filename,'w+') as f:
		s = csv.writer(f,delimiter=',',lineterminator='\n')
		for i in range(len(out)):
			s.writerow(out[i])
	print('Done predict')
	return

## main program ###
test_sen = load_data(sys.argv[1], 'test')
### word2vec
word_tr = Word2Vec.load('./w2v_size128.model.bin')
index_test = []
i = 0
for line in test_sen:
    index_test.append([])
    for word in line.split():
        if word in word_tr.wv:
            #print(word ,word_tr.wv.vocab[word].index)
            index_test[i].append(word_tr.wv.vocab[word].index)
    i += 1
test_ = pad_sequences(index_test, 36)
### Predict
from keras.models import load_model
model = load_model('./weights03-0.81.hdf5')

result1 = model.predict(test_)
print('Done 1')
## combine
final_result = result1
#final_result = result1
#print(final_result)
rounded = [round(x[0]) for x in final_result]
import csv
out = []
out.append(["id", "label"])
for i in range(len(rounded)):
	out.append([i,int(rounded[i])])
filename = sys.argv[2]
with open(filename,'w+') as f:
	s = csv.writer(f,delimiter=',',lineterminator='\n')
	for i in range(len(out)):
		s.writerow(out[i])
print('All_Done')