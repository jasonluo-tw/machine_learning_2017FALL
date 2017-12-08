import numpy as np
import pandas as pd
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

def RNN(dimens):
	model = Sequential()
	model.add(Embedding(dimens[0], 40, input_length=dimens[1]))
	model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))
	model.add(LSTM(256, return_sequences=False, dropout=0.2))
	model.add(Dense(128, activation='relu')) # 2nd
	model.add(Dropout(0.3))

	model.add(Dense(1, activation='sigmoid'))
	#adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
	#optimizer='adam',
	return model

def callbacks():
	# checkpoint
	filepath="output/best-weight-{epoch:02d}-{val_acc:.2f}.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	# early-stopping 
	call = EarlyStopping(monitor='val_loss', min_delta=0,patience=5)
	## batch_print_callback
	batch_print_callback = LambdaCallback(
    	on_epoch_end=lambda batch, logs: print(
        	'\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %
        	(batch, logs['val_acc'], batch, logs['acc'])))

	callbacks_list = [checkpoint ,call]
	return callbacks_list

def predict_data(test_data):
	result = model.predict(test_data,batch_size=32)
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
train_sen, Y = load_data(sys.argv[1], 'train')
nolabel_sen = load_data(sys.argv[2], 'nolabel')
### Tokenize
all_sen = np.append(train_sen, nolabel_sen)
tok = Tokenizer(num_words=20000, filters='1234567890!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", char_level=False)
tok.fit_on_texts(all_sen)
sequences = tok.texts_to_sequences(train_sen[:])

word_index = tok.word_index
MAX_SEQUENCE_LENGTH = 60
data = pad_sequences(sequences, maxlen= MAX_SEQUENCE_LENGTH)
### some parameter
### Training
model = RNN([word_index, MAX_SEQUENCE_LENGTH])
callback_list = callbacks()
#model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=256, epochs=50, shuffle=True, callbacks=callback_list)
model.fit(data, y=Y,  epochs=64, validation_split=0.2, shuffle=True, batch_size=32,callbacks=callback_list)
###
from keras.models import load_model
model.save('./weights03-0.81.hdf5')

print('Done program')
