import json
import numpy as np
import pickle
import argparse
import jieba
from keras.models import load_model
from gensim.models import Word2Vec
import sys

model_path = './model_retrieve_model_my.h5'  # model_path
result_path = sys.argv[3]   # result csv
# Readfile
print("Reading File")
with open(sys.argv[1], 'rb') as f:  # test.data
    pretest_sound = pickle.load(f, encoding='latin1')

with open(sys.argv[2]) as f:  # test.csv
    target = f.readlines()

model = load_model(model_path)  # load_model

with open("./w2v_dic.p", 'r') as f:  # load dics
   target_token_index = json.loads(f.read())

# Preprocess
print("Preprocessing")
input_texts = []
target_texts = []
for line in target:
    line = line[:-1]
    sentences = line.split(',')
    for i in range(4):
        sentence = sentences[i].split() + ['E']
        target_texts.append(sentence)
test_sound = []
for sound in pretest_sound:
	for i in range(4):
		test_sound.append(sound)


max_encoder_length = 246
max_decoder_length = 15

encoder_input_data = np.zeros((len(test_sound), max_encoder_length, 39), dtype='float32')
decoder_input_data = np.zeros((len(test_sound), max_decoder_length), dtype='float32')
not_found = []
for i, (input_text, target_text) in enumerate(zip(test_sound, target_texts)):
    encoder_input_data[i, -len(input_text):, :] = input_text
    decoder_input_data[i, 0] =  target_token_index['B']
    for t, char in enumerate(target_text):
        try:
            decoder_input_data[i, t+1] = target_token_index[char]
        except:
            decoder_input_data[i, t+1] = 0
            not_found.append(char)

print(encoder_input_data.shape, decoder_input_data.shape)

print("Predicting...")
result = model.predict([encoder_input_data, decoder_input_data])

### Write Result
f = open(result_path, 'w')
f.write('id,answer\n')
for i in range(2000):
    s = i*4
    score = np.array(result[s:s+4]).reshape(-1)
    ans = np.argmax(score)
    f.write(str(i+1)+','+str(ans)+'\n')
print('Done prdict...')
