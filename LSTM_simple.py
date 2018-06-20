#!/usr/bin/env python
# -*- coding: utf-8 -*-
#title:		
#description:	SVM to classify images to one of the 50 models
#author:		
#date:		20171124
#version:	0.1
#usage:			
#notes:
#==============================================================================
import matplotlib.pyplot as plt
#from __future__ import print_function
import sys
from sys import argv
import json
#from scipy.io import loadmat
#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from scipy.stats import spearmanr
import h5py
import numpy as np
import os, math
import pprint
import pickle, json
import sqlite3 as lite
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import stats
import csv
from ConfigParser import SafeConfigParser

from scipy import stats
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding, Maximum, Lambda
from keras import backend as K_BACKEND
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers.merge import concatenate

parser = SafeConfigParser()


"""
dbs_A_path = '/home/jolwave/PopularityChallenge/first_day_DSA'
dbs_B_path = '/home/jolwave/PopularityChallenge/first_day_DSB'
dbs_A_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification\\first_day_DSA'
dbs_B_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification\\first_day_DSB'
K = 50
clustering_results_path = '/home/jolwave/PopularityChallenge/statistics/clustering output/clustering_'+str(K)+'_scaled.pickle'
clustering_results_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification\statistics\clustering output\clustering_'+str(K)+'_scaled.pickle'

shape_classifier_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification\\rndforest_20_class.pkl'
#feats_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification'
"""



def get_user_info(db,us_id):
	con = lite.connect(db)
	cur = con.cursor()
	cur.execute('SELECT Ispro, Contacts, PhotoCount, MeanViews, GroupsCount, GroupsAvgMembers, GroupsAvgPictures FROM user_info WHERE UserId = \''+us_id+'\'')
	res = cur.fetchall()
	res = res[0]
	return np.array(res)

def get_image_info(db,img_id):
	con = lite.connect(db)
	cur = con.cursor()
	cur.execute('SELECT Size, Title, Description, NumSets, NumGroups, AvgGroupsMemb, AvgGroupPhotos, Tags FROM image_info WHERE FlickrId = \''+img_id+'\'')
	res = cur.fetchall()
	res = res[0]

	feat = []    # np.zeros(len(res))

	feat.append(res[0])						 # Image size
	feat.append(len(res[1]))						 # Title length
	feat.append(len(res[2]))						 # Description length
	feat.append(res[3])	
	feat.append(res[4])
	feat.append(res[5])
	feat.append(res[6])
	feat.append(len(res[7]))						 #number of tags
 
 	return feat, np.array(res)

def load_features(ids_list, path_A, path_B,reducedUserInfo=False, reducedPhotoInfo=False):

	data = []

	con = lite.connect(path_A+'/headers.db')
	cur = con.cursor()
	cur.execute('SELECT FlickrId, UserId FROM headers')
	HD_A = cur.fetchall()
	con.close()
	DS_A = [x[0] for x in HD_A]

	con = lite.connect(path_B+'/headers.db')
	cur = con.cursor()
	cur.execute('SELECT FlickrId, UserId FROM headers')
	HD_B = cur.fetchall()
	con.close()
	DS_B = [x[0] for x in HD_B]

	if VERBOSE:
		print "\nLoading features:"

	HD = []
	for flickr_id in ids_list:
		try:
			im_idx = DS_A.index(flickr_id)		
			HD = HD_A
			db_path = path_A
		except ValueError:
			im_idx = DS_B.index(flickr_id)		
			HD = HD_B
			db_path = path_B

		user_id = HD[im_idx][1]
		if reducedUserInfo:
  		    user_feat = get_user_info_reduced(db_path+'/user_info.db',user_id)
  		else:
  		    user_feat = get_user_info(db_path+'/user_info.db',user_id)
	        if reducedPhotoInfo:
      		    image_feat, image_info = get_image_info_reduced(db_path+'/image_info.db',flickr_id)
		else:
		    image_feat, image_info = get_image_info(db_path+'/image_info.db',flickr_id)
			
		data.append(list(user_feat) + list(image_feat))
	return data

def load_deep_features(ids_list, path_A, path_B, db, feat):

	data = []
	con = lite.connect(db)
	cur = con.cursor()
	for flickr_id in ids_list:
		rows = cur.execute('SELECT '+ feat +' FROM Cnndata WHERE FlickrId == \''+flickr_id+'\'')
		f = json.loads(rows.next()[0])

		if 'google' in db.split('_') and feat == 'feat_1':
			f = np.array(f).flatten().squeeze().tolist()

		data.append(f)
	con.close()
	return data

def load_popularity_seq(flickr_ids,path,day=30):

	data = []
	with open(path+'ds_'+str(day)+'_days.pickle','r') as f:
		sequences = pickle.load(f)
		id_seq = [x[0] for x in sequences]
		for flickr_id in flickr_ids:
			idx = id_seq.index(flickr_id)
			seq = sequences[idx][2]
			data.append(seq)
	return data

# Khosla's popularity score
def pop_score(c,days):
	c = float(c)
	return math.log(c/days + 1,2)
	

def performance(A,B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
   
    #squared error
    SE = (A-B) ** 2
    #absolute error
    ABS = np.abs(A-B)
#    SEP = ((A-B)/A) ** 2
    return SE, ABS


def root_mean_squared_error(y_true, y_pred):
	return K_BACKEND.sqrt(K_BACKEND.square((y_pred - y_true), axis=1))


def load_deep_features(ids_list, path_A, path_B, db, feat):

	data = []
	con = lite.connect(db)
	cur = con.cursor()
	for flickr_id in ids_list:
		rows = cur.execute('SELECT '+ feat +' FROM Cnndata WHERE FlickrId == \''+flickr_id+'\'')
		f = json.loads(rows.next()[0])

		if 'google' in db.split('_') and feat == 'feat_1':
			f = np.array(f).flatten().squeeze().tolist()

		data.append(f)
	con.close()
	return data



def define_merge_model(feat_dim = 15, ff_size= 64, seq_dim = 2, lstm_size= 128, stateful=False, return_state=False):
	# feature model
	inputs1 = Input(shape=(feat_dim,))


	# model 01
#	fe1 = Dropout(0.5)(inputs1)
#	fe2 = Dense(100, activation='relu')(fe1) #256
	# model 02
	fe1 = Dense(ff_size, activation='relu')(inputs1) #256
#	fe2 = Dropout(0.5)(fe1)

#	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(1,seq_dim))		# shape = (timesteps, featdim)
	#se1 = Dense(256, activation='relu')(inputs2)
	#se1 = Embedding(1, 256, mask_zero=True)(inputs2)
	#se2 = Dropout(0.5)(se1)
	#se3 = LSTM(256)(se2)
	se3 = LSTM(lstm_size, stateful=stateful, return_state=return_state)(inputs2)   # 256

	# IMPLEMENTARE CONSTRAINT SEQ NON DECRESCENTI.. TODO: AGGIUNGERE ALL INPUT SEQ, IL VALORE PRECEDENTE E INSERIRLO COME INPUT DI MAXIMUM
#	max_out = Maximum()([se3,prec])

	# INJECT MODEL!!
	#input_state = Input(shape=(15,))
	#state_init = Dense(128,activation='tanh')(input_state)
	#ss = LSTM(128)(inputs2,initial_state = [state_init,state_init])

	# decoder model
#	decoder1 = add([fe2, se3])
	# MOD 5	
	decoder1 = concatenate([fe1, se3])
	#decoder2 = Dense(256, activation='relu')(decoder1)
	out = Dense(1)(decoder1)
	# tie it together [image, shape+seq] [seq]
	model = Model(inputs=[inputs1, inputs2], outputs=out)
	model.compile(loss='mean_squared_error', optimizer='adam')
#	model.compile(loss=root_mean_squared_error, optimizer='rmsprop')
	# summarize model
	print(model.summary())
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model



def define_merge_constrained_model(feat_dim = 15, ff_size= 64, seq_dim = 2, lstm_size= 128, stateful=False, return_state=False):

	# feature
	inputs1 = Input(shape=(feat_dim,))
	fe1 = Dense(ff_size, activation='relu')(inputs1) #256

	# sequential
	inputs2 = Input(shape=(1,seq_dim))		# shape = (timesteps, featdim)
	se3 = LSTM(lstm_size, stateful=stateful, return_state=return_state)(inputs2)   # 256

	decoder1 = concatenate([fe1, se3])
	#decoder2 = Dense(256, activation='relu')(decoder1)
	out = Dense(1)(decoder1)

	# QUESTO LAYER IMPONE LA NON DECRESCENZA DELLE SEQUENZE DI OUTPUT
	# NON FUNZIONA PERCHE' Maximum NON E' UN LAYER VALIDO PER LA GENERAZIONE DI OUTPUT
#	max_out = Maximum()([out,inputs2[0]])
#	max_out = keras.layers.maximum([out,inputs2[0]])

	max_out = Lambda(lambda x: K_BACKEND.max(x))([out,inputs2[:,:,0]])

	# tie it together [image, shape+seq] [seq]
	model = Model(inputs=[inputs1, inputs2], outputs=max_out)
	model.compile(loss='mean_squared_error', optimizer='adam')
#	model.compile(loss=root_mean_squared_error, optimizer='rmsprop')
	# summarize model
	print(model.summary())
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model


def transform_data_to_supervised(X_o, shape_ls, centroids, y_o):
	FEAT_SET = []
	IN_SEQ_SET = []
	OUT_SEQ_SET = []
	for t_idx, x in enumerate(X_o):
		#x = x[:2]
	#	- SEQUENTIAL INPUTS
	#	NB: il valore al giorno zero (0.0) viene eliminato dopo la differenziazione (difference(ss)). 

		# TRANSFORM THE SEQUENCES TO BE STATIONARY
		# load the shape (sequential data)
		l = shape_ls[t_idx]
		ss = centroids[l]
		ss = np.array(ss)		# shape
		if NORM_DIFF:
			shape_diff = difference(ss)
			# NB: i prototipi (shape) hanno dimensione 31, quindi non inserisco lo zero iniziale 
		else:
			shape_diff = ss
	
		# load the views dynamic (sequential data)
		seq = y_o[t_idx]
		seq = np.insert(seq,0,0)
		if NORM_DIFF:
			seq_diff = difference(seq)
			IN_SEQ_SET.append([0.0,0.0])	# inserisco questa coppia solo se NORM_DIFF == True
		else:
			seq_diff = seq

#		IN_SEQ_SET.append([0.0,0.0])
		OUT_SEQ_SET.append([seq_diff[0]]) # seq_diff[0] ==>> primo valore da predire ( 0.0 ==> p1 )
		for v_i in range(len(seq_diff)-1):
			seq_in = [seq_diff[v_i], shape_diff[v_i]]
			IN_SEQ_SET.append(seq_in)		#inputs2
			OUT_SEQ_SET.append([seq_diff[v_i+1]])
	
		#print len(IN_SEQ_SET)
		#sys.exit(0)
		# ogni 30 righe di train_in_seq rappresentano i dati di una immagine (una aggiunta prima del for e le altre 29 dentro il for)
	
		"""
		# DEBUG
		print len(ss)
		print ss
		print shape_diff
	#	print inverse_difference(diff)
		print '\n'
	#	seq = np.insert(seq,0,0)
		print len(seq)
		print seq
	#	diff = difference(seq)
		print seq_diff
	#	print inverse_difference(diff)
		sys.exit(0)
		"""

	#	- NOT-SEQUENTIAL INPUT
		for i in range(len(seq)-1):
			FEAT_SET.append(x)
	return FEAT_SET, IN_SEQ_SET, OUT_SEQ_SET		# 15x30 2x30 1x30 per ogni dato

def difference(series):
	diff = []
	for i in range(1,len(series)):
		diff.append(series[i]-series[i-1])
	return diff

def inverse_difference(diff):
	series = [0.0]
	for i in range(len(diff)):
		series.append(series[i] + diff[i])
	return series


#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
config_path = os.path.dirname(os.path.realpath(sys.argv[0]))
config_path = config_path + '/configuration.ini'
parser.readfp(open(config_path))
# Read configuration parameters
machine = parser.get('general','machine_name')
dbs_A_path = parser.get(machine,'dbs_A_path')
dbs_B_path = parser.get(machine,'dbs_B_path')
clustering_results_path = parser.get(machine,'clustering_results_path')
shape_classifier_path = parser.get(machine,'shape_classifier_path')
seq_path = parser.get(machine,'seq_path')

VERBOSE = True
K = 50  # 40
NUM_TRIALS = 1
n_epochs = 2000
NORM_DIFF = True
RET_STATES = False
# Input 01:	seq_in: views+shape(vs), nseq_in: social(x)

#	- OUTPUT SETTINGS
#OUT_DIR = MOD+'_results'
#if not os.path.exists(OUT_DIR):
#    os.makedirs(OUT_DIR)

START_VAL = -1.0  # minimum of tanh function
#### END SETTINGS ####

#
#  DATA LOADING
#
#	- FLICKR IDS
#	- CENTROIDS (SHAPE PROTOTYPES)
#	- GROUND TRUTH SHAPES (LABELS)
#	- DATA SUBSET
#	- GROUND TRUTH SEQUENCES
#	- SOCIAL FEATURES
#	- PREDICTED SHAPES
#
#  TRAIN/TEST SPLITTING
#
#	- DATA PREPROCESSING
#	- SEQUENTIAL INPUT
#	- NOT-SEQUENTIAL INPUT
#
#  MODEL DEFINITION
#
#  TRAIN
#
#  TEST
#

last_day = 30
with open(clustering_results_path,'r') as f:
	clusters = pickle.load(f)

#	- FLICKR IDS
flickr_ids = clusters['images']

#	- CENTROIDS
#	- GROUND TRUTH SHAPES
centroids  = clusters['kmeans_out'].cluster_centers_
#centroids = [c[:last_day+1] for c in centroids]
labels = clusters['kmeans_out'].labels_


#	- DATA SUBSET

flickr_ids = flickr_ids[:3000]
cluster_idx = [1,3,17,28,30,31,48]   # 9, 20, 32 ... 8, 21
cluster_idx = []
"""
for C in cluster_idx:
	plt.figure()
	plt.title(str(C))
	plt.plot(centroids[C])
	plt.show()
"""
if len(cluster_idx)>0:
	subset  = [f_id for i, f_id in enumerate(flickr_ids) if labels[i] in cluster_idx]
	subset_idx  = [i for i, f_id in enumerate(flickr_ids) if labels[i] in cluster_idx]
	flickr_ids = subset


#	- GROUND TRUTH SEQUENCES
sequences = load_popularity_seq(flickr_ids,seq_path, day=last_day)
#outliers = [idx for idx, s in enumerate(sequences) if s[-1] < 5]
#Y = [pop_score(x[-1],last_day) for x in sequences]            # Khosla's pop score at day 30
Y = sequences


#	- SOCIAL FEATURES
X = load_features(flickr_ids,dbs_A_path,dbs_B_path)


#	- PREDICTED SHAPES (LOAD THE CLASSIFIER)
with open(shape_classifier_path,'r') as f:  
    classifier = pickle.load(f)

#
#	TRAIN/TEST SPLITTING
#
split_i = int(0.9*len(flickr_ids))
train_flickr_ids = flickr_ids[:split_i]
test_flickr_ids = flickr_ids[split_i:]
print "Train data:\t" +str(len(train_flickr_ids)) + '/'+str(len(flickr_ids))
print "Test data:\t" +str(len(test_flickr_ids)) + '/'+str(len(flickr_ids))
print "Total data:\t" +str(len(train_flickr_ids)+len(test_flickr_ids))

X_train_o	= X[:split_i]
X_test_o	= X[split_i:]
y_train		= Y[:split_i]
y_test		= Y[split_i:]


#	- PREDICTED SHAPES (PERFORM SHAPE CLASSIFICATION)
# Fit the scaler with the train features
scaler = preprocessing.StandardScaler().fit(X_train_o)
X_train = scaler.transform(X_train_o)
#  Transform the test features according to the learned scaler
X_test = scaler.transform(X_test_o)	

# Predicted shape prototypes (test)
test_shape_ls = classifier.predict(X_test)
# Ground Truth prototypes (train)
train_shape_ls = labels[:split_i]

#free the memory
X_test = None
X_train = None
scaler = None

#	SUMMARY:
#	- train/test flickr ids:			train_flickr_ids, test_flickr_ids
#	- train/test social features (not scaled):	X_train_o, X_test_o
#	- train/test shape labels:			train_shape_ls, test_shape_ls
#	- train/test ground truth sequences:		y_train, y_test
#
#  DATA PREPROCESSING
#


# NB: i prototipi (shape) hanno dimensione 31
#print len(centroids[0])
#sys.exit(0)



#	TRAIN DATA
train_features = []
train_in_seq = []
train_out_seq = []

train_features, train_in_seq, train_out_seq = transform_data_to_supervised(X_train_o, train_shape_ls, centroids, y_train)
"""
print np.array(train_features)
print np.array(train_in_seq)
print np.array(train_out_seq)
print len(train_features)
print len(train_in_seq)
print len(train_out_seq)
"""
#sys.exit(0)


# FIT FEATURE SCALER  
scaler = preprocessing.StandardScaler().fit(train_features)	# mean variance scaler
X_train = scaler.transform(train_features)


# FIT INPUT SEQUENCE SCALER 
in_seq_scaler = MinMaxScaler(feature_range= (-1,1))
in_seq_scaler = in_seq_scaler.fit(train_in_seq)
IN_SEQ_train =  in_seq_scaler.transform(train_in_seq)

# FIT OUT SEQUENCE SCALER 
seq_scaler = MinMaxScaler(feature_range= (-1,1))
seq_scaler = seq_scaler.fit(train_out_seq)
OUT_SEQ_train =  seq_scaler.transform(train_out_seq)


test_features, test_in_seq, test_out_seq = transform_data_to_supervised(X_test_o, test_shape_ls, centroids, y_test)

# TEST DATA
# Transform the test features according to the learned scaler
X_test = scaler.transform(test_features)
# Transform sequential input for the test set
IN_SEQ_test =  in_seq_scaler.transform(test_in_seq)


#
#  MODEL DEFINITION
#
#LSTM_model, m_summary = define_model(MOD, seq_neurons= 128, nseq_neurons = 128)
LSTM_model = define_merge_model(feat_dim = len(X_train[0]), ff_size= 128, seq_dim = len(IN_SEQ_train[0]), lstm_size= 256, stateful = False, return_state=RET_STATES)
#LSTM_model = define_merge_constrained_model(feat_dim = len(X_train[0]), ff_size= 64, seq_dim = len(IN_SEQ_train[0]), lstm_size= 128, stateful = False, return_state=RET_STATES)


#	SUMMARY:
#	- train/test flickr ids:				train_flickr_ids, test_flickr_ids
#	- train/test not-sequential features (scaled):		X_train, X_test
#	- train/test sequential input (preprocessed):		IN_SEQ_train, IN_SEQ_test
#	- train ground truth sequences (preprocessed):		OUT_SEQ_train
#

train_loss = []
train_val = []
train_size =len(X_train)
n_train_images = len(X_train)/last_day
print n_train_images
for ep in range(n_epochs):

    #TRAINING
#    for t_idx, x in enumerate(X_train):   
    for t_idx in range(n_train_images):   
	img_id = train_flickr_ids[t_idx]

	#LSTM_X1, LSTM_X2, LSTM_Y =  get_batch(MOD,x,in_seq,seq) 
	LSTM_X1 = X_train[t_idx:t_idx+last_day]
	LSTM_X2 = IN_SEQ_train[t_idx:t_idx+last_day]
	LSTM_Y = OUT_SEQ_train[t_idx:t_idx+last_day]
	"""
	print LSTM_X1
	print LSTM_X2
	print LSTM_Y
	print len(LSTM_X1)
	print len(LSTM_X2)
	print len(LSTM_Y)
	sys.exit(0)
	"""

	LSTM_X2 = LSTM_X2.reshape(LSTM_X2.shape[0],1,2)	# reshape LSTM input to (samples,time steps,features)

   	# fit batch
	batch_size = len(LSTM_X1)
	print "\nEpoch:\t" + str(ep) + "/"+str(n_epochs)+"\t\tdata #\t" + str(t_idx) + "/" +str(n_train_images)
	# TODO: TENSORBOARD CALLBACK
	hist = LSTM_model.fit([LSTM_X1,LSTM_X2],LSTM_Y, epochs=1,batch_size=batch_size,shuffle=False) #,callbacks=[checkpoint])
        train_loss.append(np.mean(hist.history['loss']))
	print "HISTORY LOSS LEN"
	print len(hist.history['loss'])
	print "HISTORY LOSS LEN"

	# reset_states
	LSTM_model.reset_states()


	# DEBUG OUT
	if ep == 1999: #ep % 50 == 0: #and t_idx % 10 == 0:
		s = sequences[t_idx]
		s = np.array(s)
		print "\n"
			
		pred = START_VAL
		pred2 = START_VAL
		#print str(pred) + "\t" + "("+str(START_VAL)+")"
#			pred_s = [START_VAL]
		pred_s = []	# predicted sequence
		pred_s2 = []
		if not NORM_DIFF:
			pred_s = [START_VAL]
			pred_s2 = [START_VAL]

		for d in range(last_day):
			in2 = [pred]+[LSTM_X2[d,0,1]]	# da LSTM_X2 prendo solo la shape, indici: (samples,time steps,features)
			in2 = np.array(in2)
			pp = LSTM_model.predict([LSTM_X1[0].reshape(1,len(X_train[0])),in2.reshape(1,1,2)])

			in22 = [pred2]+[LSTM_X2[d,0,1]]	# da LSTM_X2 prendo solo la shape, indici: (samples,time steps,features)
			in22 = np.array(in22)
			pp2 = LSTM_model.predict([LSTM_X1[0].reshape(1,len(X_train[0])),in22.reshape(1,1,2)])
			"""
			pp, H, S = LSTM_model.predict([LSTM_X1[0].reshape(1,len(X_train[0])),in2.reshape(1,1,2)])
			print "##### OUTPUTS #####"
			print pp
			print H
			print S
			"""
		#	print "PREDICTION"
		#	print pp
#			pp = LSTM_model.predict([LSTM_X1[0].reshape(1,46),np.array([pred]).reshape(1,1,1)])
			pp = pp[0,0]	# predicted value
			pred_s.append(pp)
			pred = pp	# lo uso per la prossima prediction (next timestep)
			
			#print str(pp) + "\t" + "("+str(s[d])+")"
			pp2 = pp2[0,0]	# predicted value
			pred2 = np.max([pred2,pp2])
			pred_s2.append(pred2)
					
		s = np.insert(s,0,0)
		#print len(pred_s)
		# reverse scaling
#			predicted = seq_scaler.inverse_transform(np.array(pred_s).reshape(1,len(pred_s)))
		
		predicted = seq_scaler.inverse_transform([pred_s])
		predicted = predicted[0]
	
		predicted2 = seq_scaler.inverse_transform([pred_s2])
		predicted2 = predicted2[0]

		if NORM_DIFF:
			predicted = inverse_difference(predicted)		# questa funzione ripristina il primo valore
			predicted2 = inverse_difference(predicted2)		# questa funzione ripristina il primo valore

		SE, _ = performance(predicted,s)
		e = np.sqrt(np.mean(SE))
		print "RMSE:\t" + str(e)
		if True and e < 3:
			plt.figure()
			plt.title('FlickrId: ' + img_id)
			plt.plot(s,label='views sequence', linewidth=3.0)
			plt.plot(predicted,label='predicted')
			plt.plot(predicted2,label='predicted2')
			plt.legend(loc='best')
			plt.draw()
			plt.savefig('train_ep'+str(ep)+'_'+img_id+'.png')
			plt.show()

    # for each epoch --> loss
"""
print "\nSaving the model..."
LSTM_model.save('LSTM_model_'+MOD+'.h5')

# TESTING    

# walk-forward valudation on the test data
errors = []

for t_idx, x in enumerate(X_test):   
idx = test_idx_order[t_idx]
img_id = flickr_ids[idx]
#l = test_shape_ls[t_idx]                     # centroids[l]  ---> shape prototype
#test_shape_ls.append(l)

#x = np.array(x)
#X1 = np.concatenate((x, centroids[l]))
X1 = x

seq_n = SEQ_test[t_idx]

s = sequences[idx]
s = np.array(s)
print "\n"
pred = START_VAL   # 0
#print str(pred) + "\t" + "("+str(START_VAL)+")"
#if STATIONARITY_NORM:
pred_s = []
shape = X1[15:]
for d in range(30):
	if model_type == 'model2':
	#	print LSTM_X2.shape
	#	print LSTM_X2[:3]
#		print LSTM_X2[d,0,1]
	
		in2 = [pred]+[shape[d+1]]
		in2 = np.array(in2)
		pp = LSTM_model.predict([X1[:15].reshape(1,15),in2.reshape(1,1,2)])
	else:
		pp = LSTM_model.predict([X1.reshape(1,46),np.array([pred]).reshape(1,1,1)])
	pp = pp[0,0]
	pred_s.append(pp)
	#print str(pp) + "\t" + "("+str(s[d])+")"
	pred = pp

s = np.insert(s,0,0)

if SCALED:

	if not STATIONARITY_NORM:
		pred_s = np.insert(np.array(pred_s),0,START_VAL)

	predicted = seq_scaler.inverse_transform([pred_s])
	predicted = predicted[0]
else:
	predicted = pred_s


if STATIONARITY_NORM:
	predicted = inverse_difference(predicted)		# questa funzione ripristina il primo valore pari a zero
SE, _ = performance(predicted,s)
e = np.sqrt(np.mean(SE))
errors.append(e)
print "RMSE:\t" + str(e)
if False and e < 2:
	plt.figure()
	plt.title('FlickrId: ' + img_id)
	plt.plot(s,label='views sequence', linewidth=3.0)
	plt.plot(predicted,label='predicted')
	plt.legend(loc='best')
	plt.draw()
	plt.savefig('TEST_prediction_'+MOD+'_ep'+str(ep)+'_'+img_id+'.png')
#		plt.show()

print "\n\nRMSE:\t" + str(np.mean(errors))
print "tRMSE 0.25:\t" + str(stats.trim_mean(errors,0.25))
print "RMSE MED:\t" + str(np.median(errors))


plt.figure()
plt.plot(loss_history)
plt.title('Training loss')
plt.draw()
plt.show()

"""
