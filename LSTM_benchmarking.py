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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
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
	return K.sqrt(K.square((y_pred - y_true), axis=1))


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


def old_define_model():
	# feature model
	inputs1 = Input(shape=(46,))

	# model 01
#	fe1 = Dropout(0.5)(inputs1)
#	fe2 = Dense(100, activation='relu')(fe1) #256
	# model 02
	fe1 = Dense(100, activation='relu')(inputs1) #256
	fe2 = Dropout(0.5)(fe1)

#	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(1,1))		# shape = (timesteps, featdim)
	#se1 = Dense(256, activation='relu')(inputs2)
	#se1 = Embedding(1, 256, mask_zero=True)(inputs2)
	#se2 = Dropout(0.5)(se1)
	#se3 = LSTM(256)(se2)
	se3 = LSTM(100)(inputs2)   # 256
	# decoder model
#	decoder1 = add([fe2, se3])
	# MOD 5	
	decoder1 = concatenate([fe2, se3])
	#decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(1)(decoder1)
	# tie it together [image+shape, seq] [seq]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='mean_squared_error', optimizer='adam')
#	model.compile(loss=root_mean_squared_error, optimizer='rmsprop')
	# summarize model
	print(model.summary())
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model


def define_model2():
	# feature model
	inputs1 = Input(shape=(15,))

	# model 01
#	fe1 = Dropout(0.5)(inputs1)
#	fe2 = Dense(100, activation='relu')(fe1) #256
	# model 02
	fe1 = Dense(64, activation='relu')(inputs1) #256
#	fe2 = Dropout(0.5)(fe1)

#	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(1,2))		# shape = (timesteps, featdim)
	#se1 = Dense(256, activation='relu')(inputs2)
	#se1 = Embedding(1, 256, mask_zero=True)(inputs2)
	#se2 = Dropout(0.5)(se1)
	#se3 = LSTM(256)(se2)
	se3 = LSTM(128)(inputs2)   # 256

	# INJECT MODEL!!
	#input_state = Input(shape=(15,))
	#state_init = Dense(128,activation='tanh')(input_state)
	#ss = LSTM(128)(inputs2,initial_state = [state_init,state_init])

	# decoder model
#	decoder1 = add([fe2, se3])
	# MOD 5	
	decoder1 = concatenate([fe1, se3])
	#decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(1)(decoder1)
	# tie it together [image, shape+seq] [seq]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='mean_squared_error', optimizer='adam')
#	model.compile(loss=root_mean_squared_error, optimizer='rmsprop')
	# summarize model
	print(model.summary())
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model



def define_model(mod, seq_neurons= 128, nseq_neurons = 128):

	#[M|I]_NO_<SEQ_IN>_<NSEQ_IN>
	model_code = mod.split('_')
	model_type = model_code[0]
	model_no = model_code[1]
	model_seq_in = model_code[2]
	model_nseq_in = model_code[3]


	# FEED FORWARD MODEL
	# Not-sequential input (social feats=='x', deep feats=='d')
	if model_nseq_in == 'x':
		nseq_in_size = 15
	else:
		nseq_in_size = 4096
	inputs1 = Input(shape=(nseq_in_size,))

#	fe1 = Dropout(0.5)(inputs1)
#	fe2 = Dense(100, activation='relu')(fe1) #256
	fe1 = Dense(nseq_neurons, activation='relu')(inputs1) #256
#	fe2 = Dropout(0.5)(fe1)

#	fe2 = Dense(256, activation='relu')(fe1)

	# MERGE VS. INJECT MODEL
	if model_type == 'M':
		print "Defining merge model"
		# SEQUENTIAL MODEL
		seq_in_size = len(model_seq_in)		# 1 || 2
		inputs2 = Input(shape=(1,seq_in_size))		# shape = (timesteps, featdim)
		#se1 = Dense(256, activation='relu')(inputs2)
		#se2 = Dropout(0.5)(se1)
		#se3 = LSTM(256)(se2)
		se3 = LSTM(seq_neurons)(inputs2)   # 256

		# INJECT MODEL!!
		#input_state = Input(shape=(15,))
		#state_init = Dense(seq_neurons,activation='tanh')(input_state)
		#ss = LSTM(128)(inputs2,initial_state = [state_init,state_init])

		# decoder model
	#	decoder1 = add([fe2, se3])
		decoder1 = concatenate([fe1, se3])
		#decoder2 = Dense(256, activation='relu')(decoder1)
	"""
	elif model_code == 'IN-I':		# Init-Inject model
	elif model_code == 'PR-I':		# Pre-Inject model
	elif model_code == 'PA-I':		# Par-Inject model
	"""

	outputs = Dense(1)(decoder1)
	# tie it together [image, shape+seq] [seq]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='mean_squared_error', optimizer='adam')
#	model.compile(loss=root_mean_squared_error, optimizer='rmsprop')
	# summarize model
	summary = model.summary()
	print(summary)
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model, summary


def old_get_batch(im_idx,x, label):
	IN1 = []
	IN2 = []
	OUT = []
	seq = sequences[im_idx]
	seq = np.insert(seq,0,0)
	for s_i, s in enumerate(seq[:-1]):
		feat = np.concatenate((x, centroids[label]))

		IN1.append(feat)
		IN2.append(s)
		OUT.append(seq[s_i+1])
	IN2 = np.array(IN2)
	IN2 = IN2.reshape(IN2.shape[0],1,1)	# reshape LSTM input to (samples,time steps,features)
	return np.array(IN1), np.array(IN2), np.array(OUT)

#  f: [x,shape]	   seq starts from 0
def get_batch2(feat,seq, model = 'model1'):
	IN1 = []
	IN2 = []
	OUT = []
	if len(seq) !=30:
		print "maggiore di 30"
		print seq
		sys.exit(0)
	
	time_featdim = 1
	if model == 'model2':
		time_featdim = 2
		shape = feat[15:]
		feat = feat[:15]
#		shape = np.array(shape)
	feat = np.array(feat)
	"""
	print shape
	print seq
	print len(shape)  # 31
	print len(seq)    # 30
	print [shape[2]]
	"""
	for s_i, s in enumerate(seq):
		IN1.append(feat)
		if model == 'model2':
			if s_i == 0:
#				IN2.append(np.concatenate([START_VAL],[shape[s_i+1]]))
				IN2.append([START_VAL]+[shape[s_i+1]])
	#			print ([START_VAL],[np.min(shape)])
			else:
				#IN2.append(np.concatenate([seq[s_i-1]],[shape[s_i+1]]))   
				IN2.append([seq[s_i-1]]+[shape[s_i+1]])
		else:
			if s_i == 0:
				IN2.append(START_VAL)
			else:
				IN2.append(seq[s_i-1])   #    s_i-1   -->  s_i
		OUT.append(s)
	#print IN2
	#sys.exit(0)
	IN2 = np.array(IN2)
	IN2 = IN2.reshape(IN2.shape[0],1,time_featdim)	# reshape LSTM input to (samples,time steps,features)

	return np.array(IN1), np.array(IN2), np.array(OUT)


# mod: model code
# nseq_in: not-sequential input
# seq_in: sequential input (beside the groud truth sequence)
# out_seq: ground truth sequence
def get_batch(mod,nseq_in,seq_in,out_seq):
		#feat           # seq

	#[M|I]_NO_<SEQ_IN>_<NSEQ_IN>
	model_code = mod.split('_')
	model_type = model_code[0]
	model_no = model_code[1]
	model_seq_in = model_code[2]
	model_nseq_in = model_code[3]


	IN1 = []
	IN2 = []
	OUT = []
	if len(seq) !=30:
		print "maggiore di 30"
		print seq
		sys.exit(0)
	
	time_featdim = len(model_seq_in)    # 1 oppure 2
	feat = np.array(nseq_in)
	seq_in = np.array(seq_in)
	"""
	print seq_in
	print out_seq
	print len(seq_in)  # 31 , 30 se differenziata (oppure 0)
	print len(out_seq)    # 30
	"""
	for s_i, s in enumerate(out_seq):
		IN1.append(feat)
		if len(seq_in)>0:
			seq_feat =  seq_in[s_i]			#  ex s_i+1
			if s_i == 0:
				IN2.append([START_VAL]+[seq_feat])			
	#			print ([START_VAL],[np.min(shape)])

			else:
				IN2.append([seq[s_i-1]]+[seq_feat])
		else:
			if s_i == 0:
				IN2.append([START_VAL])			
			else:
				IN2.append([seq[s_i-1]])
	#	print IN2[s_i]
		OUT.append(s)
#	print IN2
	#sys.exit(0)
	IN2 = np.array(IN2)
	IN2 = IN2.reshape(IN2.shape[0],1,time_featdim)	# reshape LSTM input to (samples,time steps,features)

	return np.array(IN1), np.array(IN2), np.array(OUT)


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
#### SETTINGS ####
#	Model code:	[M|I]_NO_<SEQ_IN>_<NSEQ_IN>
#
#		M|I	Merge or Inject model
#		NO	version number
#		SEQ_IN	sequential input (v:views, s:shape, x:social, d:deep visual)
#		NSEQ_IN	not-sequential input (v:views, s:shape, x:social, d:deep visual)
#		
#		Example:   M_01_vs_x
#
MOD = 'M_01_vs_x'
print MOD
VERBOSE = True
K = 50  # 40
NUM_TRIALS = 1
n_epochs = 1000
# Input 01:	seq_in: views(v), nseq_in: social(x)
PREPROCESSING = { 'v':
			{
				'STATIONARITY_NORM' : True,
				'SCALED' : True,
				'START_VAL' : -1.0
			},
		  's':
			{
				'STATIONARITY_NORM' : True,
				'SCALED' : True,
				'START_VAL' : -1.0
			},
		  'x':
			{
				'STATIONARITY_NORM' : True,
				'SCALED' : True
			},
		  'd':
			{
				'STATIONARITY_NORM' : True,
				'SCALED' : True
			}
}
			



#	- OUTPUT SETTINGS
OUT_DIR = MOD+'_results'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

START_VAL = -1.0
#### END SETTINGS ####
"""
model_type = 'model2'
if SCALED:
	START_VAL = -1.0
else:
	START_VAL = 0.0

if model_type == 'model2':
	LSTM_model = define_model2()
else:
	LSTM_model = define_model()
"""
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

flickr_ids = flickr_ids[:1000]
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
#outliers = [idx for idx, s in enumerate(sequences) if s[-1] < 3]
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

#		get_batch() si aspetta l'input fisso e quello sequenziale


#	TRAIN DATA
train_features = []
train_in_seq = []
train_sequences = []
for t_idx, x in enumerate(X_train_o):

#	- SEQUENTIAL INPUTS
	seq_inputs = MOD.split('_')[2]

	seq_in = np.array([])
	for feat_name in seq_inputs:
		if feat_name == 'v':		# views: aggiunte di default da get_batch
			continue
		if feat_name == 's':
			l = train_shape_ls[t_idx]
			ss = centroids[l]
			ss = np.array(ss)		# shape
			#print ss

		# remove stationarity 	
		if PREPROCESSING[feat_name]['STATIONARITY_NORM']:
			ss = difference(ss)		
		# TO CHECK: attenzionare la concatenazione dell'input SEQUENZIALE (zip?)..	
		seq_in = np.concatenate((seq_in,ss))

	train_in_seq.append(seq_in)

#	- NOT-SEQUENTIAL INPUT
	nseq_inputs = MOD.split('_')[3]

	feat = np.array([])
	for feat_name in nseq_inputs:
		# social feat
		if feat_name == 'x':
			ff = np.array(x)			
		# deep (TODO)
		#if feat_name == 'd':
			#ff = ...
		feat = np.concatenate((feat,ff))
	 
#	feat = np.concatenate((x, shape))
	train_features.append(feat)


	# LSTM OUTPUT SEQUENCE (inserita anche come input sequenziale)
	seq = y_train[t_idx]
	seq = np.insert(seq,0,0)	# se non inserisco lo zero il primo valore è negativo... valutare...
	# remove stationarity 	
	if PREPROCESSING['v']['STATIONARITY_NORM']:
		diff_seq = difference(seq)			# se eseguo 'difference' la sequenza avra' un valore in meno (lo scaler imparera' da queste sequenze)
		train_sequences.append(diff_seq)
	else:
		train_sequences.append(seq)

#	print train_features[0]
#	print train_in_seq[0]
#	print train_sequences[0]


# FIT FEATURE SCALER  
scaler = preprocessing.StandardScaler().fit(train_features)	# mean variance scaler
X_train = scaler.transform(train_features)


# FIT IN SEQUENCE SCALER 
if len(train_in_seq[0])>0:
	in_seq_scaler = MinMaxScaler(feature_range= (-1,1))
	in_seq_scaler = in_seq_scaler.fit(train_in_seq)
	IN_SEQ_train =  in_seq_scaler.transform(train_in_seq)

# FIT OUT SEQUENCE SCALER 
seq_scaler = MinMaxScaler(feature_range= (-1,1))
seq_scaler = seq_scaler.fit(train_sequences)
OUT_SEQ_train =  seq_scaler.transform(train_sequences)


# TEST DATA
test_features = []
test_in_seq = []
test_sequences = []
for t_idx, x in enumerate(X_test_o):

#	- SEQUENTIAL INPUTS
	seq_inputs = MOD.split('_')[2]

	seq_in = np.array([])
	for feat_name in seq_inputs:
		if feat_name == 'v':		# views: aggiunte di default da get_batch
			continue
		if feat_name == 's':
			l = test_shape_ls[t_idx]
			ss = centroids[l]
			ss = np.array(ss)		# shape
			#print ss

		# remove stationarity 	
		if PREPROCESSING[feat_name]['STATIONARITY_NORM']:
			ss = difference(ss)			
		seq_in = np.concatenate((seq_in,ss))

	test_in_seq.append(seq_in)

#	- NOT-SEQUENTIAL INPUT
	nseq_inputs = MOD.split('_')[3]

	feat = np.array([])
	for feat_name in nseq_inputs:
		# social feat
		if feat_name == 'x':
			ff = np.array(x)			
		# deep (TODO)
		#if feat_name == 'd':
			#ff = ...
		feat = np.concatenate((feat,ff))
	 
	test_features.append(feat)

# Transform the test features according to the learned scaler
X_test = scaler.transform(test_features)
# Transform sequential input for the test set
if len(test_in_seq[0])>0:
	IN_SEQ_test =  in_seq_scaler.transform(test_in_seq)


#
#  MODEL DEFINITION
#

LSTM_model, m_summary = define_model(MOD, seq_neurons= 128, nseq_neurons = 128)

#	SUMMARY:
#	- train/test flickr ids:				train_flickr_ids, test_flickr_ids
#	- train/test not-sequential features (scaled):		X_train, X_test
#	- train/test sequential input (preprocessed):		IN_SEQ_train, IN_SEQ_test
#	- train ground truth sequences (preprocessed):		OUT_SEQ_train
#

train_loss = []
train_val = []
train_size =len(X_train)
for ep in range(n_epochs):

    #TRAINING
    for t_idx, x in enumerate(X_train):   
	img_id = train_flickr_ids[t_idx]


#	LSTM_X1, LSTM_X2, LSTM_Y =  get_batch(idx,x,l)    # [X,p_t] [p_t+1]
	seq = OUT_SEQ_train[t_idx]
	"""
	print "Ground Truth:"
	print y_train[t_idx]
	print "Transformed:"
	print seq
	predicted = seq_scaler.inverse_transform([seq])
	predicted = predicted[0]
	predicted = inverse_difference(predicted)	
	print "Reversed:"
	print predicted
	"""
	in_seq = []
	if len(train_in_seq[0])>0:
		in_seq = IN_SEQ_train[t_idx]
        #get_batch(mod,nseq_in,seq_in,out_seq):
	if PREPROCESSING['v']['STATIONARITY_NORM']:
		LSTM_X1, LSTM_X2, LSTM_Y =  get_batch(MOD,x,in_seq,seq)    
	else:
		LSTM_X1, LSTM_X2, LSTM_Y =  get_batch(MOD,x,in_seq,seq[1:])    # si aspetta features gia concatenate e normalizzate (senza il primo valore, aggiunge lui START value)
   	# fit batch
	batch_size = len(LSTM_X1)
	print "\nEpoch:\t" + str(ep) + "/"+str(n_epochs)+"\t\tdata #\t" + str(t_idx) + "/" +str(train_size)
	hist = LSTM_model.fit([LSTM_X1,LSTM_X2],LSTM_Y, epochs=1,batch_size=batch_size,shuffle=False) #,callbacks=[checkpoint])
        train_loss.append(np.mean(hist.history['loss']))

	# reset_states
	LSTM_model.reset_states()


	# DEBUG OUT
	if ep > 5 and ep % 50 == 0: #and t_idx % 10 == 0:
		s = sequences[t_idx]
		s = np.array(s)
		print "\n"
			
		pred = START_VAL
		#print str(pred) + "\t" + "("+str(START_VAL)+")"
#			pred_s = [START_VAL]
		pred_s = []
		for d in range(30):
			in2 = [pred]+[LSTM_X2[d,0,1]]
			in2 = np.array(in2)
			pp = LSTM_model.predict([LSTM_X1[0].reshape(1,15),in2.reshape(1,1,2)])
#			pp = LSTM_model.predict([LSTM_X1[0].reshape(1,46),np.array([pred]).reshape(1,1,1)])
			pp = pp[0,0]
			pred_s.append(pp)
#			print str(pp) + "\t" + "("+str(s[d])+")"
			pred = pp
		
		s = np.insert(s,0,0)
		#print len(pred_s)
		# reverse scaling
#			predicted = seq_scaler.inverse_transform(np.array(pred_s).reshape(1,len(pred_s)))
		
		if SCALED:

			if not PREPROCESSING['v']['STATIONARITY_NORM']:
				pred_s = np.insert(np.array(pred_s),0,START_VAL)

			predicted = seq_scaler.inverse_transform([pred_s])
			predicted = predicted[0]
		else:
			predicted = pred_s
		
		
		if PREPROCESSING['v']['STATIONARITY_NORM']:
			predicted = inverse_difference(predicted)		# questa funzione ripristina il primo valore
#			else:
#				predicted = np.insert(np.array(predicted),0,0)
		
		#print len(predicted)
		#print predicted
		

		SE, _ = performance(predicted,s)
		e = np.sqrt(np.mean(SE))
		print "RMSE:\t" + str(e)
		if True and e < 2:
			plt.figure()
			plt.title('FlickrId: ' + img_id)
			plt.plot(s,label='views sequence', linewidth=3.0)
			plt.plot(predicted,label='predicted')
			plt.legend(loc='best')
			plt.draw()
			plt.savefig('train_'+MOD+'_ep'+str(ep)+'_'+img_id+'.png')
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
