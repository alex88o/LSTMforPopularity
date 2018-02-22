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

def define_model():
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


#def define_model2(in1_neurons=64,lstm_size=):
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


def get_batch(im_idx,x, label):
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
MOD = '06'
VERBOSE = True
K = 50  # 40
NUM_TRIALS = 1
n_epochs = 1500
STATIONARITY_NORM = True
SCALED = True
model_type = 'model2'


#### END SETTINGS ####
if SCALED:
	START_VAL = -1.0
else:
	START_VAL = 0.0

if model_type == 'model2':
	LSTM_model = define_model2()
else:
	LSTM_model = define_model()
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

#flickr_ids = flickr_ids[:1000]
cluster_idx = [1,3,17,28,30,31,48]   # 9, 20, 32 ... 8, 21
#cluster_idx = []
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

#	- DATA PREPROCESSING
#	- SEQUENTIAL INPUT
#	- NOT-SEQUENTIAL INPUT

for i in range(NUM_TRIALS):

    ##### PREPROCESSING #####            
    #Shuffle the train/test features
    print "New data splitting..."
    X_train_o, X_test_o, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    # Fit the scaler with the train features
    scaler = preprocessing.StandardScaler().fit(X_train_o)
    X_train = scaler.transform(X_train_o)
    #  Transform the test features according to the learned scaler
    X_test = scaler.transform(X_test_o)	
    
    # Predicted shape prototypes (test data)
    test_shape_ls = classifier.predict(X_test)
 
    #free the memory
    X_test = None
    X_train = None
    scaler = None

    # Find the train/test data indices 
    test_idx_order = []
    idx_list = range(len(X))
    for el_i, el in enumerate(X_test_o):
        FOUND = False
        for x_i in idx_list:
            x = X[x_i]
	    x = np.array(x)
	    el = np.array(el)
            if all(x == el):                    # check whether the vector is the same
                if y_test[el_i] != Y[x_i]:      # check whether the label is the same
                    continue
                test_idx_order.append(x_i)
                idx_list.remove(x_i)            # remove to avoid duplicates
                FOUND = True
                break
        if not FOUND:
            print "test idx non trovato!"
            print x_i
            sys.exit(0)
            
    print "Test indices created"
    
    
    train_idx_order = []
    idx_list = [val for val in range(len(X)) if val not in test_idx_order]
    for el_i, el in enumerate(X_train_o):
        FOUND = False
        for x_i in idx_list:
            x = X[x_i]
	    x = np.array(x)
	    el = np.array(el)
            if all(x == el):
                if y_train[el_i] != Y[x_i]:
                    continue
                train_idx_order.append(x_i)
                idx_list.remove(x_i)
                FOUND = True
                break
        if not FOUND:
            print "train idx non trovato!"
            print x_i
            sys.exit(0)
        
    print "Train indices created"
    
 
#    train_shapes = []
    train_features = []
    train_sequences = []
    for t_idx, x in enumerate(X_train_o):
	idx = train_idx_order[t_idx]
	l = labels[idx]
	shape = centroids[l]
	#train_shapes.append(shape)
	
	# INPUT 1
	shape = np.array(shape)
	x = np.array(x)
	feat = np.concatenate((x, shape))
	train_features.append(feat)
	
	
	# INPUT 2 AND Y
	seq = sequences[idx]
	seq = np.insert(seq,0,0)
	# remove stationarity 	
	if STATIONARITY_NORM:
		diff_seq = difference(seq)			# se eseguo 'difference' la sequenza avra' un valore in meno (lo scaler imparera' da queste sequenze)
		train_sequences.append(diff_seq)
	else:
		train_sequences.append(seq)
		

    # FIT FEATURE SCALER  
    scaler = preprocessing.StandardScaler().fit(train_features)
    X_train = scaler.transform(train_features)
 
    # FIT SEQUENCE SCALER 
    if SCALED:
	    seq_scaler = MinMaxScaler(feature_range= (-1,1))
	    seq_scaler = seq_scaler.fit(train_sequences)
	    SEQ_train =  seq_scaler.transform(train_sequences)
    else:
	    SEQ_train = train_sequences

 
    test_features = []
    test_sequences = []
    for t_idx, x in enumerate(X_test_o):
	idx = test_idx_order[t_idx]
	l = test_shape_ls[t_idx] # same order than X_test_o
	shape = centroids[l]
	#train_shapes.append(shape)
	
	# INPUT 1
	shape = np.array(shape)
	x = np.array(x)
	feat = np.concatenate((x, shape))
	test_features.append(feat)
	
	# INPUT 2 AND Y
	seq = sequences[idx]
	seq = np.insert(seq,0,0)
	# remove stationarity 	
	if STATIONARITY_NORM:
		diff_seq = difference(seq)
		test_sequences.append(diff_seq)
	else:
		test_sequences.append(seq)

    # Transform the test features according to the learned scaler
    X_test = scaler.transform(test_features)
    # Transform test sequences
    if SCALED:
	    SEQ_test =  seq_scaler.transform(test_sequences)	
    else:
	    SEQ_test = test_sequences

    # prepare train/test data for LSTM
    # TRAIN:     [X, shape_prototype, seq_val]
    # TEST:      [X, predicted_shape, predicted_seq_val]

   # train_shape_ls = []

    # iterate EPOCHS   ----   OR
    loss_history = []
    train_size =len(X_train)
    for ep in range(n_epochs):

	    #TRAINING
	    # Clustered shape prototypes (train data)
	    for t_idx, x in enumerate(X_train):   
		idx = train_idx_order[t_idx]
		img_id = flickr_ids[idx]
#		l = labels[idx]                     # centroids[l]  ---> shape prototype
		#train_shape_ls.append(l)

#		LSTM_X1, LSTM_X2, LSTM_Y =  get_batch(idx,x,l)    # [X,p_t] [p_t+1]


		seq = SEQ_train[t_idx]
		if STATIONARITY_NORM:
			LSTM_X1, LSTM_X2, LSTM_Y =  get_batch2(x,seq,model_type)    
		else:
			LSTM_X1, LSTM_X2, LSTM_Y =  get_batch2(x,seq[1:],model_type)    # si aspetta features gia concatenate e normalizzate (senza il primo valore, aggiunge lui START value)
	

	   	# fit batch
		batch_size = len(LSTM_X1)
		print "\nEpoch:\t" + str(ep) + "\t\tdata #\t" + str(t_idx) + "/" +str(train_size)
		hist = LSTM_model.fit([LSTM_X1,LSTM_X2],LSTM_Y, epochs=1,batch_size=batch_size,shuffle=False) #,callbacks=[checkpoint])
		
		# reset_states
		LSTM_model.reset_states()


		# DEBUG OUT
		if ep > 5 and ep % 50 == 0 and t_idx % 10 == 0:
			s = sequences[idx]
			s = np.array(s)
			print "\n"
				
			pred = START_VAL
			#print str(pred) + "\t" + "("+str(START_VAL)+")"
#			pred_s = [START_VAL]
			pred_s = []
			for d in range(30):
				if model_type == 'model2':
					"""
					print LSTM_X2.shape
					print LSTM_X2[:3]
					print LSTM_X2[d,0,1]
					"""
					in2 = [pred]+[LSTM_X2[d,0,1]]
					in2 = np.array(in2)
					pp = LSTM_model.predict([LSTM_X1[0].reshape(1,15),in2.reshape(1,1,2)])
				else:
					pp = LSTM_model.predict([LSTM_X1[0].reshape(1,46),np.array([pred]).reshape(1,1,1)])
				pp = pp[0,0]
				pred_s.append(pp)
	#			print str(pp) + "\t" + "("+str(s[d])+")"
				pred = pp
			
			s = np.insert(s,0,0)
			#print len(pred_s)
			# reverse scaling
#			predicted = seq_scaler.inverse_transform(np.array(pred_s).reshape(1,len(pred_s)))
			
			if SCALED:
	
				if not STATIONARITY_NORM:
					pred_s = np.insert(np.array(pred_s),0,START_VAL)

				predicted = seq_scaler.inverse_transform([pred_s])
				predicted = predicted[0]
			else:
				predicted = pred_s
			
			
			if STATIONARITY_NORM:
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
	#			plt.draw()
				plt.savefig('prediction_'+MOD+'_ep'+str(ep)+'_'+img_id+'.png')
	#			plt.show()

	    # for each epoch --> loss
	    loss_history.append(hist.history['loss'])
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
			"""
			print LSTM_X2.shape
			print LSTM_X2[:3]
			print LSTM_X2[d,0,1]
			"""
		
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


