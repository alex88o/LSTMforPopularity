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

#from __future__ import print_function
import sys
from sys import argv
import json
#from scipy.io import loadmat
#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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

parser = SafeConfigParser()

config_path = os.path.dirname(os.path.realpath(sys.argv[0]))
config_path = config_path + '/configuration.ini'
parser.readfp(open(config_path))
# Read configuration parameters
machine = parser.get('general','machine_name')

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
	
#	feat = np.zeros(len(res)+1)
	feat = []    # np.zeros(len(res))
	#print res[0]
	#sys.exit(0)

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
	#		if VERBOSE:
	#			print "FlickrId\t:\t"+flickr_id+"\tin dataset A."
			HD = HD_A
			db_path = path_A
		except ValueError:
			im_idx = DS_B.index(flickr_id)		
	#		if VERBOSE:
	#			print "FlickrId\t:\t"+flickr_id+"\tin dataset B."
			HD = HD_B
			db_path = path_B

		user_id = HD[im_idx][1]
	#	if VERBOSE:
	#		print "Getting user info..."
		if reducedUserInfo:
  		    user_feat = get_user_info_reduced(db_path+'/user_info.db',user_id)
  		else:
  		    user_feat = get_user_info(db_path+'/user_info.db',user_id)
	#	if VERBOSE:
	#		print "Getting image info..."
	        if reducedPhotoInfo:
      		    image_feat, image_info = get_image_info_reduced(db_path+'/image_info.db',flickr_id)
		else:
		    image_feat, image_info = get_image_info(db_path+'/image_info.db',flickr_id)
			
		data.append(list(user_feat) + list(image_feat))
	#	data.append(list(user_feat))

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

def load_popularity_seq(flickr_ids,day=30):

	data = []
	with open('ds_'+str(day)+'_days.pickle','r') as f:
		sequences = pickle.load(f)
		id_seq = [x[0] for x in sequences]
		for flickr_id in flickr_ids:
			idx = id_seq.index(flickr_id)
			seq = sequences[idx][2]
			"""
			print "id\t"+flickr_id
			print "idx\t"+str(idx)
			print seq
			print "\n"
			sys.exit(0)
			"""
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


#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#

#### SETTINGS ####
dbs_A_path = '/home/jolwave/PopularityChallenge/first_day_DSA'
dbs_B_path = '/home/jolwave/PopularityChallenge/first_day_DSB'
dbs_A_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification\\first_day_DSA'
dbs_B_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification\\first_day_DSB'

clustering_results_path = '/home/jolwave/PopularityChallenge/statistics/clustering output/clustering_'+str(K)+'_scaled.pickle'
clustering_results_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification\statistics\clustering output\clustering_'+str(K)+'_scaled.pickle'

shape_classifier_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification\\rndforest_20_class.pkl'
#feats_path = 'C:\Users\\aless\Documents\Dottorato di Ricerca\PopularityChallenge\george\classification'

VERBOSE = True
K = 50  # 40
NUM_TRIALS = 1
n_jobs = 1
#### END SETTINGS ####
    
with open(clustering_results_path,'r') as f:
	clusters = pickle.load(f)
flickr_ids = clusters['images']
labels = clusters['kmeans_out'].labels_
centroids  = clusters['kmeans_out'].cluster_centers_
last_day = 30
centroids = [c[:last_day+1] for c in centroids]
sequences = load_popularity_seq(flickr_ids, day=last_day)
#outliers = [idx for idx, s in enumerate(sequences) if s[-1] <5]
with open(shape_classifier_path,'r') as f:  
    classifier = pickle.load(f)
    
# Social features
X = load_features(flickr_ids,dbs_A_path,dbs_B_path)
Y = [pop_score(x[-1],last_day) for x in sequences]            # Khosla's pop score at day 30


for i in range(NUM_TRIALS):

    ##### PREPROCESSING #####            
    #Shuffle the train/test features
    print "New data splitting..."
    X_train_o, X_test_o, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    # Fit the scaler with the train features
    scaler = preprocessing.StandardScaler().fit(X_train_o)
    X_train = scaler.transform(X_train_o)
    # Transform the test features according to the learned scaler
    X_test = scaler.transform(X_test_o)	
 
 
    # Find the train/test data indices 
    test_idx_order = []
    idx_list = range(len(X))
    for el_i, el in enumerate(X_test_o):
        FOUND = False
        for x_i in idx_list:
            x = X[x_i]
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
    
 
 
    # Predicted shape prototypes (test data)
    test_shape_ls = classifier.predict(X_test)
    train_shape_ls = []
    # Clustered shape prototypes (train data)
    for t_idx in range(len(X_train)):   
        idx = train_idx_order[t_idx]
        img_id = flickr_ids[idx]
        l = labels[idx]                     # centroids[l]  ---> shape prototype
        train_shape_ls.append(l)
        


    # prepare train/test data for LSTM
