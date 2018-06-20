import pickle
import sqlite3 as lite
import numpy as np

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

def load_features(ids_list, path_A, path_B,reducedUserInfo=False, reducedPhotoInfo=False, verbose = False):

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

	if verbose:
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





def transform_data_to_supervised(X_o, shape_ls, centroids, y_o, norm_diff=False):
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
		if norm_diff:
			shape_diff = difference(ss)
			# NB: i prototipi (shape) hanno dimensione 31, quindi non inserisco lo zero iniziale 
		else:
			shape_diff = ss
	
		# load the views dynamic (sequential data)
		seq = y_o[t_idx]
		seq = np.insert(seq,0,0)
		if norm_diff:
			seq_diff = difference(seq)
			IN_SEQ_SET.append([0.0,0.0])	# inserisco questa coppia solo se norm_diff == True
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
