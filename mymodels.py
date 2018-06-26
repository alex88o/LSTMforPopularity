import numpy as np

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
	if return_state:
		se3, h, c = LSTM(lstm_size, stateful=stateful, return_state=return_state)(inputs2)   # 256		
	else:
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
	out = Dense(1,activation='sigmoid')(decoder1)
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

	#non funziona..
	#max_out = Lambda(lambda x: K_BACKEND.max(x))([out,inputs2[:,:,0]])
	# Questa funziona, nel senso che viene accettato il modello definito in questo modo e keras avvia il training
	max_out = Lambda(lambda oi: K_BACKEND.maximum(oi[0], oi[1][:,:,0]),output_shape=lambda oi : oi[0])([out,inputs2])	
	
	# tie it together [image, shape+seq] [seq]
	model = Model(inputs=[inputs1, inputs2], outputs=max_out)
	model.compile(loss='mean_squared_error', optimizer='adam')
#	model.compile(loss=root_mean_squared_error, optimizer='rmsprop')
	# summarize model
	print(model.summary())
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model






