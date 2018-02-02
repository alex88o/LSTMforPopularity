from numpy import array
from pickle import load
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

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model



# feature extractor model
#inputs1 = Input(shape=(46,))
inputs1 = Input(shape=(46,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence model
inputs2 = Input(shape=(1,1))		# shape = (timesteps, featdim)
#se1 = Dense(256, activation='relu')(inputs2)
#se1 = Embedding(1, 256, mask_zero=True)(inputs2)
#se2 = Dropout(0.5)(se1)
#se3 = LSTM(256)(se2)
se3 = LSTM(256)(inputs2)
# decoder model
decoder1 = add([fe2, se3])
#decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(1)(decoder1)
# tie it together [image, seq] [word]
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
#model.compile(loss='categorical_crossentropy', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')
# summarize model
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)


