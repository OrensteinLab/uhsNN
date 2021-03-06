import numpy as np
import pandas as pd
import itertools
import os
import argparse
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense,Masking,LSTM,GRU
from keras import metrics

from tensorflow.keras.callbacks import TensorBoard
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Trains the model for provided k-mer lengths (k0, k1).')
	parser.add_argument('-k0', metavar='k0', type=int, help='Minimum k-mer size for the UHS')
	parser.add_argument('-k1', metavar= 'k1', type=int, help='Maximum k-mer size for the UHS')
	parser.add_argument('-p', metavar='p', help='Model prefix')
	parser.add_argument('-e', metavar='e', help='Number of epochs')
	parser.add_argument('-b', metavar='b', help='Batch size')

args = parser.parse_args()
mink=args.k0
maxk=args.k1
maxmaxk=15

total_labels=np.empty(0,dtype=np.int8)
total_lst=np.empty([0,2*maxmaxk+2],dtype=np.int8)

for k in range (mink,maxk+1,1):
	#print(k)
	decycling=pd.read_csv("int/decyc"+str(k)+".int")
	lst = np.array(list(itertools.product([0, 1], repeat=2*k)))
	for L in range(20, 201, 10):
		#print(L)
		lst = np.array(list(itertools.product([0, 1], repeat=2*k)))
		padlst = np.pad(lst, ((0, 0),(0,maxmaxk*2-lst.shape[1])), 'constant', constant_values=((2, 2),(2,2)))
		lstL = np.append(np.full(lst.shape[0], L).reshape(lst.shape[0],1), padlst, axis=1)
		lstkL = np.append(np.full(lst.shape[0], k).reshape(lstL.shape[0],1), lstL, axis=1)
		labels=np.zeros(np.power(4,k), dtype=np.int8)
		labels[decycling]=2
		#print("PASHA"+str(k)+str(L)+".int")
		#print(os.path.isfile("int/PASHA"+str(k)+str(L)+".int"))
		if (os.path.isfile("int/PASHA"+str(k)+str(L)+".int")):
			docks=np.array(pd.read_csv("int/PASHA"+str(k)+str(L)+".int", header=None))
			labels[docks]=1
		combined = np.append(lstkL, labels.reshape(labels.shape[0],1), axis=1)
		combined = combined[np.logical_not(combined[:,2*maxmaxk+2]==2)]
		total_labels=np.append(total_labels, combined[:,2*maxmaxk+2],axis=0)
		total_lst=np.append(total_lst, combined[:,0:2*maxmaxk+2],axis=0)

#print(total_labels.shape)
#print(total_lst.shape)


# Define Tensorboard as a Keras callback
tensorboard = TensorBoard(
  log_dir='.\logs',
  histogram_freq=1,
  write_images=True
)
keras_callbacks = [
  tensorboard
]

model = Sequential()

model.add(Masking(mask_value=2., input_shape=(maxmaxk*2+2, 1)))
model.add(GRU(100, input_dim=1))
#model.add(Dense(100, input_dim=2*k+1, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.AUC(),'accuracy'])

model.fit(total_lst,
		  total_labels,
		  epochs=int(args.e),
		  batch_size=int(args.b),
		  callbacks=keras_callbacks)

#tensorboard --logdir=logs
model.save(str(args.p) + ".h5")
