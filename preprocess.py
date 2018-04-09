import os
import sys
import numpy as np
import nibabel as nib

def convert_slice(data, n):
	l = []
	# print data.shape
	data_new = np.pad(data, ((0,0), (0, 0), (n, n), (0,0)), 'constant', constant_values=0)
	# print data_new.shape
	for i in range(n, data_new.shape[2]-n):
		l.append(data_new[:,:,i-n:i+n+1,:])
	return l

def convert_seg(data):
	l=[]
	for i in range(0, data.shape[2]):
		l.append(data[:,:,i,:])
	return l

#TRAIN
train_x_dir = "Cerebellum_IBSR/brain/train/"
train_y_dir = "Cerebellum_IBSR/labels/train/"

train_x_samples = []
train_y_samples = []

lst = os.listdir(train_x_dir)
lst.sort()

l=[]
for file in lst:
	img = nib.load(train_x_dir + file)
	data = img.get_data()
	l = convert_slice(data, 25)
	train_x_samples+=(l)

lst = os.listdir(train_y_dir)
lst.sort()

l=[]
for file in lst:
	img = nib.load(train_y_dir + file)
	data = img.get_data()
	l = convert_seg(data)
	train_y_samples+=(l)

train_x = np.asarray(train_x_samples)
train_y = np.asarray(train_y_samples)

train_y[train_y==7]=1
train_y[train_y==8]=1
train_y[train_y==46]=1
train_y[train_y==47]=1
train_y[train_y!=1]=0
print train_x.shape, train_y.shape



def write_numpy(s):
	x_dir = "Cerebellum_IBSR/brain/"+s+"/"
	y_dir = "Cerebellum_IBSR/labels/"+s+"/"

	x_samples = []
	y_samples = []

	lst = os.listdir(x_dir)
	lst.sort()

	l=[]
	for file in lst:
		img = nib.load(x_dir + file)
		data = img.get_data()
		l = convert_slice(data, 25)
		x_samples.append(l)

	lst = os.listdir(y_dir)
	lst.sort()

	l=[]
	for file in lst:
		img = nib.load(y_dir + file)
		data = img.get_data()
		l = convert_seg(data)
		y_samples.append(l)

	x = np.asarray(x_samples)
	y = np.asarray(y_samples)
	print x.shape, y.shape
	print x[0].shape, y[0].shape

	# for i in range(0, x.shape[0]):
		# np.savez("./Data/"+s+"/"+str(i)+".npz", x = x[i], y = y[i])

print "Writing train to disk"
# np.savez("./Data/train.npz", x = train_x, y = train_y)
print "Writing train"
write_numpy("train")
# print "Writing val"
# write_numpy("val")
# print "Writing test"
# write_numpy("test")