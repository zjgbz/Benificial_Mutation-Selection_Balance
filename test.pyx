import numpy as np
import collections
import math
import h5py
import os
cimport numpy as np
from libc.stdlib cimport malloc, free
import csv

DTYPE = np.double
ctypedef np.double_t DTYPE_t

DTYPE1 = np.int
ctypedef np.int_t DTYPE1_t
 
cdef int DIM = 30
cdef int id = 0

def initial(DTYPE_t N):
	global DIM
	cdef np.ndarray[DTYPE_t, ndim=2] n = np.zeros((DIM, 2))
	n[0, 0] = N
	n[0, 1] = 1.
	return n

def offspring(np.ndarray[DTYPE_t, ndim=2] n):
	cdef int length = n.shape[0]
	weight_array = np.zeros((length,1))
	mut_sum_real = np.dot(n[:,1],n[:,0])
	for i in range(0, length):
		weight_array[i,0] = n[i,1]*n[i,0]/mut_sum_real
	offspring_gene = np.random.choice(np.arange(length),size=int(np.sum(n[:,0])),replace=True,p=weight_array[:,0])
	offspring_distribution = collections.Counter(offspring_gene)
	for i in range(0, length):
		n[i,0] = offspring_distribution[i]
	#	birth_rate = n[i,1]
	#	num = n[i,0]
	#	num_offspring = np.random.poisson(num * birth_rate, 1)
	#	n[i, 0] = num_offspring
	return n

def mutation(np.ndarray[DTYPE_t, ndim=2] n, double Ub, double s):
	global DIM
	global id
	cdef int length = n.shape[0]
	cdef np.ndarray[DTYPE1_t, ndim=1] num_mut
	mut_list = []
	for i in range(0, length):
		num = int(n[i,0])
		num_mut = np.random.poisson(Ub, num)
	#	num_mut_pre = np.floor(np.random.uniform(0,1,num) + Ub * np.ones(num))
	#	num_mut = num_mut_pre.astype(int)
		if num_mut.shape[0]>0 and i + 1 > DIM-1:
			DIM += 10
			id = 1
		mut_list.append(num_mut)
	cdef int *new = <int *>malloc(DIM * sizeof(int))
	n = np.zeros((DIM,2))
	for i in range(DIM):
		new[i] = 0
	cdef int mut_sum = 0
	cdef int tmp_num = 0
	for i in range(0, length):
		for mut in mut_list[i]:
			new[i+mut] += 1
		tmp_num = new[i]
		n[i,0] = tmp_num
		mut_sum += i*tmp_num
	for i in range(length, DIM):
		tmp_num = new[i]
		n[i,0] = tmp_num
		mut_sum += i*tmp_num
	total_num = np.sum(n[:,0])
	free(new)
	for i in range(0, DIM):
	#	birth_rate = n[i,0]/total_num*(1. + s * (i - mut_sum / total_num))
		birth_rate = 1. + s * (i - mut_sum / total_num)
		n[i,1] = birth_rate if (birth_rate>0) else 0
	return n

def info(np.ndarray[DTYPE_t, ndim=2] n):
	N = np.sum(n[:,0])
	mut_list = np.argwhere(n[:,0]>0)
	min = np.min(mut_list)
	max = np.max(mut_list)
	return N,min,max

#if __name__ == '__main__':
def run(DTYPE_t N=10**5, double Ub=10.**(-5), double s=0.01, int generation = 3600, suffix=""):
	global DIM
	global id
	file_name = "./smb_N%d_G%d-%s"%(N, generation, suffix)
	print('Parameters:\nN=%d\tUb=%.2e\ts=%f\tgeneration=%d'%(N,Ub,s,generation))
	print('Data are saved in \'%s.hdf5\'\n'%file_name)
	if  os.path.exists('%s.hdf5'%file_name):
		print('existed')
		return 1
	f = h5py.File('%s.hdf5'%file_name, 'w')
	f.attrs['N'] = N
	f.attrs['Ub'] = Ub
	f.attrs['s'] = s
	f.attrs['generation'] = generation
	data = f.create_dataset('data', (generation+1, DIM, 2), dtype='double', maxshape=(generation+1, None, 2))
	print('Initializing...')
	cdef np.ndarray[DTYPE_t, ndim=2] n = initial(N)
	data[0, :] = n
	f.close()
	print('N=%d, min_mut=%d, max_mut=%d'%info(n))
	for i in range(0, generation):
		n = offspring(n)
		total = np.sum(n[:,0])
		if total<=0:
			f = open('death', 'a+')
			writer = csv.writer(f)
			row = [file_name, i]
			writer.writerow(row)
			f.close()
			return 0
		n = mutation(n, Ub, s)
		f = h5py.File('%s.hdf5'%file_name, 'a')
		data = f['data']
		if id == 1:
			data.resize(DIM, 1)
			id = 0
		data[i+1, :] = n
		f.attrs['finished'] = i+1
		f.close()
		if i%100 == 0:
			print('Generation %d finised'%(i+1))
			print('N=%d, min_mut=%d, max_mut=%d'%info(n))
			print('Running...')
	print('Finished')
	print('N=%d, min_mut=%d, max_mut=%d'%info(n))
	return 0
