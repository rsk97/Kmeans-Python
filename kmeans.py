import numpy as np
import os,sys,random,math
import pandas as pd

np.random.seed(200)

def allot_clust(data,clust,data_len,belongs_to):
	df = []
	for i in clust.keys():
			df.append(np.sqrt(sum(((data - clust[i])**2).transpose()).transpose()))
	df = np.asarray(df,dtype = np.float32).transpose()
	k = [np.argmin(i) +1 for i in df]
	for i,j in zip(k,range(data_len)):
		belongs_to[i].append(data[j])
	for i in clust.keys():
		clust[i] = np.mean(np.asarray(belongs_to[i],dtype = np.float32).transpose(),axis = 1)

def normalize(data):
	mi = np.min(data,axis = 0)
	mx = np.max(data,axis = 0)
	return (data - mi)/(mx - mi)

def check_equal(a,b):
	for i in a.keys():
		if not(np.array_equal(a[i],b[i])):
			return 0
	return 1

def main():
	if len(sys.argv) < 3 or len(sys.argv) > 4:
	    print "TYPE: python kmeans.py filename no_of_clusters OR python kmeans.py filename no_of_clusters outputfilepath"
	    exit(1)
	else:
		data = []
		file = sys.argv[1]
		cluster = int(sys.argv[2])
		out_file = sys.argv[3] if len(sys.argv) == 4 else "output.txt"
        with open(file,'r') as f:
			d = f.readlines()
			for i in d:
				data.append(i.strip().split(" "))
	data = np.asarray(data,dtype = np.float32)
	no_of_Samp,dimension = data.shape[0],data.shape[1]
	data = normalize(data)
	if no_of_Samp < cluster:
		print "No of clusters more tha samples....aborting"
		exit(1)
	clust = {i+1 : data[np.random.randint(data.shape[0],size = 1),:] for i in range(cluster)}
	belongs_to = {i+1 : [] for i in range(cluster)}
	while True:
		belongs_to = {}
		belongs_to = {i+1 : [] for i in range(cluster)}
		old_clust = dict(clust)
		allot_clust(data,clust,no_of_Samp,belongs_to)
		if check_equal(old_clust,clust):
			with open(out_file,'w') as f:
				for i in clust.keys():
					f.write(str(clust[i])[1:-1])
					f.write("\n")
			break

if __name__ == '__main__':
	main()
