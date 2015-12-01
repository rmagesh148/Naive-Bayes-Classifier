"""The below lines are the libraries"""
import sys
import collections
import csv, warnings
import numpy as np, scipy
from scipy.sparse import lil_matrix
from sklearn.metrics import confusion_matrix
import numpy.matlib
from scipy.stats import itemfreq

"""Reading all the input files:: The place to change the path"""
traindata=open('C:\\Users\\MaGesh\\Desktop\\data\\train.data','r');
testdata=open('C:\\Users\\MaGesh\\Desktop\\data\\test.data','r');
trainlabel=open('C:\\Users\\MaGesh\\Desktop\\data\\train.label','r');
testlabel=open('C:\\Users\\MaGesh\\Desktop\\data\\test.label','r');
voc=open('C:\\Users\\MaGesh\\Desktop\\data\\vocabulary.txt','r');
trainlabel_np=np.loadtxt(trainlabel)
traindata_np=np.loadtxt(traindata)
testlabel_np=np.loadtxt(testlabel)
testdata_np=np.loadtxt(testdata)
voc_np=np.loadtxt(voc,dtype=str)
n=len(traindata_np)                                                     #length of the training data
vc=np.zeros((20,61188))                                                 #Initalizing the sparse matrix for finding the MAP
for i in range(0,n):
        with warnings.catch_warnings(record=True) as warning_list:      #using the sklearn it gives me the Deprecation warning, so catching this here in two lines
            warnings.simplefilter("ignore", DeprecationWarning)
            vc[trainlabel_np[traindata_np[i,0]-1]-1,traindata_np[i,1]-1]=vc[trainlabel_np[traindata_np[i,0]-1]-1,traindata_np[i,1]-1]+traindata_np[i,2]

maximum_value_threshold=(vc.max(axis=0))                        #Maximum_value_threshold is an array which will have the maximum value of the coulmn 
array_threshold=np.mean(maximum_value_threshold)                #Finding the mean of the maximum value of the array, approximate value is 10.0
print "Average value of vc", array_threshold                    #Printing the value of the mean, which will act as a threshold value 
b=np.where(vc>array_threshold)                                  #Taking out the index of the values which are greater than threshold value
b1=b[1]                                                         #Taking out the coulmn indices 
new_matrix=[]                                                   #Defining new_matrix
rank_matrix=itemfreq(b1)                                        #Finding the item frequency of the coulmn indicies
l=len(rank_matrix)                                              #Finding the length of the rank_matrix
for i in range(0,l):                                             
        if ((rank_matrix[i,1]) < 3):                            #the word id shouldn't repeat in more than 3 documents 
                new_matrix.append(rank_matrix[i,0])             #if that passes, adding that indicies of that value to the array
words=voc_np[new_matrix]                                        #calling the words with the array got it from the previous step
tq=words.shape                                                    
print "Shape of the word output",tq                             #printing shape of the array 
print words[:100]                                               #Printing First 100 words
print words[-100:]                                              #Printing last 100 words
