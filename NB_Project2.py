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
n=len(traindata_np)                                                  #length of the training data

"""Finding the Counts of the Train Label(Doc Ids), then hardcoding the counter values to find the Maximum Likelihood P(Y)"""
with open('C:\\Users\\MaGesh\\Desktop\\data\\train.label') as infile:
        counts = collections.Counter(l.strip() for l in infile)
print "Counts of the Training document, to find the MLE", counts
x=np.array([[480],[581],[572],[587],[575],[592],[582],[592],[596],[594],[598],[594],[591],[594],[593],[599],[545],[564],[464],[376]])
myInt=11269
c=x/float(myInt)

"""Method to Find the MAP P(X|Y)"""
vc=np.zeros((20,61188))                                                 #Initalizing the sparse matrix for finding the MAP
for i in range(0,n):
        with warnings.catch_warnings(record=True) as warning_list:      #using the sklearn it gives me the Deprecation warning, so catching this here in two lines
            warnings.simplefilter("ignore", DeprecationWarning)
            vc[trainlabel_np[traindata_np[i,0]-1]-1,traindata_np[i,1]-1]=vc[trainlabel_np[traindata_np[i,0]-1]-1,traindata_np[i,1]-1]+traindata_np[i,2]

"""beta is the constant and sum_of_words is counting the words in the Y Axis""" 
beta=1/(float(61188))
sum_of_words= vc.sum(axis=1)
denominator=sum_of_words+(beta*61188);

"""To find the P(Ynew)= argmax[(log2(P(Y)))+(summation (X)*log2(P(Xi|Yk)))]"""

with warnings.catch_warnings(record=True) as warning_list:       #using the sklearn it gives me the Deprecation warning, so catching this here in two lines
        warnings.simplefilter("ignore", DeprecationWarning)
        vp=np.zeros((20,61188))
        vp=vc+beta
        vp=vp.astype('float')
        for i in range(0,20):
                vp[i,:]=vp[i,:]/denominator[i]
        A=lil_matrix((7505,61188),dtype=np.float32) #lil_matrix is a sparse matrix used to build testdata matrix
        testdata_np=testdata_np.astype('float')          #conversion of testdata matrix as float
        A[testdata_np[:,0]-1,testdata_np[:,1]-1]=A[testdata_np[:,0]-1,testdata_np[:,1]-1]+testdata_np[:,2]
        A_Tran=A.transpose()          #matrix of 61188*7505
        vp1=np.log2(vp)               #matrix of 20*61188
        vp2=vp1*A_Tran                #matrix of 20*7505
        vp2=vp2+np.matlib.repmat(c,1,7505) #repmat is used to repeat the matrix 7505 times in order to get the index of maximum_value of each column
        maximum_value=(vp2.argmax(axis=0)+1)               #Matrix of 1*7505        
        c=confusion_matrix(testlabel_np,maximum_value)#Confusion matrix is used to find the number of classified documents, input given is testlabel and maximum_value
        print "Confusion Matrix\n",c  
        acc=sum(c.diagonal())/float(7505)    #diagonal values of confusion matrix will give you the number of correctly classified documents
        acc=acc*100
        print "Accuracy of the matrix",acc   #Accuracy is printed
