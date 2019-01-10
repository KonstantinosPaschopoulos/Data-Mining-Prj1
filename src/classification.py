import pandas as pd
import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import re
import string
import knn_functions as knn

def main():

#------------------------------DATA----------------------------------

    train_data=pd.read_csv('train_set.csv',sep="\t")
    test_data=pd.read_csv('test_set.csv',sep="\t")
    train_data.drop('RowNum',axis=1)		#ignore rownum
    test_data.drop('RowNum',axis=1)

#------------------------------Processing----------------------------

    extra_words=["said","say","seen","come","end","came","year","years","new","saying"]		#extra stopwords
    stopwords=ENGLISH_STOP_WORDS.union(extra_words)
    tfidf=TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stopwords)		#convert to tf-idf
    tsvd=TruncatedSVD(n_components=200, algorithm='randomized', random_state=42)		#set dimensions

    set(train_data['Category'])		#check categories
    le=preprocessing.LabelEncoder()	#set labels
    le.fit(train_data["Category"])	#fit them to the number of our categories
    y_train=le.transform(train_data["Category"])	#transform categories
    set(y_train)

    count_vectorizer=CountVectorizer(stop_words=stopwords)	#set stopwords for vectorizer
    X_trainNoLSI=count_vectorizer.fit_transform(train_data['Content'])		#vectorize out data
    tsvd.fit(X_trainNoLSI)				#truncate data
    X_train=tsvd.transform(X_trainNoLSI)		#store them

    test_noLSI=count_vectorizer.transform(test_data['Content'])		#test data
    test=tsvd.transform(test_noLSI)

    k_fold = KFold(n_splits=10)				#10 fold validation

#--------------------------------SVM---------------------------------

    clf=svm.SVC(kernel='rbf', C=100, gamma='auto')		#algorithm for application
    clf.fit(X_train, y_train)
    y_pred=clf.predict(test)

#--------------------------------SVM_scores--------------------------
    print "SVM scores:"

    SVMprecs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='precision_micro')
    svm_prec=SVMprecs.mean()
    print "precision:" ,svm_prec

    SVMrecs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='recall_micro')
    svm_rec=SVMrecs.mean()
    print "recall:" ,svm_rec

    SVMfms=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='f1_micro')
    svm_fm=SVMfms.mean()
    print "F-measure:" ,svm_fm

    SVMaccs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='accuracy')
    svm_acc=SVMaccs.mean()
    print "accuracy:" ,svm_acc

#---------------------------------RF---------------------------------

    clf=RandomForestClassifier(max_depth=6,random_state=1)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(test)

#---------------------------------RF_scores--------------------------

    print "RF scores:"

    RFprecs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='precision_micro')
    rf_prec=RFprecs.mean()
    print "precision:" ,rf_prec

    RFrecs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='recall_micro')
    rf_rec=RFrecs.mean()
    print "recall:" ,rf_rec

    RFfms=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='f1_micro')
    rf_fm=RFfms.mean()
    print "F-measure:" ,rf_fm

    RFaccs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='accuracy')
    rf_acc=RFaccs.mean()
    print "accuracy:" ,rf_acc

#----------------------------------MNB--------------------------------

    clf=MultinomialNB()
    clf.fit(X_trainNoLSI,y_train)
    y_pred=clf.predict(test_noLSI)

#----------------------------------MNB_scores-------------------------

    print "MNB scores:"

    MNBprecs=cross_val_score(clf, X_trainNoLSI, y_train, cv=k_fold, scoring='precision_micro')
    mnb_prec=MNBprecs.mean()
    print "precision:" ,mnb_prec

    MNBrecs=cross_val_score(clf, X_trainNoLSI, y_train, cv=k_fold, scoring='recall_micro')
    mnb_rec=MNBrecs.mean()
    print "recall:" ,mnb_rec

    MNBfms=cross_val_score(clf, X_trainNoLSI, y_train, cv=k_fold, scoring='f1_micro')
    mnb_fm=MNBfms.mean()
    print "F-measure:" ,mnb_fm

    MNBaccs=cross_val_score(clf, X_trainNoLSI, y_train, cv=k_fold, scoring='accuracy')
    mnb_acc=MNBaccs.mean()
    print "accuracy:" ,mnb_acc

#-----------------------------------K-Nearest_Neighbor------------------

    clf=knn.myKNN(10)			# K=10,check knn_functions.py(imported)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(test)

#---------------------------------KNN_scores--------------------------

    print "KNN scores:"

    KNNprecs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='precision_micro')
    knn_prec=KNNprecs.mean()
    print "precision:" ,knn_prec

    KNNrecs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='recall_micro')
    knn_rec=KNNrecs.mean()
    print "recall:" ,knn_rec

    KNNfms=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='f1_micro')
    knn_fm=KNNfms.mean()
    print "F-measure:" ,knn_fm

    KNNaccs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='accuracy')
    knn_acc=KNNaccs.mean()
    print "accuracy:" ,knn_acc

#----------------------------------------------------------------------
#                                   My Method
#----------------------------------------------------------------------
    #our method
    #data punctuation
    test_data['Content']=test_data['Content'].str.replace('[^\w\s]', '')
    train_data['Content']=train_data['Content'].str.replace('[^\w\s]', '')
    #convert multiple spaces to one
    test_data['Content']=test_data['Content'].str.replace('\s+', ' ')
    train_data['Content']=train_data['Content'].str.replace('\s+', ' ')

    #same process as before
    set(train_data['Category'])
    le=preprocessing.LabelEncoder()
    le.fit(train_data["Category"])
    y_train=le.transform(train_data["Category"])
    set(y_train)

    X_train=count_vectorizer.fit_transform(train_data['Content'])

    test=count_vectorizer.transform(test_data['Content'])
    #usage of MNB
    max=0.0
    maxi=0.0
    i=0.01
    #search for the best smoothing parameter(alpha)
    while i<1.0:
        clf=MultinomialNB(alpha=i)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(test)
        myprecs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='precision_micro')
        my_prec=myprecs.mean()
        if my_prec>max:
            max=my_prec
            maxi=i
        i+=0.01
    print "My Method scores:"

    clf=MultinomialNB(alpha=maxi, fit_prior=True)
    clf.fit(X_train,y_train)
    the_pred=clf.predict(test)

    print "precision:" ,max

    myrecs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='recall_micro')
    my_rec=myrecs.mean()
    print "recall:" ,my_rec

    myfms=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='f1_micro')
    my_fm=myfms.mean()
    print "F-measure:" ,my_fm

    myaccs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='accuracy')
    my_acc=myaccs.mean()
    print "accuracy:" ,my_acc

#------------------------------------CSV---------------------------------
    #my method csv
    output='testSet_categories.csv'
    predicted=le.inverse_transform(the_pred)
    testingfile=pd.DataFrame({'ID': test_data['Id'], 'Predicted_Category': list(predicted)}, columns=['ID', 'Predicted_Category'])
    testingfile.to_csv(output,encoding='utf-8',index=False,sep='\t')
    #results csv
    output='EvaluationMetric_10fold.csv'
    d={'StatisticMeasure': ['Accuracy','Precision','Recall','F-Measure'],'Naive Bayes':[mnb_acc,mnb_prec,mnb_rec,mnb_fm],'Random Forest':[rf_acc,rf_prec,rf_rec,rf_fm],'SVM': [svm_acc,svm_prec,svm_rec,svm_fm],'KNN': [knn_acc,knn_prec,knn_rec,knn_fm] ,'My Method': [my_acc,max,my_rec,my_fm]}
    df=pd.DataFrame(data=d,columns=['StatisticMeasure','Naive Bayes','Random Forest','SVM','KNN','My Method'])
    df.to_csv(output,encoding='utf-8',index=False,sep='|')


if __name__ == "__main__":
    main()
