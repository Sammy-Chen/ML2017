import numpy as np
import string
import sys
import keras
import keras.backend as K 
from keras.models import load_model
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 


test_path = sys.argv[1]
output_path = sys.argv[2]


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def f1_score(y_true,y_pred):
    thresh = 0.3
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))

#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (_, X_test,_) = read_data(test_path,False)
    #count_vect = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("map1.pkl", "rb"))) 
    count_vect = pickle.load(open('map1.pkl', 'rb'))
    tf = pickle.load(open('map2.pkl', 'rb'))
    test_counts = count_vect.transform(X_test).toarray()
    #tf = TfidfTransformer(use_idf=False).fit(X_train_counts)
    test_counts = tf.transform(test_counts).toarray() 

    model = load_model('model_bags.h5')

    tag_list = []
    with open('tag_list') as f:
        for line in f:
            tag_list.append(line.strip())
     

    Y_pred = model.predict(test_counts)
    pred = Y_pred.argsort(axis=1)
    num = pred.shape[1] 
    thresh = 0.3
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        for index in range(pred.shape[0]):
            labels = []
            for i, j in enumerate(pred[index, num-4:num]):
                if i == 2 or i == 3:
                    labels.append(tag_list[j])
                elif Y_pred[index, j] > thresh:
                    labels.append(tag_list[j])

            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
