import sys, argparse, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# main
def main(args):
    result = predict_nb(args.name, args.train)
    print ("Prediksi jenis kelamin dengan Naive Bayes")
    jk_label = {1:"Pria", 0:"Wanita"}
    print(args.name, ' : ', jk_label[result])

# load dataset
def load_data(dataset="./data/data-pemilih-kpu.csv"):
    df = pd.read_csv(dataset, encoding = 'utf-8-sig')
    df = df.dropna(how='all')
    
    jk_map = {"Laki-Laki" : 1, "Perempuan" : 0}
    df["jenis_kelamin"] = df["jenis_kelamin"].map(jk_map)

    feature_col_names = ["nama"]
    predicted_class_names = ["jenis_kelamin"]
    X = df[feature_col_names].values     
    y = df[predicted_class_names].values 
    
    #split train:test data 70:30
    split_test_size = 0.30
    text_train, text_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, stratify=y, random_state=42) 
    
    print("Dataset Asli Pria       : {0} ({1:0.2f}%)".format(len(df.loc[df['jenis_kelamin'] == 1]), (len(df.loc[df['jenis_kelamin'] == 1])/len(df.index)) * 100.0))
    print("Dataset Asli Wanita     : {0} ({1:0.2f}%)".format(len(df.loc[df['jenis_kelamin'] == 0]), (len(df.loc[df['jenis_kelamin'] == 0])/len(df.index)) * 100.0))
    print("")
    print("Dataset Training Pria   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
    print("Dataset Training Wanita : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
    print("")
    print("Dataset Test Pria       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
    print("Dataset Test Wanita     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))

    return (text_train, text_test, y_train, y_test)

# Naive Bayes implementation
def predict_nb(name, dataset):
    if os.path.isfile("./data/pipe_nb.pkl") and dataset is None:        
        file_nb = open('./data/pipe_nb.pkl', 'rb')
        pipe_nb = pickle.load(file_nb)
    else:
        file_nb = open('./data/pipe_nb.pkl', 'wb')
        pipe_nb = Pipeline([('vect', CountVectorizer(analyzer = 'char_wb', ngram_range=(2,6))),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB())])       
        #train and dump to file                     
        dataset = load_data(dataset)
        pipe_nb = pipe_nb.fit(dataset[0].ravel(), dataset[2].ravel())
        pickle.dump(pipe_nb, file_nb)
        
        #Akurasi
        predicted = pipe_nb.predict(dataset[1].ravel())
        Akurasi = np.mean(predicted == dataset[3].ravel())*100
        print("\nAkurasi dibanding dengan data asli:", Akurasi, "%")
    
    return pipe_nb.predict([name])[0]



# args setting
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "Menentukan jenis kelamin berdasarkan nama Bahasa Indoensia")
 
  parser.add_argument(
                      "name",
                      help = "Nama",
                      metavar='nama'
                      )
  parser.add_argument(
                      "-t",
                      "--train",
                      help="Training ulang dengan dataset yang ditentukan")
  args = parser.parse_args()
  
  main(args)
