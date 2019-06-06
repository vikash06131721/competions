
import libraries
import imp
import pickle
imp.reload(libraries)
from libraries import *
class utils(object):
    def __init__(self,dataframe):
        self.dataframe = dataframe
        
    def print_df_info(self,target_column):
        dataframe = self.dataframe
        print ("Detailed DF Info:")
        df = pd.DataFrame(100.0*dataframe.isnull().sum()/len(dataframe),columns=['percentage_nulls_in_each_column'])
        categorical_cols =[]
        unique_vals =[]
        for col in dataframe.columns:
            unique_vals.append(dataframe[col].nunique())

            if dataframe[col].nunique()<10:
                categorical_cols.append(True)
            else:
                categorical_cols.append(False)
        df['categorical_vals'] = categorical_cols
        df['num_unique_vals'] = unique_vals
        df['datatypes'] = dataframe.dtypes.values
        try:
            plt.figure(figsize=(20,10))
            (100.0*dataframe[target_column].value_counts(normalize=True)).plot(kind='bar',
                                                                           title="Percentage Wise Distribution Of Target Variable")
            plt.xlabel('Detected Classes')
            plt.ylabel('Percentage')
        except KeyError:
            pass
        return df
    
    def clean_text(self,x):
        x= re.sub('[^A-Za-z0-9]+', ' ', x)
        x=x.lower()
        x = x.strip(' ')
        return x
    
    
    def map_to_target(self,column,data):
        map_target ={}
        keys = data[column].value_counts().keys()
        for i in range(len(keys)):
            map_target[keys[i]]=i
        data['target_map'] = data[column].map(lambda x:map_target[x])
        return data,map_target
    
    
    def corpus_inspection_data_understanding(self,all_text_data,data_cleaned,target_col):
        sentences = all_text_data
        texts = sentences
        print ("Building features on 40k popular words")
        vectorizer = TfidfVectorizer(stop_words="english",max_features=40000)
        X = vectorizer.fit_transform(texts)
        X=X.toarray()

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X[:5000])
        principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])


        print ("Performing Dimensionality Reduction")
        finalDf = pd.concat([principalDf, data_cleaned[[target_col]][:5000]], axis = 1)
        fig = plt.figure(figsize = (12,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = data_cleaned[target_col].unique()
        colors = ['r', 'g', 'b','c','m','y','k','w']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf[target_col] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c = color
                       , s = 50)
        ax.legend(targets)
        ax.grid()
        return X
    def return_max(self,x):
        return np.argmax(x)
        
    def fitting_white_box_models(self,clf,X,y,stratifiedKfold_num,model_name):
        skf = StratifiedKFold(n_splits=stratifiedKfold_num, random_state=None)

        f1 =[]
        acc =[]

        for train_index, test_index in skf.split(X,y):  
            X_train, X_test = X[train_index], X[test_index] 
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train,y_train)
            prob_pred = list(map(self.return_max,clf.predict_proba(X_test)))
            f1.append(f1_score(y_test,prob_pred,average='macro'))
            acc.append(accuracy_score(y_test,prob_pred))

#             print ("F1 Score, Acc Score, Recall Score,Precision Score:",f1_score(y_test,prob_pred,average='macro'),accuracy_score(y_test,prob_pred))

        print ("Average F1,Accuracy:",np.mean(f1),np.mean(acc))
        plt.figure(figsize=(20,10))
        plt.plot(f1,color='black', marker='o', linestyle='dashed',label='F1')
        plt.plot(acc,color='blue', marker='*', linestyle='dashed',label='Acc')

        plt.title("Various Accuracy Measures")
        plt.legend(loc='best')
        with open(model_name+'.pickle', 'wb') as handle:
            pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(model_name+'.pickle', 'rb') as handle:
            b = pickle.load(handle)
        return b
        
        
    def corpus_details(self,list_of_sents):
        list_of_words = []
        corpus ={}
        for s in list_of_sents:
            for w in s.split(' '):
                list_of_words.append(w)
                if w not in corpus.keys():

                    corpus[w]=len(corpus)
        list_of_words_ = [i for i in list_of_words if i]
        frequency_words = nltk.FreqDist(list_of_words_)
        perc_95 = int(round(len(corpus)*0.95))
        most_comm = frequency_words.most_common(perc_95)
        df= pd.DataFrame(most_comm,columns=['words','freq']).set_index('words')
        
        df[:100].plot(kind='bar',figsize=(20,10))
        
        return frequency_words.most_common(perc_95),len(corpus)
    
    def one_hot_encoding(self,data,target_column):
        data = data[target_column].tolist()
        encoded = to_categorical(data)
        print (len(encoded))
        return encoded
    
    
    def create_embedding_matrix(self,MAX_NB_WORDS,tokenizer):
        EMBEDDING_FILE = '/Users/vprasad/Desktop/haptik/code/glove.840B.300d.txt'
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        word_index = tokenizer.word_index
        nb_words = min(MAX_NB_WORDS, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in word_index.items():
            if i >= MAX_NB_WORDS: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        return embedding_matrix,embed_size,embeddings_index
    
    
    def f1(self, y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    def create_model(self,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,embedding_matrix,embed_size,activation,output_class):
        inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
        x = Embedding(MAX_NB_WORDS, embed_size, weights=[embedding_matrix])(inp)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)

        x = GlobalMaxPool1D()(x)
        x = Dense(256, activation=activation)(x)

        x = Dropout(0.8)(x)
        x = Dense(output_class, activation="softmax")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print (model.summary())
        return model
    
    def normalize_document(self,doc):
        wpt = nltk.WordPunctTokenizer()
        stop_words = nltk.corpus.stopwords.words('english')

        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
        doc = doc.lower()
        doc = doc.strip()
        # tokenize document
        tokens = wpt.tokenize(doc)
        # filter stopwords out of document
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # re-create document from filtered tokens
        doc = ' '.join(filtered_tokens)
        return doc
    def normal_features_deriver(self,all_text_data,numerical_vals,dataframe):
        all_text_data = [self.normalize_document(corp) for corp in all_text_data]

        cv = CountVectorizer(min_df=0., max_df=1.)
        cv_matrix = cv.fit_transform(all_text_data)
        cv_matrix = cv_matrix.toarray()




        lda = LatentDirichletAllocation(n_topics=8, max_iter=10, random_state=0)
        dt_matrix = lda.fit_transform(cv_matrix)
        features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3','T4','T5','T6','T7','T8'])


        scaler = StandardScaler()

        length_text =[len(i) for i in all_text_data]
        numerical_vals['length_text']=length_text
        
        all_feats = features.join(numerical_vals)
        try:
            all_feats['target'] = dataframe['target_map'].values
            return all_feats
        except KeyError:
            return all_feats

    