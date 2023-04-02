import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ps = PorterStemmer()

# Pipeline Building

class MlPipeline():
    def __init__(self, message) -> None:
        self.message = message
        self.transform_message = ''

    def transform_text(self):
        global ps        
        self.message = self.message.lower()
        self.message = nltk.word_tokenize(self.message)
        x = []
        for word in self.message:
            if word.isalnum():
                x.append(word)
        
        y = []
        for _word in x:
            if _word not in stopwords.words('english') and _word not in string.punctuation:
                y.append(_word)
                
        tokenized_text = []
        for term in y:
            tokenized_text.append(ps.stem(term))       
                
        self.transform_message = " ".join(tokenized_text)
    
    def ml_pipeline_predict(self, choices):
        # Load the ML models
        mnb_model = pickle.load(open('.\ML Models\MultinomialNaiveByes_model.pkl', 'rb'))
        svm_model = pickle.load(open('.\ML Models\SupportVectorMachine_model.pkl', 'rb'))
        # Load the Vectorizers
        cv_vectorizer = pickle.load(open('.\Vectorizers\cv_vectorizer.pkl', 'rb'))  
        
        tm_list = [self.transform_message]
        vectorized_transformed_text = cv_vectorizer.transform(tm_list).toarray()

        if choices[0] == 'SVM Model':
            return svm_model.predict(vectorized_transformed_text)
        else:
            return mnb_model.predict(vectorized_transformed_text)    