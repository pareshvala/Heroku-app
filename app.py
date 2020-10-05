from flask import Flask, request, render_template
import pandas as pd


import nltk 
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# def get_info(c_name):
#     xl = pd.ExcelFile('data_for_content_based - Ext.xlsx')
#     df = xl.parse('Sheet1')
    
#     df = df.drop(columns=['Year', 'Ratings', 'No. of Employees', 'Industry'])
#     rslt_df = df[df['Name'] == c_name].Overview.to_string()

#     return rslt_df
    

def give_rec(title):
    
    xl = pd.ExcelFile('data_for_content_based - Ext.xlsx')
    df = xl.parse('Sheet1')
    
    
    df = df.drop(columns=['Year', 'Ratings', 'No. of Employees', 'Industry'])
    
    
    # Remove punctuations
    def remove_punct(text):   
        text  = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        return text
    
    df['Txt_punct'] = df['Overview'].apply(lambda x: remove_punct(x))
    df.head(10)
    
    
    # Tokenization
    def tokenization(text):
        text = re.split('\W+', text)
        return text
    
    df['Txt_tokenized'] = df['Txt_punct'].apply(lambda x: tokenization(x.lower()))
    
    
    
    # Remove stopwords
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    
    def remove_stopwords(text):
        text = [word for word in text if word not in stop_words]
        return text
        
    df['Txt_nonstop'] = df['Txt_tokenized'].apply(lambda x: remove_stopwords(x))
    
    
    
    ps = nltk.PorterStemmer()
    def word_stemmer(text):
        stem_text = " ".join([i for i in text])
        return stem_text
    
    df['cleaned_overview'] = df['Txt_nonstop'].apply(lambda x: word_stemmer(x))
    
    
    cleaned_df = df
    cleaned_df['Overview'] = df['cleaned_overview']
    cleaned_df.head(1)['Overview']
  
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                ngram_range=(1, 3),
                stop_words = 'english')
    
    # Filling NaNs with empty string
    cleaned_df['Overview'] = cleaned_df['Overview'].fillna('')
     
    # Fitting the TF-IDF on the 'Overview' text
    tfv_matrix = tfv.fit_transform(cleaned_df['Overview'])
     
    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    
 
    # Reverse mapping of indices and movie titles
    indices = pd.Series(cleaned_df.index, index=cleaned_df['Name']).drop_duplicates()
    
    
    company = 'CareSuites'
    index_no = indices[company]
    
    
    list(enumerate(sig[indices[company]]))
    
    sorted(list(enumerate(sig[indices[company]])), key=lambda x: x[1], reverse=True)

    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the Companies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar Company
    sig_scores = sig_scores[1:11]

    # Company indices
    company_indices = [i[0] for i in sig_scores]

    # Top 10 most similar Companies
    # a = cleaned_df['Name'].iloc[company_indices] + ',       ' + cleaned_df['Location'].iloc[company_indices]

    a = cleaned_df['Name'].iloc[company_indices]
    arr = a.tolist()
    return arr

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    '''
    For rendering results on HTML GUI
    '''
    c_name = request.form['c_name']
    model = give_rec(c_name)
    # info = get_info(c_name)
    return render_template('recommend.html',c_name=c_name,r=model)


if __name__ == "__main__":
    app.run(debug=True)