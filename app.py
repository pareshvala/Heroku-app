from flask import Flask, request, render_template
import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# def get_info(c_name):
#     xl = pd.ExcelFile('data_for_content_based - Ext.xlsx')
#     df = xl.parse('Sheet1')
    
#     rslt_df = df[df['Name'] == c_name].Overview.to_string()

#     return rslt_df
    

def give_rec(title):
    
    cleaned_df = pd.read_csv('cleaneddf.csv', index_col=0)
  
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