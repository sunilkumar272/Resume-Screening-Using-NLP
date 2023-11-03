import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# Cleaning the data using regex
def cleanResume(text):
    # removing URLs
    text = re.sub('http\S+\s*', ' ', text)
    # remoing RT and cc
    text = re.sub('RT|cc', ' ', text)
    # removing hashtags
    text = re.sub('#\S+', '', text) 
    # removing mentions- @
    text = re.sub('@\S+', '  ', text)
    # removing punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    # removing non-ASCII character
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    # removing extra whitespace
    text = re.sub('\s+', ' ', text)
    
    return text
    
    
    
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Defining a function to lemmatize the cleaned resume text
def lemmat(text):
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text
    
    
# Now encode the data
def encode_labels(df):
    label = LabelEncoder()
    df['encoded_Category'] = label.fit_transform(df['Category'])
    return df
    
    
    
# Vectorizing the cleaned columns
def vectorize_text(df):
    text = df['cleaned'].values
    target = df['encoded_Category'].values
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    word_vectorizer.fit(text)
    WordFeatures = word_vectorizer.transform(text)
    return WordFeatures, target