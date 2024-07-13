# First test about the Elmo 2 layers to get embeddings 
from sentence_transformers import SentenceTransformer
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm')

## load data 
df = pd.read_csv("./data_with_pos_dep.csv")
#df = pd.read_csv(file_path, delimiter='\t', header=None, names=['UUID', 'Tokens', 'Tags'])

# Initialize the list to store the results
data = []

# Function to get POS and dependency tags
#def get_pos_dep(token):
#    doc = nlp(token)
#    for tok in doc:
#        return tok.pos_, tok.dep_
#

## Apply the function to each token
#df[['Pos_Tag', 'Dep']] = df['tokens'].apply(lambda x: pd.Series(get_pos_dep(x)))
#df.to_csv("./data_with_pos_dep.csv", index=False)
#print(df)
# Process each row
#for index, row in df.iterrows():
#    uuid = row["id"]
#    tokens = eval(row["tokens"])  # Use eval to convert string representation of list to actual list
#    tags = eval(row["ner_tags"])
#    
#    # Create rows for each token-tag pair
#    for token, tag in zip(tokens, tags):
#        data.append([uuid, token, tag])
#
## Create a DataFrame from the processed data
#result_df = pd.DataFrame(data, columns=['id', 'tokens', 'ner_tags'])
#
## Save the DataFrame to a new CSV file
#output_file = './deconstructed_tokens.csv'
#result_df.to_csv(output_file, index=False)
#
#print(f"Deconstructed tokens and tags saved to {output_file}")

y = df['ner_tags'].values
classes = np.unique(y)
classes = classes.tolist()
new_classes = classes.copy()
new_classes.pop()
#print(new_classes)

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t, p) for w, t, p in zip(s['tokens'].values.tolist(), 
                                                           s['ner_tags'].values.tolist(), 
                                                           s['Pos_Tag'].values.tolist())]
        self.grouped = self.data.groupby('id').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None

getter = SentenceGetter(df)
sentences = getter.sentences
#print(sentences[0])

# Load pre-trained model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Function to generate embeddings for tokens in a sentence
def generate_embeddings(sentences):
    all_embeddings = []
    for sent in sentences:
        tokens = [token for token, label, pos_tag in sent]
        embeddings = model.encode(tokens)
        all_embeddings.append(embeddings)
    return all_embeddings


# Get embeddings for each sentence
embeddings = generate_embeddings(sentences)
def word2features(sent, i, embeddings):
    word = sent[i][0]
    word_embedding = embeddings[i]
    postag = sent[i][2]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }

     # Add embedding features
    for j, value in enumerate(word_embedding):
        features[f'embedding_{j}'] = value

    if i > 0:
        word1 = sent[i-1][0]
        word1_embedding = embeddings[i-1]
        postag1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })

        for j, value in enumerate(word1_embedding):
            features[f'-1:embedding_{j}'] = value

    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        word1_embedding = embeddings[i+1]
        postag1 = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
        for j, value in enumerate(word1_embedding):
            features[f'+1:embedding_{j}'] = value
    else:
        features['EOS'] = True
    return features

def sent2features(sent, embeddings):
    return [word2features(sent, i,embeddings) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label, pos_tag in sent]

def sent2tokens(sent):
    return [token for token, label, pos_tag in sent]

X = [sent2features(s,e) for s, e in zip(sentences, embeddings)]
y = [sent2labels(s) for s in sentences]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.36, random_state=9)

#print(X_train[0])
#train
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)


crf.fit(X_train, y_train)
y_pred = crf.predict(X_test)


print(metrics.flat_classification_report(y_test, y_pred, labels = new_classes))
