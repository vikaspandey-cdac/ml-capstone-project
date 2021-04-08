import pickle
import os
import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# load the nlp model and tfidf vectorizer from disk
MODEL_FILE_NAME = 'model/nlp_XGboost_model.pkl'
TRANSFORM_FILE_NAME = 'model/tranform.pkl'

product_mapping = pd.read_csv('data/Product_Details_Reviews.csv')
user_final_rating = pd.read_csv('data/user_final_rating.csv', index_col=0)
user_final_rating.columns.name = user_final_rating.index.name
user_final_rating.index.name = user_final_rating.index[0]
# remove first row of data
user_final_rating = user_final_rating.iloc[1:]


def get_suggestions():
    return list(user_final_rating.index)


def assignSentiments(reviews):
    model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
    vectorizer = pickle.load(open(TRANSFORM_FILE_NAME, 'rb'))
    review_list = np.array([reviews])
    product_vector = vectorizer.transform(review_list)
    sentiments = model.predict(product_vector)
    if sentiments:
        return 'Positive'
    else:
        return 'Negative'


def create_similarity(user_id):
    d = user_final_rating.loc[user_id].sort_values(ascending=False)[0:20]
    recommended_df = pd.merge(d, product_mapping, left_on='prod_ID', right_on='prod_ID', how='left')
    recommended_df['sentiments'] = recommended_df['reviews'].apply(assignSentiments)
    return recommended_df.loc[(recommended_df['sentiments'] == 'Positive')][0:5]
