import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from flask import Response

import pickle
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# load the nlp model and tfidf vectorizer from disk
filename = 'model/nlp_XGboost_model.pkl'
model = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('model/tranform.pkl', 'rb'))

product_user_rating = pd.read_csv('data/Product_User_Rating.csv')
product_mapping = pd.read_csv('data/Product_Details_Reviews.csv')

train, test = train_test_split(product_user_rating, test_size=0.30, random_state=42)
df_pivot = train.pivot_table(index='userId', columns='prod_ID', values='rating').fillna(0)

# Copy the train dataset into dummy_train
dummy_train = train.copy()

# The movies not rated by user is marked as 1 for prediction.
dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x >= 1 else 1)

# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(index='userId', columns='prod_ID', values='rating').fillna(1)

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
user_correlation[user_correlation < 0] = 0
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_final_rating = np.multiply(user_predicted_ratings, dummy_train)


def assignSentiments(reviews):
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


def get_suggestions():
    return list(product_user_rating['userId'].str.capitalize())


app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)


@app.route("/recommend", methods=['GET'])
def recommend():
    userid = request.args.get('userid')
    df = create_similarity(userid)
    csv_reader = df.to_dict("records")
    results =[]
    for row in csv_reader:
        results.append(dict(row))

    fieldnames = [key for key in results[0].keys()]

    return render_template('recommend.html', results=results, fieldnames=fieldnames, len=len)


if __name__ == '__main__':
    print('**** Product_Recommendation App Started ****')
    app.run(debug=True)
