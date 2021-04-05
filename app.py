import numpy as np
import pandas as pd
from flask import Flask, render_template, request

import traceback
import pickle
import os

import gc
import os
import tracemalloc

import psutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# load the nlp model and tfidf vectorizer from disk
MODEL_FILE_NAME = 'model/nlp_XGboost_model.pkl'
TRANSFORM_FILE_NAME = 'model/tranform.pkl'


# product_user_rating = pd.read_csv('data/Product_User_Rating.csv')
product_mapping = pd.read_csv('data/Product_Details_Reviews.csv')
user_final_rating = pd.read_csv('data/user_final_rating.csv', index_col=0)
user_final_rating.columns.name = user_final_rating.index.name
user_final_rating.index.name = user_final_rating.index[0]
# remove first row of data
user_final_rating = user_final_rating.iloc[1:]


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


def get_suggestions():
    return list(user_final_rating.index)


app = Flask(__name__)
global_var = []
process = psutil.Process(os.getpid())
tracemalloc.start()
s = None


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)


@app.route('/gc')
def get_foo():
    before = process.memory_info().rss;
    gc.collect()  # does not help
    return {'memory': process.memory_info().rss - before}


@app.route("/recommend", methods=['GET'])
def recommend():
    try:
        userid = request.args.get('userid')
        df = create_similarity(userid)
        csv_reader = df.to_dict("records")
        results = []
        for row in csv_reader:
            results.append(dict(row))

        fieldnames = [key for key in results[0].keys()]

        return render_template('recommend.html', results=results, fieldnames=fieldnames, len=len)
    except Exception:
        traceback.print_exc()
    finally:
        gc.collect()


@app.route('/memory')
def print_memory():
    return {'memory': process.memory_info().rss}


@app.route("/snapshot")
def snap():
    global s
    if not s:
        s = tracemalloc.take_snapshot()
        return "taken snapshot\n"
    else:
        lines = []
        top_stats = tracemalloc.take_snapshot().compare_to(s, 'lineno')
        for stat in top_stats[:5]:
            lines.append(str(stat))
        return "\n".join(lines)


if __name__ == '__main__':
    print('**** Product_Recommendation App Started ****')
    app.run(debug=True)
