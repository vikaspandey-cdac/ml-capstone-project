from flask import Flask, render_template, request

import traceback
import gc
import os
import tracemalloc

import psutil

from model import get_suggestions, create_similarity

app = Flask(__name__)
global_var = []
process = psutil.Process(os.getpid())
tracemalloc.start()
s = None


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('index.html', suggestions=suggestions)


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

        return render_template('recommend.html', results=results, fieldnames=fieldnames, len=len, title=userid)
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
