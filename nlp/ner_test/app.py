from flask import Flask,request
from ner_util import *

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/ner',methods = ['GET'])
def ner_match():
    query = request.args.get('query').lower()
    result = match_pinyin(query)
    return f"<p>{result}</p>"


@app.route('/update',methods = ['POST'])
def update():
    query = request.json.get('USE_PINYIN')
    # result = match_pinyin(query)
    print(query)
    return f"<p>{query}</p>"

if __name__=="__main__":
    pass