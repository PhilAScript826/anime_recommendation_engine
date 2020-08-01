from flask import Flask, render_template, request, make_response
from nlp import recommender

app = Flask(__name__) 

@app.route("/", methods=['GET','POST'])  # What we type into our browser to go to different pages
def home():
    if request.method == 'POST':
        if "text" in request.form:
            text = request.form['text']
            results = recommender(text)
            return render_template('index.html', results=zip(list(range(1,6)),results))
        else:
            return render_template('index.html')
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)