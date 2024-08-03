from flask import Flask, render_template, request, send_file
from generate_score import generate_score

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        style = request.form['style']
        tempo = request.form['tempo']
        key = request.form['key']

        generate_score(style, tempo, key)
        return send_file('generated_score.png', mimetype='image/png')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
