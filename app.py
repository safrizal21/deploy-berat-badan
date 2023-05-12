from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        print(dict(request.form))
        print(dict(request.form))
        features = dict(request.form).values()
        features = np.array([float(x) for x in features])
        model = joblib.load(
            'model-development/prediksi-BB.pkl')
        print(features)
        result = model.predict([features])
        return render_template('index.html', result=result[0])
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
