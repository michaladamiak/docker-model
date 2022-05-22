from flask import Flask
from flask import request
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/predict", methods=['GET'])
def make_prediction():
    sl = request.args.get("sl", 0, type=float)
    sw = request.args.get("sw", 0, type=float)
    pl = request.args.get("pl", 0, type=float)
    pw = request.args.get("pw", 0, type=float)
    data = np.array([sl, sw, pl, pw]).reshape(1, -1)
    
    with open("model.pkl", "rb") as fh:    
        model = pickle.load(fh)
    
    prediction = model.predict(data)[0]
    
    return f"prediction: {prediction}"

if __name__ == '__main__':
    app.run()