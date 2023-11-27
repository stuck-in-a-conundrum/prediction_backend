from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
rectum_model = pickle.load(open('rectum_model.pkl','rb'))
bladder_model = pickle.load(open('bladder_model.pkl','rb'))

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

@app.route('/predict_rectum', methods = ['POST'])
@cross_origin()
def predict_rectum():
    try:
        data = request.get_json()
        #print(type(request))
        #ref=trans_text(data['text'])
        ref = np.array(data['values']).reshape(1, -1)
        rectum_prediction = rectum_model.predict(ref)
        #print(prediction)
        output = {
            'prediction': rectum_prediction.tolist()[0],
            }
        return jsonify( output )
    except NameError:
        #print(NameError)
        return 'something went wrong'
    
@app.route('/predict_bladder', methods = ['POST'])
@cross_origin()
def predict_bladder():
    try:
        data = request.get_json()
        #print(type(request))
        ref = np.array(data['values']).reshape(1, -1)
        bladder_prediction = bladder_model.predict(ref)
        #print(prediction)
        output = {
            'prediction': bladder_prediction.tolist()[0]
            }
        return jsonify( output )
    except NameError:
        #print(NameError)
        return 'something went wrong'

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error='The requested URL was not found on the server'), 404

if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5000, debug = True)


    #{"values" : [0.057769,0.036991,86.3,81.1,100.4,776.1,0.0,100.0]}
