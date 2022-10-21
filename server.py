##########################################################
# to run: FLASK_APP=server.py flask run
##########################################################
import json
from flask import Flask, request, jsonify  
import shap
import pickle
import pandas as pd
import numpy as np
#from Model import ClientModel, Client
app = Flask(__name__)

loaded_model = pickle.load(open("https://github.com/belmontclaire/Test/blob/main/finalized_model.sav?raw=true", 'rb'))

df = pd.read_csv("https://raw.githubusercontent.com/belmontclaire/Test/main/DataTestSample.csv", header = 0)

@app.route('/api/<int:id_client>')
def credit(id_client):
        data = df[df["SK_ID_CURR"] == id_client]
        X = data.drop(["TARGET","index","SK_ID_CURR"],axis=1)
        prediction = loaded_model.predict(X.values)
        probability = loaded_model.predict_proba(X.values)[:,0]
        dict_final = {
                'prediction' : int(prediction),
                'proba' : float(probability)
        }
        return jsonify(dict_final)


#lancement de l'application
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5000)
    app.run(debug=True)

