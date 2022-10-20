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

loaded_model = pickle.load(open("/Users/belmontclaire/Documents/Projet7/finalized_model.sav", 'rb'))

df = pd.read_csv("/Users/belmontclaire/Documents/Projet7/DataTestSample.csv", header = 0)

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
    app.run(debug=True)

