##########################################################
# to run: FLASK_APP=server.py flask run
##########################################################
import json
from flask import Flask, request, jsonify  
import pickle
import pandas as pd
import numpy as np
import uvicorn
#from waitress import serve
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://ohnuukntxotyni:cad8f6f371db393af27e81e31a934f6ec58d55d18e45b61da86a1556a11e8078@ec2-63-32-248-14.eu-west-1.compute.amazonaws.com:5432/dccm5mjjtl5nsq'

loaded_model = pickle.load(open("finalized_model.sav", 'rb'))

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
    #uvicorn.run(app, host='10.0.2.2', port=8080)
    #app.run(host='52.49.176.128', port=24737, debug=True)
    #serve(app, host="10.0.2.2", port=8080)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    #app.run()
