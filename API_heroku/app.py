from flask import Flask, request, jsonify, send_file
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
import pandas as pd

df_proba = pd.read_csv('df_stream_proba.csv',index_col="SK_ID_CURR",sep=",")
pipeline= joblib.load('pipeline.pkl')

app = Flask(__name__)


@app.route('/data', methods=['GET'])
def get_data():
    param = request.args.get("proba")
    param = int(param)
    p=df_proba.loc[param].to_dict()
    return jsonify(p)


# Définir l'endpoint POST
@app.route('/prediction', methods=['POST'])
def prediction():
    # Récupérer le dictionnaire JSON à partir de la requête
    data = request.get_json()
    df = pd.DataFrame(data)
    proba = pipeline.predict_proba(df)[0][0]
    proba = float(proba)
    # Renvoyer la réponse sous forme de JSON
    response = {'probability': proba}
    return jsonify(response)

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)