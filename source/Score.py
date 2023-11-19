#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
  
#app = Flask(__name__)
#model = pickle.load(open('/Users/soneitaraherimalala/xgb1_model.pkl', 'rb'))

#@app.route('/predict', methods=['POST'])
#def predict_credit_score():
    #data = request.get_json(force=True)
    #print(data)
    #prediction = model.predict_proba([[np.array(data['data'])]])
    #prediction = model.predict_proba([[np.array(data['data'])]])
    #output = prediction[0]
    #return jsonify(output)

#if __name__ == "__main__":
    #app.run(debug=True)
    
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

#Chargez votre modèle XGBoost
model = pickle.load(open('/Users/soneitaraherimalala/Desktop/P7/xgb_model.pkl', 'rb'))

# Chargez votre dataframe
df_select = pd.read_csv('/Users/soneitaraherimalala/Desktop/P7/df_select.csv')


@app.route('/predict_score', methods=['POST'])
def predict_score():
    print('test1')
    try:
        # Récupérez les données d'entrée au format JSON depuis la requête
        data = request.get_json(force=True)
        print(data)
       #Assurez-vous que les données reçues correspondent aux caractéristiques attendues par le modèle
        input_data = data.get('data')
        
       
        print(input_data)
        if input_data is None:
           return jsonify({'error': 'Missing data field'})

        # Transformez les données JSON en un DataFrame
        client = df_select.loc[[input_data]]
        # Assurez-vous que les colonnes du DataFrame sont dans le même ordre que celles du modèle
        #client = df_select.loc[[input_data], columns=df_select.columns]

        print(client.shape)
       
        print(client.shape)
        # Assurez-vous que le DataFrame a les mêmes caractéristiques que celui sur lequel le modèle a été formé
        # Vous devrez peut-être effectuer un prétraitement sur les données en entrée pour les adapter au modèle si nécessaire.
        
        
        y_pred_classes = model.predict_proba(client)[:, 1]  # Obtenez la probabilité de la classe positive

       # Convertissez le tableau NumPy en une liste Python avant de le renvoyer dans la réponse JSON
        y_pred_classes_list = y_pred_classes.tolist()
        # Retournez les scores prédits sous forme de liste
        return jsonify({'prediction': y_pred_classes_list})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)


