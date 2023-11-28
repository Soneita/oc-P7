import pandas as pd
import streamlit as st
import requests
import joblib
import shap
import numpy as np
import xgboost
from xgboost import XGBClassifier, plot_importance
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
from streamlit_shap import st_shap



# Définir l'URL de votre API Flask
api_url = 'http://127.0.0.1:5000/predict_score'
#Chargez votre modèle XGBoost
model = pickle.load(open('/home/ubuntu/oc-P7/data/xgb_model.pkl', 'rb'))
def main():
    st.title('Dashboard')
    
    # Charger le DataFrame contenant les informations des clients
    df_select = pd.read_csv('/home/ubuntu/oc-P7/df_select.csv')
    #st.subheader(df_filled)

    # Demandez à l'utilisateur de taper le numéro du client
    selected_client_id = st.text_input("Tapez le numéro du client", value="")

    # Afficher le numéro du client et la décision de crédit s'il existe

    if selected_client_id:

        selected_client_id = int(selected_client_id)
        if selected_client_id in df_select['SK_ID_CURR'].values:
            # Récupérez les informations du client en fonction du numéro du client
            client_info = df_select[df_select['SK_ID_CURR'] == selected_client_id].iloc[0]

            # Affichez les détails du client
            st.subheader("Détails du client")
            st.write(f"N° Client : {selected_client_id}")

            # Envoyez une requête à l'API pour obtenir le score du client
            api_data = {'data': selected_client_id}  # Utilisez la même clé que dans votre code Flask
            response = requests.post(api_url, json=api_data)
            # Vérifiez si la requête a réussi avant de traiter les résultats
            if response.status_code == 200:
                st.write(response.text)
    
            # Obtenez le score prédit
            prediction = response.json()

            # Accédez à la liste des prédictions sous la clé 'prediction'
            prediction_list = prediction.get('prediction', [])

            # Obtenez le premier élément de la liste, ou None s'il est vide
            predicted_score = prediction_list[0] if prediction_list else None

            # Tentez de convertir la valeur du score en float
            try:
                predicted_score = float(predicted_score)
            except (TypeError, ValueError):
                    pass  # Laissez la valeur inchangée si la conversion échoue

            # Vérifiez si la valeur du score n'est pas None et est de type numérique
            if predicted_score is not None and isinstance(predicted_score, (int, float)):
                #Classez le client en fonction du score
                if predicted_score < 0.8:
                    st.write("Classe : 0 (Crédit accordé)")
                else:
                    st.write("Classe : 1 (Crédit refusé)")
                    
            if predicted_score is not None and isinstance(predicted_score, (int, float)):
                if predicted_score < 0.8:
                    df_select['Classe'] = 0
                else:
                    df_select['Classe'] = 1
                #st.write(f"Score : {predicted_score}")
                #st.write("Classe :", df_select['Classe'].values[0])
                
            st.write(f"Score : {predicted_score}")
         
           # Seuil
            seuil = 0.8

            # Couleur de la barre de progression (vert si le score est inférieur au seuil, rouge sinon)
            if predicted_score < seuil:
                couleur = 'green'
            else:
                couleur = 'red'

    # Créez la barre de progression
            st.subheader("Barre de progression du score")
            st.progress(predicted_score)

        # Ajoutez une indication du seuil
            st.write(f"Seuil : {seuil}")
            # Calcul de l'importance locale avec SHAP
            st.subheader("Importance locale des features")
            #shap.initjs() 
            X_numpy = df_select.to_numpy()
            explainer = shap.Explainer(model, X_numpy)
            numpy_index = df_select[df_select['SK_ID_CURR'] == selected_client_id].index.values[0]

# Calculer les valeurs SHAP pour l'index spécifié dans X_numpy
            shap_values = explainer.shap_values(X_numpy[numpy_index].reshape(1, -1))
            shap.initjs() 
# Créer le graphique SHAP pour la feature importance locale
            shap.force_plot(explainer.expected_value, shap_values, df_select.iloc[numpy_index], feature_names=df_select.columns)
            # Calculez les valeurs SHAP pour le client sélectionné
            #shap_values = explainer.shap_values(X_numpy[selected_client_id].reshape(1, -1))
            st_shap( shap.force_plot(explainer.expected_value, shap_values, df_select.iloc[numpy_index], feature_names=df_select.columns))
            # Calculez les valeurs SHAP pour le client sélectionné
            #shap_values = explainer.shap_values(X_numpy[selected_client_id].reshape(1, -1))

            #shap.initjs()  # Assurez-vous d'appeler cette fonction pour activer JavaScript pour les graphiques SHAP
            
         
            
            #Feature Importance globale
            st.subheader("Importance globale des features")
            explainer = shap.Explainer(model, df_select)
            shap_values = explainer(df_select)

            #st_shap(shap.plots.waterfall(shap_values[0]), height=300)
            st_shap(shap.plots.beeswarm(shap_values), height=300)
            
            
            st.title("Visualisation des fonctionnalités")

            # Sélectionnez la fonctionnalité dans une liste déroulante
            feature1 = st.selectbox("Sélectionnez une fonctionnalité :", df_select.columns[:-1], key="feature1")
           
            # Affichez la distribution de la fonctionnalité selon les classes (barplot)
            st.subheader(f"Distribution de {feature1} selon les classes")
           
            # Créez une figure et un axe
            fig, ax = plt.subplots(figsize=(8, 6))

            # Utilisez sns.histplot pour créer le graphique de distribution
            sns.histplot(data=df_select, x=feature1, hue=df_select['Classe'], multiple="stack", kde=True, ax=ax)

            # Affichez le graphique dans Streamlit en utilisant st.pyplot(fig)
            st.pyplot(fig)


            # Affichez le positionnement de la valeur du client sur la fonctionnalité sélectionnée (scatterplot)
            st.subheader(f"Positionnement de la valeur du client sur {feature1}")
            # Ajoutez ici le code pour afficher le scatterplot

    # Affichez les informations sur la valeur sélectionnée
            # Affichez les informations sur la valeur sélectionnée
            selected_value1 = st.slider(
                f"Sélectionnez la valeur de {feature1}",
                float(df_select[feature1].min()),
                float(df_select[feature1].max()),
                float(df_select[feature1].mean()), key="feature1_slider"
                )
            st.write(f"Vous avez sélectionné la valeur {selected_value1} pour {feature1}")
            # Affichez la distribution de la première fonctionnalité selon les classes (barplot)
            
            # Sélectionnez la deuxième fonctionnalité dans une liste déroulante
            feature2 = st.selectbox("Sélectionnez une fonctionnalité :", df_select.columns[:-1], key="feature2")
            st.subheader(f"Distribution de {feature2} selon les classes")

            
            # Utilisez plt.figure() pour créer la figure au lieu de plt.subplots()
            fig2 = plt.figure(figsize=(8, 6))

            # Utilisez sns.histplot pour créer le graphique de distribution
            sns.histplot(data=df_select, x=feature2, hue='Classe', multiple="stack", kde=True)

            # Affichez le graphique dans Streamlit
            st.pyplot(fig2)
            
            st.subheader(f"Positionnement de la valeur du client sur {feature2}")
            selected_value2 = st.slider(
                f"Sélectionnez la valeur de {feature2}",
                float(df_select[feature2].min()),
                float(df_select[feature2].max()),
                float(df_select[feature2].mean()), key="feature2_slider"
                )
                    
            st.write(f"Vous avez sélectionné la valeur {selected_value2} pour {feature2}")
            # Affichez un graphique de dispersion (scatter plot) entre feature1 et feature2
            st.subheader(f"Analyse bivariée entre {feature1} et {feature2}")
            #df_select['score'] = prediction_list # R
            df_select['scores'] = model.predict(df_select[df_select.columns[:-1]])
            # Utilisez plt.figure() pour créer la figure
            fig3 = plt.figure(figsize=(8, 6))

            # Utilisez sns.scatterplot pour créer le graphique de dispersion avec une palette de couleurs
            sns.scatterplot(data=df_select, x=feature1, y=feature2, palette='coolwarm', legend='full')

            # Affichez le graphique dans Streamlit
            st.pyplot(fig3)
           
            
            
# Mettez à jour le scatterplot avec la ligne correspondant à la valeur sélectionnée
# Ajoutez ici le code pour mettre à jour le scatterplot en fonction de la valeur sélectionnée
        else:
            st.write("Le numéro du client n'existe pas dans le DataFrame.")
     
if __name__ == '__main__':
    main()
#Assurez-vous que api_url contient l'URL correcte de votre API, qui devrait être celle que vous avez configurée pour l'endpoint /predict_credit_score. Lorsque l'utilisateur entre le numéro du client, vous envoyez une requête à l'API, récupérez la prédiction et l'affichez dans votre dashboard Streamlit.

#N'oubliez pas de personnaliser davantage votre tableau de bord avec d'autres éléments et fonctionnalités selon vos besoins.








  


