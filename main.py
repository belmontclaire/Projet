##########################################################
# to run: streamlit run main.py
##########################################################
import plotly.express as px
import numpy as np 
import streamlit as st
import pandas as pd
import requests
import json
import pickle
import lime
from lime import lime_tabular
from PIL import Image
from urllib.request import urlopen
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Credit",
    page_icon="✅",
    layout="wide",
)

st.set_option('deprecation.showPyplotGlobalUse', False)

image = Image.open('shap.png')

@st.cache  # No need for TTL this time. It's static data :)
def get_data():
	return pd.read_csv(
            "https://raw.githubusercontent.com/belmontclaire/Test/main/DataTestSample.csv", header = 0
            )
@st.cache(allow_output_mutation=True)
def get_model():
    return pickle.load(open("finalized_model.sav", 'rb'))

df = get_data()
X_full = df.drop(["TARGET","index","SK_ID_CURR"],axis=1)
X_value  = df.drop(["TARGET","index","SK_ID_CURR"],axis=1).values
loaded_model = get_model()
explainer = lime_tabular.LimeTabularExplainer(X_value, mode='classification', 
     feature_names= X_full.columns)
predict_fn_rf = lambda x: loaded_model.predict_proba(x).astype(float)


# dashboard title
st.title("Probabilité d'avoir un crédit")

# top-level filters

client_filter = st.sidebar.selectbox("Sélectionner votre ID dans cette liste déroulante :", pd.unique(df["SK_ID_CURR"].sort_values()))

# creating a single-element container
placeholder = st.empty()

# dataframe filter

API_url = "http://127.0.0.1:5000/api/" + str(client_filter)

json_url = urlopen(API_url)

API_data = json.loads(json_url.read())
classe_predite = API_data['prediction']
if classe_predite == 1:
    y_value = 'NON'
else:
    y_value = 'OUI'
proba = API_data['proba'] 


with placeholder.container():

    # create three columns
    kpi1, kpi2 = st.columns(2)

    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Est-ce que le crédit est accepté ?",
        value=y_value,
    )
    kpi2.metric(
        label="La probabilité d'avoir un crédit :",
        value=np.round(proba,2), # TODO Donner la proba d'avoir le crédit surment grace au modèle
    )

    # feature imprtance local
    with st.spinner('Attente de la feature importance local ...'):
        idx = df.index[df["SK_ID_CURR"] == client_filter]
        explanation = explainer.explain_instance(X_value[idx[0]], predict_fn_rf, 
           num_features=10)
        st.markdown("### Feature importance locale")
        explanation.as_pyplot_figure()
        st.pyplot()
        st.markdown(explanation, unsafe_allow_html=True)
        st.write('Les variables avec des barres verte montrent que ces variables induisent que le client soit dans la classe 1 tandis que les variables avec des barres rouges induisent le contraire.')
    
    # feature importance glocal
    st.markdown("### Feature importance globale")
    st.image(image)
    st.write('Voici les 10 features qui influencent le plus la prédiction')

    # Select feature
    st.markdown("### Graphique d’analyse bi-variée entre les deux features sélectionnées")
    feature1 = st.selectbox("Sélectionner une variable dans cette liste déroulante :", pd.unique(df.drop(["TARGET"],axis=1).select_dtypes(include=[np.float64]).columns))
    feature2 = st.selectbox("Sélectionner une deuxième variable dans cette liste déroulante :", pd.unique(df.drop(["TARGET"],axis=1).select_dtypes(include=[np.float64]).columns))
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df[feature1][df.CODE_GENDER==0],
                             y=df[feature2][df.CODE_GENDER==0],
                             mode='markers',
                             marker=dict(color='rgba(0, 255, 0, 0.5)',opacity = 0.15),
                             name='Client pouvant prendre un crédit'))
    fig.add_trace(go.Scatter(x=df[feature1][df.CODE_GENDER==1],
                             y=df[feature2][df.CODE_GENDER==1],
                             mode='markers',
                             marker=dict(color='rgba(255, 0, 0, 0.5)',opacity = 0.15),
                             name='Client ne pouvant pas prendre un crédit'))
    fig.add_trace(go.Scatter(name="Valeur de l'id choisie",x=df[feature1][df["SK_ID_CURR"] == client_filter], y=df[feature2][df["SK_ID_CURR"] == client_filter],
                             mode = 'markers',
                             marker=dict(
                                        color="blue",
                                        size=15,
                                        )
                             )
                  )

    st.plotly_chart(fig)
