##########################################################
# to run: streamlit run main.py
##########################################################
import plotly.express as px
import numpy as np 
import streamlit as st
import pandas as pd
import requests
import pickle
import lime
from lime import lime_tabular
import shap

st.set_page_config(
    page_title="Credit",
    page_icon="✅",
    layout="wide",
)

st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv(
    "/Users/belmontclaire/Documents/TestProjet7/DataTest.csv", header = 0
)
X_full = df.drop(["TARGET","index","SK_ID_CURR"],axis=1)
X_value  = df.drop(["TARGET","index","SK_ID_CURR"],axis=1).values
filename ="/Users/belmontclaire/Documents/TestProjet7/finalized_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(X_full)

# dashboard title
st.title("Probabilité d'avoir un crédit")

# top-level filters
client_filter = st.selectbox("Sélectionner votre ID dans cette liste déroulante :", pd.unique(df["SK_ID_CURR"]))

# creating a single-element container
placeholder = st.empty()

# dataframe filter
data = df[df["SK_ID_CURR"] == client_filter]

# fonction display shap
#def st_shap(plot, height=None):
#    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#    components.html(shap_html, height=height)

# labels
#labels = requests.get("http://localhost:5000/api/labels").json()
#selector = st.selectbox("Sélectionner votre ID dans cette liste déroulante :", labels)

# load data
#data = pd.read_json(
#    requests.get("http://localhost:5000/api/data").json()
#)
X = data.drop(["TARGET","index","SK_ID_CURR"],axis=1)
y = loaded_model.predict(X)
if y == 0:
    y_value = "OUI"
else:
    y_value = "NON"
proba = loaded_model.predict_proba(X)[:,0]

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
    #idx = df.index[df["SK_ID_CURR"] == client_filter]
    #explainer = lime_tabular.LimeTabularExplainer(X_value, mode='classification', 
    #   feature_names= X_full.columns)
    #predict_fn_rf = lambda x: loaded_model.predict_proba(x).astype(float)
    #explanation = explainer.explain_instance(X_value[idx[0]], predict_fn_rf, 
    #   num_features=10)
    #st.markdown("### Feature importance locale")
    #res = explanation.show_in_notebook(show_all=False)
    #st.markdown(res, unsafe_allow_html=True)
    #st.pyplot(explanation.show_in_notebook(show_all=False))
    
    # feature importance glocal
    st.markdown("### Feature importance globale")
    st.pyplot(shap.summary_plot(shap_values, X_full, max_display = 10))
    
    # Select feature
    feature_filter = st.selectbox("Sélectionner une variable dans cette liste déroulante :", pd.unique(df.columns))
    st.markdown("### Histogramme de la variable choisi")
    fig = px.histogram(df, x=feature_filter)
    st.plotly_chart(fig)

