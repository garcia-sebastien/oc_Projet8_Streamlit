import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import shap

# Configurer la page
st.set_page_config(page_title="Prêt à dépenser - Dashboard d'Octroi de Crédit")

# Ajouter des styles CSS personnalisés pour respecter les critères d'accessibilité du WCAG
st.markdown("""
<style>
/* Style pour les légendes personnalisées */
.custom-caption {
    font-size: 0.9em; /* Taille de la police */
    color: #000000; /* Couleur du texte en noir */
    background-color: #f0f0f0; /* Couleur de fond (optionnelle) pour améliorer la lisibilité */
    padding: 4px; /* Espacement autour du texte */
    border-radius: 4px; /* Coins arrondis */
    border: 1px solid #dcdcdc; /* Bordure autour de la légende */
}

/* Style pour le tableau */
.st-table {
    width: 100%; /* Largeur du tableau */
    border-collapse: collapse; /* Fusionner les bordures */
    font-size: 1em; /* Taille de la police */
}

.st-table th, .st-table td {
    border: 1px solid #dcdcdc; /* Bordure des cellules */
    padding: 8px; /* Espacement interne des cellules */
    text-align: left; /* Alignement du texte */
}

.st-table th {
    background-color: #f4f4f4; /* Couleur de fond des en-têtes */
    color: #000000; /* Couleur du texte des en-têtes */
    font-weight: bold; /* Met le texte en gras */
}

.st-table tr:nth-child(even) {
    background-color: #f9f9f9; /* Couleur de fond pour les lignes paires */
}

.st-table tr:nth-child(odd) {
    background-color: #ffffff; /* Couleur de fond pour les lignes impaires */
}

/* Style pour le corps du texte */
body {
    font-size: 1.1em; /* Taille de la police pour le corps du texte */
    color: #000000; /* Couleur du texte en noir pour améliorer le contraste */
}
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("Prêt à dépenser - Dashboard d'Octroi de Crédit")

# Introduction
st.markdown("""
Ce dashboard permet aux chargés de relation client d'expliquer de façon transparente les décisions d’octroi de crédit.
""")
st.markdown("""---""")

# Sidebar navigation
section = st.sidebar.selectbox("Aller à", ["Prédiction du modèle", 
                                           "Interprétation de la prédiction",
                                           "Informations descriptives du client",
                                           "Analyse bi-variée"])
st.sidebar.markdown("""---""")
st.sidebar.title("OpenClassrooms - Projet 8")
st.sidebar.markdown("Créé par Sébastien Garcia")

# Importer les données brutes
df = pd.read_csv('data.csv')

# Importer les données prétraitées
X = pd.read_csv('data_api.csv')
X = X.drop(columns='SK_ID_CURR')

# Importer le modèle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Saisie de l'identifiant du client
client_id = st.text_input("Spécifiez l'ID du client (de 100002 à 100230)", max_chars=6)

if len(client_id) == 6:
    st.markdown("""---""")
    client_data = df.loc[df['SK_ID_CURR'] == int(client_id)]
    client_data_display = client_data.drop(columns=['SK_ID_CURR', 'TARGET'])
    
    with st.spinner(text="Récupération de la prédiction du modèle..."):

        # URL de l'API
        URL = "https://sg-oc-projet7-api-5589f3e47be6.herokuapp.com/predict"

        # Effectuer la requête POST vers l'API
        response = requests.post(URL, data={'client_id': int(client_id)})

        if response.status_code == 200:
            response = response.json()

            if 'prediction' in response:
                if section == "Prédiction du modèle":
                    st.markdown(f"## Prédiction du modèle pour le client {client_id}")
                    st.markdown("### Statut du crédit")
                    # Afficher les résultats de la classification
                    prediction = response['prediction'][0]
                    proba = response['proba']
                    proba = [1-proba[0], proba[0]]

                    if prediction == 1:
                        st.error("Crédit refusé")
                    else:
                        st.success("Crédit accordé")

                    # Afficher les probabilités de faillite du client
                    st.markdown("### Probabilité de faillite")
                    fig, ax = plt.subplots()
                    ax.pie(proba,
                           labels=['Non-Faillite', 'Faillite'],
                           colors=['#008bfb', '#ff0051'],
                           startangle=90,
                           counterclock=False,
                           wedgeprops=dict(width=0.3))
                    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                    fig.gca().add_artist(centre_circle)
                    ax.text(0, 0, f"{proba[1]*100:.2f}%", ha='center', va='center', size='xx-large')
                    ax.axis('equal')
                    st.pyplot(fig)
                    st.markdown('<p class="custom-caption">Diagramme circulaire montrant la probabilité de faillite du client</p>', unsafe_allow_html=True)

                elif section == "Interprétation de la prédiction":
                    st.markdown("## Interprétation de la prédiction")
                    # Utilisation de SHAP pour l'explicabilité
                    explainer = shap.Explainer(model, X)
                    shap_values = explainer(X)
                    client_index = df[df['SK_ID_CURR'] == int(client_id)].index[0]
                        
                    st.markdown(f"### Importance des caractéristiques du client {client_id}")
                    shap.initjs()
                    shap.plots.waterfall(shap_values[client_index], max_display=10, show=False)
                    st.pyplot(bbox_inches='tight')
                    st.markdown('<p class="custom-caption">Graphique en cascade montrant l\'importance des caractéristiques pour la prédiction du client</p>', unsafe_allow_html=True)

                    st.markdown("### Importance globale des caractéristiques")
                    shap.plots.beeswarm(shap_values, show=False)
                    st.pyplot(bbox_inches='tight')
                    st.markdown('<p class="custom-caption">Graphique en essaim montrant l\'importance globale des caractéristiques</p>', unsafe_allow_html=True)

                elif section == "Informations descriptives du client":
                    st.markdown(f"## Informations descriptives du client {client_id}")
                    # Afficher les caractéristiques du client
                    st.markdown("### Informations du client")
                    
                    # Convertir le DataFrame en HTML avec les styles
                    client_data_transposed = client_data_display.T
                    client_data_transposed.columns = ['Valeur']
                    
                    # Créer le tableau HTML avec les classes CSS appliquées
                    html_table = client_data_transposed.to_html(classes='st-table', border=0, index_names=True)
                    st.markdown(html_table, unsafe_allow_html=True)
                    
                    st.markdown("### Analyse comparative")
                    features = st.multiselect("Sélectionnez des caractéristiques à comparer avec l'ensemble des clients", client_data_display.columns)
                    if features:
                        for feature in features:
                            fig, ax = plt.subplots()
                            sns.histplot(df[feature], kde=True, ax=ax, label='Tous les clients')
                            ax.axvline(client_data_display[feature].values[0], color='red', linestyle='--', label=f'Client {client_id}')
                            ax.set_title(f'Comparaison de {feature}')
                            ax.legend()
                            st.pyplot(fig)
                            st.markdown(f'<p class="custom-caption">Histogramme comparatif pour la caractéristique {feature}</p>', unsafe_allow_html=True)

                elif section == "Analyse bi-variée":
                    st.markdown("## Analyse bi-variée entre deux caractéristiques sélectionnées")
                    feature_x = st.selectbox("Sélectionnez la première caractéristique", client_data_display.columns)
                    feature_y = st.selectbox("Sélectionnez la deuxième caractéristique", client_data_display.columns)
                    if feature_x and feature_y:
                        fig, ax = plt.subplots()
                        sns.scatterplot(data=df, x=feature_x, y=feature_y, ax=ax, alpha=0.5, label='Tous les clients')
                        ax.scatter(client_data_display[feature_x].values[0], client_data_display[feature_y].values[0], color='red', label=f'Client {client_id}')
                        ax.set_title(f'Analyse bi-variée : {feature_x} vs {feature_y}')
                        ax.legend()
                        st.pyplot(fig)
                        st.markdown(f'<p class="custom-caption">Diagramme de dispersion pour les caractéristiques {feature_x} et {feature_y}</p>', unsafe_allow_html=True)

            elif 'erreur' in response:
                st.write(response['erreur'])
        else:
            st.write("Erreur lors de la requête à l'API")
