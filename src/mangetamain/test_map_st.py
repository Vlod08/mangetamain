import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from pathlib import Path  # Indispensable pour trouver le fichier

# --- Configuration de la page ---
st.set_page_config(layout="wide")
st.title("Carte Cliquable 🌍")

# --- Définir le chemin absolu vers le GeoJSON ---
# (C'est la méthode la plus robuste pour que Streamlit trouve ton fichier)
try:
    SCRIPT_DIR = Path(__file__).parent
    JSON_FILEPATH = SCRIPT_DIR / "my_countries.json" # Assure-toi que c'est le bon nom
except NameError: # Si __file__ n'est pas défini (ex: dans certains environnements)
    st.warning("Impossible de définir le chemin absolu via __file__, recherche de 'my_countries.json' dans le dossier courant.")
    JSON_FILEPATH = Path("my_countries.json")


# -----------------------------------------------------------------
# ÉTAPE 1 : CHARGEMENT DES DONNÉES (MIS EN CACHE)
# -----------------------------------------------------------------

@st.cache_data
def load_geojson(filepath: Path | str) -> dict | None:
    """
    Charge ton fichier GeoJSON local.
    """
    st.write(f"Tentative de chargement du GeoJSON depuis : {filepath}") # Message de débogage
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            st.success(f"Fichier GeoJSON chargé avec succès ({len(data.get('features', []))} pays).")
            return data
    except FileNotFoundError:
        st.error(f"ERREUR : Le fichier est introuvable au chemin suivant :")
        st.error(f"{filepath.resolve()}") # Affiche le chemin complet tenté
        st.info("Assure-toi que ton fichier GeoJSON ('my_countries.json') est dans le même dossier que ce script.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"ERREUR : Le fichier '{filepath}' n'est pas un JSON valide.")
        st.error(f"Détails de l'erreur JSON : {e}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du fichier JSON : {e}")
        return None

# --- Chargement des données ---
geo_data = load_geojson(filepath=JSON_FILEPATH)

# -----------------------------------------------------------------
# ÉTAPE 2 : AFFICHAGE (CARTE ET RÉSULTATS)
# -----------------------------------------------------------------

if geo_data:
    # Sépare l'écran en 2 colonnes ("cellules")
    col1, col2 = st.columns([2, 1])

    # --- Colonne 1 : La Carte ---
    with col1:
        st.subheader("Cliquez sur un pays")

        m = folium.Map(location=[30, 0], zoom_start=2)

        # Vérifie que les features existent et que la clé 'name' est présente
        try:
            folium.GeoJson(
                geo_data,
                name="Pays",
                # Ajoute un tooltip pour voir le nom du pays au SURVOL
                # Confirme que la clé est bien "name"
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["name"],
                    aliases=["Pays :"]
                ),
                # Style pour mieux voir les pays
                style_function=lambda feature: {
                    'fillColor': '#lightblue',
                    'color': 'blue',
                    'weight': 1,
                    'fillOpacity': 0.6,
                },
                highlight_function=lambda x: {'weight':3, 'color':'yellow'},
            ).add_to(m)
        except KeyError:
            st.error("Erreur dans le GeoJSON : La clé 'name' est introuvable dans les propriétés ('properties'). Vérifiez votre fichier.")
            st.stop() # Arrête l'exécution si la clé manque
        except Exception as e:
             st.error(f"Erreur lors de l'ajout du GeoJSON à la carte Folium : {e}")
             st.stop()

        # Rendu de la carte et capture des événements (le clic)
        st.write("Affichage de la carte...")
        map_output = st_folium(m, width=700, height=500)
        st.write("--- Fin de la carte ---") # Pour voir si le rendu se termine

   # --- Colonne 2 : Les Résultats ---
    with col2:
        st.subheader("Pays Sélectionné")

        # --- Débogage (tu peux le laisser ou l'enlever) ---
        #st.write("Infos retournées par st_folium:")
       #st.json(map_output)
        # ----------------------------------------------------

        selected_country_name = None

        # --- MODIFICATION ICI ---
        # On vérifie si le tooltip a été capturé lors du clic
        if map_output and map_output.get("last_object_clicked_tooltip"):
            # Récupère le texte complet, ex: "Pays : \n\n France"
            tooltip_text = map_output["last_object_clicked_tooltip"]
            
            # Essaye d'extraire le nom après "Pays : " et les sauts de ligne
            try:
                # Sépare le texte au niveau de ":", prend la 2e partie, enlève les espaces/sauts de ligne
                selected_country_name = tooltip_text.split(":")[-1].strip() 
            except Exception as e:
                st.warning(f"Impossible d'extraire le nom du pays depuis le tooltip : '{tooltip_text}'. Erreur: {e}")

        # Affiche le résultat (le reste est identique)
        if selected_country_name:
            # Affiche le message que tu as demandé
            st.success(f"C'est le pays : {selected_country_name} !")
        else:
            st.info("Cliquez sur un pays sur la carte.")

else:
    # S'affiche si le geo_data n'a pas pu être chargé
    st.warning("Le chargement des données de la carte a échoué.")
    st.info("Veuillez vérifier les messages d'erreur et recharger la page.")