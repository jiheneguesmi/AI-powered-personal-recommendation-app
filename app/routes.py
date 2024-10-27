import os
import pandas as pd
from .config import Config
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from .forms import RecommendationForm
from google_maps_reviews import ReviewsClient
from outscraper import ApiClient
import firebase_admin
from firebase_admin import auth, firestore
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from .forms import RecommendationForm

#Importation Tache 3
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import numpy as np


# Create a Flask Blueprint
main = Blueprint('main', __name__)
geolocator = Nominatim(user_agent="geoapiExercises")



# Initialize Firebase Admin SDK
cred = firebase_admin.credentials.Certificate(os.path.join(os.path.dirname(__file__), 'tunisia-tourism-firebase-adminsdk-seadi-18c763db2c.json'))
firebase_admin.initialize_app(cred)
db = firestore.client()

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        username = data['username']
        email = data['email']
        password = data['password']
        
        # Check if the email already exists
        try:
            auth.get_user_by_email(email)
            return jsonify({'error': 'There is already an account with this email.'}), 400
        except auth.UserNotFoundError:
            pass  # Email doesn't exist, we can proceed

        # Check if the username already exists
        username_ref = db.collection('usernames').document(username)
        if username_ref.get().exists:
            return jsonify({'error': 'Username already exists. Please choose another one.'}), 400

        # Create a new user in Firebase Authentication
        user = auth.create_user(email=email, password=password)
        hashed_password = generate_password_hash(password)
        # Save username and email to Firestore without the password
        username_ref.set({
            'email': email,
            'uid': user.uid,  # Store the user's UID in Firestore
            'password': hashed_password

        })

        # Store username in session
        session['username'] = username
        return jsonify({'success': True})

    return render_template('signup.html')

@main.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        data = request.get_json()
        email = data['email']
        password = data['password']

        try:
            # Get user by email from Firebase Authentication
            user = auth.get_user_by_email(email)
            
            # Fetch the associated username from Firestore
            usernames_ref = db.collection('usernames')
            query = usernames_ref.where('email', '==', email).limit(1).get()

            if not query:
                return jsonify({'error': 'User not found.'}), 400

            # Assuming there's only one document with this email
            user_data = query[0].to_dict()
            username = query[0].id  # The document ID is the username
            
            # Verify the password using the hashed password stored in Firestore
            if check_password_hash(user_data['password'], password):
                session['username'] = username
                return jsonify({'success': True})
            else:
                return jsonify({'error': 'Invalid password.'}), 400

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('signin.html')

@main.route('/profile')
def profile():
    username = session.get('username')
    if username:
        return render_template('index_profile.html', username=username)
    return redirect('/')  # Redirect to home if not logged in

@main.route('/local_experiences')
def find_local_experiences():
    _username = session.get('username')
    if _username:
        return render_template('find_local_experiences.html', username=_username)
    return redirect('/')

@main.route('/find_places', methods=['GET', 'POST'])
def find_places():
    _username = session.get('username')
    if request.method == 'POST' and _username:
        _region = request.form.get('region')
        _state = request.form.get('state')
        if (not _region) and (_state == 'Choose...'):
            return render_template('find_local_experiences.html', username=_username)
        return render_template('find_places.html', username=_username, region=_region, state=_state)
    if request.method == 'GET' and _username:
        _region = request.args.get('region')
        _state = request.args.get('state')
        if (not _region) and (not _state):
            return render_template('find_local_experiences.html', username=_username)
        return render_template('find_places.html', username=_username, region=_region, state=_state)
    return redirect('/')

@main.route('/places', methods=['GET', 'POST'])
def places():
    _username = session.get('username')
    if request.method == 'GET' and _username:
        p = request.args.get('p')
        region = request.args.get('region')
        state = request.args.get('state')
        PlacesClient = ApiClient(api_key='YWZkZWMzNzk3NDY0NGYyMTgwODQwOGU1ZjlkNjliMWN8MDFhYjAyNDc1NA')
        searchString = p
        if region:
            searchString = searchString + ' ' + region
        if state != 'Choose...':
            searchString = searchString + ' ' + state
        results = PlacesClient.google_maps_search(
        [searchString],
        limit=10, # limit of palces per each query
        language='en',
        region='TN',
        )
        ll = list()
        for query_places in results:
            for place in query_places:
                dd = dict()
                dd['place_id'] = place['place_id']
                dd['name'] = place['name']
                dd['phone'] = place['phone']
                dd['photo'] = place['photo']
                ll.append(dd)
        return render_template('places.html', username=_username, places=ll)
    return redirect('/')

@main.route('/getreviews', methods=['GET', 'POST'])
def reviews():
    _username = session.get('username')
    if not _username:
        return redirect('/')
    if request.method == 'GET':
        id = request.args.get('id')
        name = request.args.get('name')
        if id and name:
            #credential = AzureKeyCredential("9qd2zBKHEMhgfOzp35pcsoIkJnCUb5iIctjIrRqso2UJbpAq3yojJQQJ99AJAC5RqLJXJ3w3AAAaACOGHNAy")
            #text_analytics_client = TextAnalyticsClient(endpoint="https://applicationhackathon.cognitiveservices.azure.com/", credential=credential)
            try:
                # Initialize the API client and call it if 'id' is present
                ReviewsClient = ApiClient(api_key='YWZkZWMzNzk3NDY0NGYyMTgwODQwOGU1ZjlkNjliMWN8MDFhYjAyNDc1NA')
                os.environ['AZURE_TEXT_ANALYTICS_ENDPOINT'] = 'https://applicationhackathon.cognitiveservices.azure.com/'
                os.environ['AZURE_TEXT_ANALYTICS_KEY'] = '9qd2zBKHEMhgfOzp35pcsoIkJnCUb5iIctjIrRqso2UJbpAq3yojJQQJ99AJAC5RqLJXJ3w3AAAaACOGHNAy'
                endpoint = os.environ['AZURE_TEXT_ANALYTICS_ENDPOINT']
                key = os.environ['AZURE_TEXT_ANALYTICS_KEY']
                text_analytics_client = TextAnalyticsClient(endpoint, AzureKeyCredential(key))
                results = ReviewsClient.google_maps_reviews([id], reviews_limit=10, language='en')
                ll = list()
                index_total = 0
                index_negative = 0
                index_positive = 0
                index_neutral = 0
                index_total = 0
                index_positive = 0
                for review in results:
                    review_data = review.get('reviews_data')
                    for rr in review_data:
                        dd = dict()
                        dd['author_title'] = rr.get('author_title')
                        dd['review_text'] = rr.get('review_text')
                        index_total = index_total + 1
                        if dd.get('author_title') is not None and dd.get('review_text') is not None:
                            response = text_analytics_client.analyze_sentiment([dd.get('review_text')])
                            for result in response:
                                if result.kind == "SentimentAnalysis":
                                    if (result.sentiment == 'positive'):
                                        index_positive = index_positive + 1
                                        ll.append(dd)
                                    if (result.sentiment == 'neutral'):
                                        index_neutral = index_neutral + 1
                                    if (result.sentiment == 'negative'):
                                        index_negative = index_negative + 1
                return render_template('reviews.html', username=_username, score = (index_positive / index_total), score_positive=(index_positive / index_total), score_neutral=(index_neutral / index_total),score_negative=(index_negative / index_total), place_name=name, llist=ll)
            except Exception as e:
                # Log the error and render an error page
                print(f"API client error: {e}")
        else:
            # No `id` in request, render `find_local_experiences.html`
            return render_template('find_local_experiences.html', username=_username)


@main.route('/signout')
def signout():
    session.pop('username', None)  # Remove the username from the session
    return redirect('/')  # Redirect to home page
def get_weather_by_coordinates(api_key, lat, lon):
    # First, get the current weather
    current_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        current_response = requests.get(current_url)
        current_response.raise_for_status()
        current_weather = current_response.json()

        # Now, get the hourly forecast
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        forecast_response = requests.get(forecast_url)
        forecast_response.raise_for_status()
        hourly_forecast = forecast_response.json()

        # Combine the current weather and hourly forecast
        return {
            'current': current_weather,
            'hourly': hourly_forecast['list'][:12]  # Get next 12 hours of forecast
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")  # Log the error
        return None

@main.route('/weather', methods=['GET'])
def weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if lat and lon:
        weather_data = get_weather_by_coordinates(Config.WEATHER_API_KEY, lat, lon)
        if weather_data:
            return jsonify(weather_data)  # Send JSON response
        else:
            return {"error": "Error fetching weather data."}, 500
    return {"error": "No location provided."}, 400




@main.route('/preferences')
def preferences():
    _username = session.get('username')
    if not _username:
        return redirect('/')
    return render_template('preferences.html', username=_username)

@main.route('/submit_preferences', methods=['GET', 'POST'])
def submit_preferences():
    form_data = request.json
    user_preferences = (
        f"Âge : {form_data['age']}, "
        f"Genre : {form_data['gender']}, "
        f"Nationalité : {form_data['nationality']}, "
        f"Fréquence des voyages en Tunisie : {form_data['travelFrequency']}, "
        f"Type de vacances : {form_data['vacationType']}, "
        f"Budget : {form_data['Budget']}, "
        f"Durée du séjour : {form_data['stayDuration']}, "
        f"Saison préférée : {form_data['season']}, "
        f"Taille du groupe : {form_data['groupSize']}, "
        f"Category : {', '.join(form_data['category'])}, "
        f"Importance accordée au tourisme durable : {form_data['sustainableTourism']}, "
        f"Prêt à faire des compromis : {form_data['compromise']}, "
        f"Activités spécifiques : {form_data['specificActivities']}"
    )
    
    print(user_preferences)
   
    #return jsonify({"recommendations":user_preferences })
    return (user_preferences)


@main.route('/weather-location', methods=['GET'])
def weather_page():
    _username = session.get('username')
    if not _username:
        return redirect('/')
    _location = request.args.get('location')
    return render_template('weather.html', username=_username, location=_location)


def get_latitude_longitude(location):
    try:
        location_data = geolocator.geocode(location, timeout=10)
        if location_data:
            return location_data.latitude, location_data.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        return get_latitude_longitude(location)


# Charger les données depuis le fichier CSV
df = pd.read_csv(r"C:\\Users\\IMINFO\\tunisian_tourism\\app\\tunisie_destinations.csv")
df = df.fillna('unknown')

# Convertir les colonnes catégorielles en numériques avec One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['name', 'category', 'location', 'budget'])
knn = NearestNeighbors(n_neighbors=5, metric='cosine')

def train_knn_model():
    """Entraîne le modèle KNN avec les données encodées."""
    knn.fit(df_encoded)

def recommend_destinations(budget, category):
    # Filtrage des destinations en fonction du budget et de la catégorie
    filtered_destinations = df[(df['budget'] == budget) & (df['category'] == category)]
    
    if not filtered_destinations.empty:
        # Encodage de la destination filtrée
        filtered_destination_encoded = pd.get_dummies(filtered_destinations.iloc[0:1], 
                                                    columns=['name', 'category', 'location', 'budget'])
        
        # Aligner les colonnes avec df_encoded
        missing_cols = set(df_encoded.columns) - set(filtered_destination_encoded.columns)
        for col in missing_cols:
            filtered_destination_encoded[col] = 0
        filtered_destination_encoded = filtered_destination_encoded[df_encoded.columns]
        
        # Calculer les similarités
        similarities = cosine_similarity(filtered_destination_encoded, df_encoded)
        similar_indices = similarities.argsort()[0][::-1][:3]
        return df.iloc[similar_indices].to_dict(orient='records')
    else:
        # Si aucune correspondance exacte, utiliser KNN
        # Créer un vecteur avec les préférences de l'utilisateur
        user_pref_encoded = pd.get_dummies(pd.DataFrame([[None, category, None, budget]], 
                                         columns=['name', 'category', 'location', 'budget']))
        
        # Aligner les colonnes
        missing_cols = set(df_encoded.columns) - set(user_pref_encoded.columns)
        for col in missing_cols:
            user_pref_encoded[col] = 0
        user_pref_encoded = user_pref_encoded[df_encoded.columns]
        
        distances, indices = knn.kneighbors(user_pref_encoded)
        return df.iloc[indices[0]].to_dict(orient='records')

@main.route('/recommendation', methods=['GET', 'POST'])
def show_recommendation_form():
    form = RecommendationForm()
    username = session.get('username')
    return render_template('recommend.html', form=form, username=username)

@main.route('/recommend', methods=['POST'])
def recommend():
    form = RecommendationForm()
    if form.validate_on_submit():
        budget = form.budget.data
        category = form.category.data
        
        # S'assurer que le modèle est entraîné
        train_knn_model()
        
        # Obtenir les recommandations
        recommended_destinations = recommend_destinations(budget, category)
        
        return render_template('result.html', destinations=recommended_destinations)
    
    # En cas d'erreur de validation, retourner au formulaire
    return redirect(url_for('main.show_recommendation_form'))



#Tache3
#Architecture RAG
#Lecture et Analyse des données du site Web
#Utilisation du chargeur de Documents WebBaseLoader de LangChain  

pages = PyPDFLoader(r"C:\\Users\\IMINFO\\tunisian_tourism\\app\\templates\\activity-tourisme-tunisie.pdf").load_and_split()
print("Read {0} pages".format(len(pages)))
alltext = " ".join(p.page_content.replace("\n", " ") for p in pages)

import nltk
nltk.download('punkt_tab')
lengths=[]
cleantext = ""

for s in nltk.sent_tokenize(alltext):
    if(len(s) > 1 and len(s) < 2000):
        lengths.append(len(s))
        cleantext = cleantext + s + " "

print("Number of sentences: {0}".format(len(lengths)))
print("Clean text length %d" % len(cleantext))

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=380,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)

docs = text_splitter.create_documents([cleantext])
chunks = [Document(page_content=d.page_content) for d in docs]

# Initialiser les embeddings Google Generative AI
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = FAISS.from_documents(chunks, gemini_embeddings)
vectorstore.save_local("ActivityDurableDB.faiss")

#test vectorDB
"""vectorstore = FAISS.load_local("ActivityDurableDB.faiss", embeddings=gemini_embeddings,  allow_dangerous_deserialization=True)

res = vectorstore.search("Que montre lanalyse de lactivité touristique en tunisie ?", search_type="similarity")

for d in res:
    print(d.page_content)"""

# Obtenir l'interface Retriever
vscontext = vectorstore.as_retriever()
#print(vscontext)
# Initialiser le modèle LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", temperature=0.7, top_p=0.85)

# Définir le template de l'invite
llm_prompt_template = """Tu es un expert en tourisme en Tunisie.
Ta tâche est de recommander UNIQUEMENT 2 activités touristiques disponibles à {user_places} ou dans sa région immédiate, en tenant en compte {user_preferences}.

Gafsa
Découverte des techniques d'irrigation traditionnelles
Participation aux récoltes de dattes biologiques
Randonnées dans les oasis
Nuits en maison troglodyte écologique
Apprentissage des techniques de permaculture

Sud de la Tunisie
Parc National de Jbil
Safari photos respectueux de la faune
Randonnées chamelières avec guides locaux
Observation des étoiles (astrotourisme)
Découverte de la flore saharienne avec un botaniste
Participation aux projets de préservation des espèces
Zaghouan: Rue de Zaghouan vers 1900 Idéal pour une séjour moins 2 semaines, moins d’un mois ou un mois au plus pour les tranches d’age 18-25, 26-35 ou 36-45 pour homme ou femme toute nationnalité 
peu importe la taille du groupe 1,2,3,4,5 personnes et plus.
Sousse : Visite de Port El Kantaoui
ribat de Monastir : Ribat de Monastir, Palais présidentiel Habib Bourguiba
ouverte des techniques traditionnelles de distillation du bigaradier

Ksar Ghilane
Hébergement en campements écologiques
Baignade dans les sources thermales naturelles
Randonnées à pied dans les dunes
Découverte de l'architecture traditionnelle des ksour
Participation aux projets de stabilisation des dunes

Île de Djerba
Circuit à vélo des potiers traditionnels
Observation des oiseaux dans la lagune de Bin El Ouediane
Visite des marchés bio locaux
Participation à la récolte des olives
Découverte de la pêche traditionnelle durable

Bonnes Pratiques Écotouristiques
Transport
Privilégier les transports en commun
Utiliser les vélos et la marche pour les courtes distances
Opter pour des véhicules électriques quand disponible
Participer au covoiturage local

pour Tamerza tu peux recommander les activités
-Randonnée autour de l'oued et des cascades près du village

pour Matmata:Découverte des pièces troglodytes creusées dans la montagne, cuisine traditionnelle

Uthina
-Site archéologique
-édifice religieux et pièces voûtées aménagées sous le capitole

Pour Jendouba tu peux proposer comme activité Site archéologique de Bulla Regia

Question: {context}

Important:
- Ne propose que des activités réellement disponibles à {user_places} ou dans un rayon très proche
- Ne mentionne PAS d'activités dans d'autres villes de Tunisie
-Toutes les activités que que va proposer sont Idéal pour une séjour moins 2 semaines, moins d’un mois ou un mois au plus pour les tranches d’age 18-25, 26-35 ou 36-45 pour homme ou femme toute nationnalité 
peu importe la taille du groupe 1,2,3,4,5 personnes et plus.
- Propose un programme de la journée en fonction de tes connaissance générales au user pour bien méner l'activité c'est une recommandation pour lui

IMPORTANT : Tu DOIS structurer ta réponse exactement selon ce format JSON, sans aucune autre explication :

accolade
    "activites": [
        accolade
            "nom": "Nom de l'activité 1",
            "description": "Description détaillée",
            "lieu": "Emplacement précis",
            "budget": "XX TND",
            "duree": "Durée estimée",
            "conseils": "Conseils pratiques"
        accolade,
        accolade
            "nom": "Nom de l'activité 2",
            "description": "Description détaillée",
            "lieu": "Emplacement précis",
            "budget": "XX TND",
            "duree": "Durée estimée",
            "conseils": "Conseils pratiques"
        accolade
    ],
    "programme facultatif pur vous": accolade
        "jour": accolade
            "matin": "Activité du matin",
            "apresmidi": "Activité de l'après-midi",
            "soir": "Activité du soir"
        accolade
    accolade
accolade

Ne fournis que le JSON, sans texte avant ou après. Assure-toi que le JSON est valide et correctement formaté.
Réponse:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# Créer la chaîne RAG
rag_chain = (
    {"context": vscontext, "user_places": RunnablePassthrough(), "user_preferences": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)
import json

@main.route('/locationChoice')
def locationChoix():
    destination_name = request.args.get('destination_name')
    return render_template('locationChoice.html', destination_name=destination_name)
@main.route('/submit_location', methods=['POST'])
def submit_location():
    try:
        form_data = request.json
        print("Received form data:", form_data)  # Debug log

        user_places = form_data.get('destination_name')
        user_preferences = form_data.get('preferences')

        print("User places:", user_places)  # Debug log
        print("User preferences:", user_preferences)  # Debug log

        # Créer le prompt avec la ville spécifique et les préférences de l'utilisateur
        query = f"Quelles sont les activités touristiques disponibles à {user_places} en tenant en compte {user_preferences}?"
        print("Query:", query)  # Debug log

        # Appeler la chaîne RAG
        response = rag_chain.invoke(query, user_places=user_places, user_preferences=user_preferences)
        print("Raw response from LLM:", response)  # Debug log

        # Vérifier si la réponse est déjà un JSON valide
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)  # Debug log
                return jsonify({
                    "error": "Invalid JSON response from LLM",
                    "raw_response": response
                }), 500

        return jsonify({"recommendations": json.dumps(response)})

    except Exception as e:
        print("Error in submit_location:", str(e))  # Debug log
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500
        
#Fin Tache 3


@main.route('/feed_back', methods=['GET', 'POST'])
def feed_back():
    _username = session.get('username')
    if not _username:
        return redirect('/')
    return render_template('feed_back.html', username=_username)

@main.route('/save_feed', methods=['POST'])
def save_feed():
    _username = session.get('username')
    if not _username:
        return redirect('/')
    relevancy = request.form.get('recom')
    activities = request.form.get('actvs')
    will_return = request.form.get('return')
    feel = request.form.get('feel')
    about = request.form.get('act')
     # Define the data to be saved
    review_data = {
        'username': _username,
        'relevancy': relevancy,
        'activities': activities,
        'will_return': will_return,
        'feel': feel,
        'about': about,
        'timestamp': firestore.SERVER_TIMESTAMP  # adds a timestamp field
    }

    try:
        # Save data to Firestore collection `reviews`
        db.collection('reviews').add(review_data)
        return render_template('index_profile.html', username=_username)
    except Exception as e:
        # Handle potential errors
        print(f"An error occurred: {e}")
        return render_template('error.html', error="Could not save your feedback. Please try again.")
