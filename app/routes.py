import os
import pandas as pd

from flask import Blueprint, render_template, request, jsonify, session, redirect
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

<<<<<<< HEAD
#Importation Tache 3
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
#from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import numpy as np

=======
# Create a Flask Blueprint
>>>>>>> ff29182c1a2dcfbd8ad1b8667e40e473d9733042
main = Blueprint('main', __name__)
geolocator = Nominatim(user_agent="geoapiExercises")
# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), 'tunisie_destinations.csv')
df = pd.read_csv(data_path)
df.fillna('unknown', inplace=True)

# Convert categorical columns to numeric with One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['name', 'category', 'location', 'budget'])

# Initialize the KNN model
knn = NearestNeighbors(n_neighbors=3, metric='cosine')
knn.fit(df_encoded)

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
                for review in results:
                    review_data = review.get('reviews_data')
                    for rr in review_data:
                        dd = dict()
                        dd['author_title'] = rr.get('author_title')
                        dd['review_text'] = rr.get('review_text')
                        if dd.get('author_title') is not None and dd.get('review_text') is not None:
                            response = text_analytics_client.analyze_sentiment([dd.get('review_text')])
                            for result in response:
                                if result.kind == "SentimentAnalysis":
                                    if (result.sentiment == 'positive') or (result.sentiment == 'neutral'):
                                        ll.append(dd)
                return render_template('reviews.html', username=_username, place_name=name, llist=ll)
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

@main.route('/weather', methods=['GET'])
def weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if lat and lon:
        weather_data = get_weather_by_coordinates(WEATHER_API_KEY, lat, lon)
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

@main.route('/submit_preferences', methods=['POST'])
def submit_preferences():
    form_data = request.json
    user_preferences = (
        f"Âge : {form_data['age']}, "
        f"Genre : {form_data['gender']}, "
        f"Nationalité : {form_data['nationality']}, "
        f"Fréquence des voyages en Tunisie : {form_data['travelFrequency']}, "
        f"Type de vacances : {form_data['vacationType']}, "
        f"Budget : {form_data['budget']}, "
        f"Durée du séjour : {form_data['stayDuration']}, "
        f"Saison préférée : {form_data['season']}, "
        f"Taille du groupe : {form_data['groupSize']}, "
        f"Category : {', '.join(form_data['interests'])}, "
        f"Importance accordée au tourisme durable : {form_data['sustainableTourism']}, "
        f"Prêt à faire des compromis : {form_data['compromise']}, "
        f"Activités spécifiques : {form_data['specificActivities']}"
    )
    
    print(user_preferences)
   
    return jsonify({"recommendations":user_preferences })





def get_latitude_longitude(location):
    try:
        location_data = geolocator.geocode(location, timeout=10)
        if location_data:
            return location_data.latitude, location_data.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        return get_latitude_longitude(location)

@main.route('/recommend', methods=['POST'])
def recommend():
    _username = session.get('username')
    if not _username:
        return redirect('/')
    # Get user preferences from the form
    user_preferences = {
        'budget': request.form.get('budget'),
        'category': request.form.getlist('interests'),  # Get interests as a list
        #'location': 'Tunis'  # Example location, modify as needed
    }

    # Filter destinations that match user preferences
    recommended_df = df[(df['budget'] == user_preferences['budget']) & 
                        (df['category'].isin(user_preferences['category'])) ]

    # If no destinations match, use cosine similarity
    if recommended_df.empty:
        # Take the first destination as an example
        selected_destination = df_encoded.iloc[0].values.reshape(1, -1)
        similarities = cosine_similarity(selected_destination, df_encoded)
        similar_indices = similarities.argsort()[0][::-1]
        similar_destinations = df.iloc[similar_indices]
        recommended_destinations = similar_destinations.head(3)
    else:
        recommended_destinations = recommended_df.head(3)

    return render_template('result.html', username=_username, destinations=recommended_destinations.to_dict(orient='records'))

def get_weather_by_coordinates(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
<<<<<<< HEAD
        return None

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
print(vscontext)
# Initialiser le modèle LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", temperature=0.7, top_p=0.85)

# Définir le template de l'invite
llm_prompt_template = """Tu es un expert en tourisme en Tunisie. 
Ta tâche est de recommander UNIQUEMENT 2 activités touristiques disponibles à {user_places} ou dans sa région immédiate.
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
- Si tu n'as pas d'information spécifique pour {user_places},  fais des suggestions de lieux proche de {user_places}

Format pour chaque recommandation:
Activité : [nom de l'activité ]
Description : [description détaillée]
Lieu : [emplacement précis dans ou près de {user_places}]
Budget approximatif : [coût en TND]
Durée de l'activité : [durée estimée]
Conseils supplémentaires : [conseils pratiques]

Réponse:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# Créer la chaîne RAG
rag_chain = (
    {"context": vscontext, "user_places": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)

@main.route('/preferences')
def preferences():
    return render_template('locationChoice.html')

@main.route('/submit_preferences', methods=['POST'])
def submit_preferences():
    form_data = request.json
    user_places = form_data.get('city', 'Inconnu')

    print(user_places)
    # Créer le prompt avec la ville spécifique
    query = f"Quelles sont les activités touristiques disponibles à {user_places}?"
    
    # Appeler la chaîne RAG
    response = rag_chain.invoke(query, user_places=user_places)
    print (response)
    return jsonify({"recommendations": response})

#Fin Tache 3
=======
        return None
>>>>>>> ff29182c1a2dcfbd8ad1b8667e40e473d9733042
