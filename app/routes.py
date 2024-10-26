import os
from flask import Blueprint, render_template, request, jsonify, session, redirect
import firebase_admin
from firebase_admin import auth, firestore
import requests
from werkzeug.security import generate_password_hash, check_password_hash

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

main = Blueprint('main', __name__)

# Initialize Firebase Admin SDK with the service account key
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

def get_weather_by_coordinates(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
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