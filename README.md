# AI-powered-personal-recommendation-app
 ## Project Overview
The Eco-Friendly Personalized Tourism Application is a web-based platform aimed at enhancing the experience of tourists in Tunisia through personalized, AI-driven recommendations. This application considers user preferences, real-time data, and eco-friendly tourism practices to help users discover hidden gems, plan their journeys more effectively, and promote sustainable tourism in Tunisia.

Key Features
User Onboarding Form

Users enter their travel preferences, such as budget and type of tourism (e.g., cultural, religious, historical, natural).
Tailors recommendations to individual tastes and interests.
K-Nearest Neighbors (KNN) with Cosine Similarity

The KNN algorithm matches user preferences with destinations in our dataset, suggesting the top three closest matches.
This ensures personalized recommendations based on proximity between user-specified characteristics and existing destination profiles.
Immersive Destination Exploration

Users receive detailed information on suggested destinations, including associated locations like restaurants, cafés, and attractions.
Reviews and ratings from other visitors provide an immersive experience, helping users make informed decisions.
Sentiment Analysis with Azure AI Language

Reviews are classified into three categories: positive, negative, and neutral.
Sentiment scores are calculated based on the proportion of positive reviews, helping users gauge the popularity and appeal of destinations.
User Profile Form

A secondary form gathers detailed user profile data, including age, gender, nationality, vacation type preference, budget, stay duration, preferred season, group size, and personal interests.
This information allows for even more refined recommendations, contributing to a tailored travel experience.
Activity Generation Using RAG and LLM Gemini

Uses Retrieval-Augmented Generation (RAG) and the LLM Gemini to dynamically generate activity suggestions based on individual profile characteristics.
Focuses on promoting eco-responsible and sustainable options to encourage environmentally-friendly tourism.
Real-Time Weather Forecasts

The platform provides up-to-date weather forecasts for each destination, helping users plan activities with local conditions in mind.
Technologies Used
Flask: Backend framework for web development.
Machine Learning: KNN algorithm with cosine similarity for recommendation generation.
Azure AI Language: Sentiment analysis for visitor reviews.
RAG (Retrieval-Augmented Generation) and LLM Gemini: For generating activity recommendations.
Weather API: To fetch real-time weather data for each destination.
Installation
1-Clone the Repository:
git clone https://github.com/your-account/AI-powered-personal-recommendation-app.git
cd AI-powered-personal-recommendation-app
2-Set Up the Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3-Install Required Libraries:
pip install -r requirements.txt
4-Configure API Keys
Azure AI Language API: Set up for sentiment analysis.
Weather API: Configure to fetch real-time weather data.
Place your API keys in a .env file or directly in your configuration files as needed.
5-Run the Application:
flask run

# Usage
User Preferences and Recommendations
Upon starting, users can specify their preferences and receive personalized destination recommendations.
Explore Destinations and Reviews
Users can dive into detailed profiles of suggested destinations, check reviews, and view sentiment scores.
Activity Suggestions
Based on the user profile, activity recommendations are generated to enhance the travel experience with eco-friendly options.
Weather Updates
The app displays weather conditions for each recommended destination, aiding in trip planning.

# Contributing
Feel free to contribute by opening issues, suggesting new features, or submitting pull requests.
