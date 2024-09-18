import os
import streamlit as st
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
GOOGLE_API_KEY = "AIzaSyCBAbXqCGh2b3hI04m8V5KrvliTyEkbl3U"

# Configure Google Generative AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Gemini Pro model with API key
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    max_tokens=None,
    convert_system_message_to_human=True,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,  # Pass the API key here
)

# Streamlit app
st.title('Probahini - Menstruation Expert Chatbot')

# User input
user_message = st.text_input("Ask a question:")

if st.button("Submit"):
    if user_message:
        # Define the primary prompt for Probahini
        prompt = (
            "You are Probahini, a chatbot knowledgeable on menstrual health issues. "
            "Provide detailed and specific answers related to menstruation, health, and hygiene."
        )

        query = f"{prompt}\n\nUser Question: {user_message}"

        # Call the Gemini model
        try:
            response = model.invoke([
                ("system", prompt),
                ("human", user_message)
            ], safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            })

            bot_response = response.content

        except Exception as e:
            logger.error(f"Error with Gemini model: {e}")
            bot_response = "I'm sorry, I couldn't retrieve an answer to your question."

        # Display the response
        st.write(bot_response)
    else:
        st.write("Please enter a question to get a response.")
