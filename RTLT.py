from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
import os
import hashlib
import sqlite3
import threading
import subprocess

app = Flask(__name__)

# Database setup
def setup_database():
    """Sets up the SQLite database for user authentication."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

setup_database()

# Language options
LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "ja": "Japanese",
    "zh-cn": "Chinese (Simplified)", "it": "Italian", "ru": "Russian", "ko": "Korean",
    "ar": "Arabic", "pt": "Portuguese", "nl": "Dutch", "sv": "Swedish", "pl": "Polish",
    "tr": "Turkish", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay", "th": "Thai",
    "hi": "Hindi"
}

@app.route("/", methods=["GET", "POST"])
def index():
    """Handles all incoming requests."""
    if request.method == "POST":
        action = request.form.get("action")

        if action == "register":
            return handle_registration(request.form)
        elif action == "login":
            return handle_login(request.form)
        elif action == "translate":
            return handle_translation(request.form)
        elif action == "voice":
            return handle_voice(request.form)

    return render_template("index.html", languages=LANGUAGES)

def handle_registration(form_data):
    """Handles user registration."""
    username = form_data.get("username")
    password = form_data.get("password")
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Registration successful."})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Username already exists."})

def handle_login(form_data):
    """Handles user login."""
    username = form_data.get("username")
    password = form_data.get("password")
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    user = cursor.fetchone()
    conn.close()

    if user:
        return jsonify({"success": True, "message": "Login successful."})
    else:
        return jsonify({"success": False, "message": "Invalid username or password."})

def handle_translation(form_data):
    """Handles text translation."""
    input_text = form_data.get("input_text")
    target_lang = form_data.get("target_lang")

    translated_text = translate_text(input_text, target_lang)
    generate_voice_output(translated_text, target_lang)

    return jsonify({"translated_text": translated_text})

def handle_voice(form_data):
    """Handles voice input and translation."""
    try:
        target_lang = form_data.get("target_lang")
        threading.Thread(target=process_voice_request, args=(target_lang,)).start()
        return jsonify({"message": "Voice processing started."})
    except Exception as e:
        return jsonify({"error": str(e)})

def process_voice_request(target_lang):
    """Processes voice input, translation, and audio generation."""
    with app.app_context():
        input_text = capture_voice_input()
        translated_text = translate_text(input_text, target_lang)
        generate_voice_output(translated_text, target_lang)
        print("Voice processing complete")

def capture_voice_input():
    """Captures voice input using speech recognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Increased duration
            audio = recognizer.listen(source, timeout=15)  # Increased timeout
            print("Recognizing...")
            text = recognizer.recognize_google(audio, language='en-US')
            return text
    except sr.WaitTimeoutError:
        return "Listening timed out."
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except Exception as e:
        print(f"Error during voice capture: {e}")
        return "An unexpected error occurred."

def translate_text(input_text, target_language):
    """Translates text using Google Translate."""
    translator = Translator()
    try:
        translated = translator.translate(input_text, dest=target_language)
        return translated.text
    except Exception as e:
        return f"Translation error: {e}"

def generate_voice_output(text, language):
    """Generates voice output from text."""
    try:
        tts = gTTS(text, lang=language)
        tts.save("output.mp3")
        if os.name == 'nt':
            subprocess.Popen(['start', 'output.mp3'], shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
        elif os.name == 'posix':
            subprocess.Popen(['xdg-open', 'output.mp3'])
    except Exception as e:
        print(f"Voice output error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
