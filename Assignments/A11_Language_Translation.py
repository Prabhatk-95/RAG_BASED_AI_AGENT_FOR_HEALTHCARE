from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Load API key securely (recommended to set as an environment variable)
API_KEY = os.getenv("GEMINI_API_KEY", "DEFAULT_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1-translate:generate"

def translate_text_gemini(text, target_language):
    """
    Translate text using the Gemini API.
    """
    if not API_KEY or API_KEY == "DEFAULT_API_KEY":
        return {"error": "API key is missing or invalid."}, 500

    headers = {"Content-Type": "application/json"}
    payload = {"text": text, "target_language": target_language}

    try:
        response = requests.post(f"{BASE_URL}?key={API_KEY}", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return {"translation": data.get("translation", "Translation unavailable")}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}, 500

@app.route("/translate", methods=["POST"])
def translate():
    """
    Translate endpoint to process user inputs and return translated text.
    """
    data = request.get_json()
    
    # Validate input
    text = data.get("text")
    target_language = data.get("target_language")
    if not text or not target_language:
        return jsonify({"error": "Invalid input. Provide 'text' and 'target_language'."}), 400

    result, status_code = translate_text_gemini(text, target_language)
    return jsonify(result), status_code

if __name__ == "__main__":
    app.run(debug=True)
