from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Set Google Gemini API Key
GENAI_API_KEY = "AIzaSyD3nzvABZbugVaM5yF-49WT986YSqs1aCc"
genai.configure(api_key=GENAI_API_KEY)

# Load the Gemini model
try:
    model = genai.GenerativeModel("gemini-1.5-pro")
except Exception as e:
    print("Error loading Gemini model:", str(e))

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.json
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "No question provided"}), 400


        # Generate AI response
        health_prompt = f"Provide a professional healthcare response. Only answer healthcare-related questions. If the question is not related to medicine, politely refuse. \n\n Question: {question}"
        response = model.generate_content(health_prompt)


        # Extract the model's answer
        if not hasattr(response, "text"):
            return jsonify({"error": "Invalid response from AI"}), 500

        answer = response.text
        return jsonify({"answer": answer})

    except Exception as e:
        print("Error processing request:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Flask API running on port 5000")
    app.run(debug=True, port=5000)
