# 💊 Healthcare AI Chatbot (Powered by Gemini GPT-4)

This is an **AI-powered healthcare chatbot** that answers **medical-related questions** using **Google's Gemini GPT-4 API**.
It **filters non-medical queries** and **provides medically accurate responses** based on trusted health sources (WHO, CDC, Mayo Clinic).

---

## **🚀 Features**

✔ **Provides healthcare-specific responses**
✔ **Blocks non-medical queries**
✔ **Interactive chat interface using Streamlit**
✔ **Flask API for backend processing**
✔ **Uses Gemini GPT-4 for intelligent medical responses**

---

## **📸 Screenshots**

### **1️⃣ Chatbot answering a medical question**

![Screenshot 1](chat_medical.png)

### **2️⃣ Chatbot rejecting a non-medical question**

![Screenshot 2](chat_reject.png)

### **3️⃣ Flask backend processing API request**

![Screenshot 3](api_response.png)

### **4️⃣ Complete chatbot interface with multiple queries**

![Screenshot 4](chat_history.png)

---

## **📌 Setup Instructions**

### 1️⃣ **Install Dependencies**

```bash
pip install flask flask-cors google-generativeai streamlit requests
2️⃣ Set Up API Key
Before running the Flask server, create your gemini api key and replace it in app.py file:

3️⃣ Start Flask Backend

python app.py
4️⃣ Run Streamlit Frontend
streamlit run ui.py

📌 Example Usage
✅ Medical Query
User: "What are the symptoms of diabetes?"
Bot: "Diabetes symptoms include excessive thirst, frequent urination, weight loss, and blurred vision. Please consult a doctor."

❌ Non-Medical Query
User: "Who is the president of the USA?"
Bot: "I'm a healthcare assistant and can only answer medical-related questions."

📌 Technologies Used
Flask – Backend API
Google Gemini GPT-4 API – AI Model
Streamlit – Frontend UI
Python – Programming Language

```
