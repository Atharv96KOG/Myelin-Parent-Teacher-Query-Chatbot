# 🎓 EduBot – AI Powered Parent–Teacher Assistant

EduBot is an advanced **Retrieval-Augmented Generation (RAG)** based school assistant that helps parents and teachers access student data, academic performance, attendance, transport details, and school policies through an intelligent multilingual chatbot.

Built with **LLMs, Vector Databases, MongoDB, and Streamlit**, EduBot delivers fast, contextual, and voice-enabled responses.

---

## 🚀 Features

✅ AI-powered chatbot using **Llama 3**
✅ Retrieval-Augmented Generation (RAG) architecture
✅ Voice input + audio responses
✅ Automatic language detection & translation
✅ Student performance dashboard with charts
✅ Beautiful PDF report generation
✅ MongoDB + ChromaDB hybrid database
✅ Offline translation using Facebook NLLB
✅ Clean and modular architecture

---

## 🧠 Tech Stack

* Python
* Streamlit
* Ollama (Llama3)
* MongoDB
* ChromaDB (Vector Database)
* LangChain Embeddings
* Transformers (HuggingFace)
* PyTorch
* gTTS (Text-to-Speech)
* Matplotlib & Pandas
* ReportLab (PDF generation)

---

## 📂 Project Structure

```
EduBot/
│
├── app.py
├── chroma_db/
├── assets/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Atharv96KOG/Myelin---Edubot.git
cd Myelin---Edubot
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

**Windows**

```
.venv\Scripts\activate
```

**Mac/Linux**

```
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install streamlit langdetect pymongo chromadb sentence-transformers torch transformers ollama gTTS streamlit-mic-recorder pandas matplotlib reportlab
```

---

### 4️⃣ Install Ollama & Pull Model

Download:

👉 [https://ollama.com/download](https://ollama.com/download)

Then run:

```bash
ollama pull llama3:8b
```

---

### 5️⃣ Install MongoDB

Download:

👉 [https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)

Ensure MongoDB is running on:

```
mongodb://localhost:27017/
```

---

### 6️⃣ Seed the Database (IMPORTANT)

```bash
python app.py --seed
```

This generates demo students, curriculum, attendance, and vector embeddings.

---

### 7️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🎤 Example Queries

* “Show Aarav's attendance”
* “What is the fee structure?”
* “Tell me pending homework”
* “When are the exams?”
* “Bus route details”

---

## 🔮 Future Improvements

* Cloud deployment
* WhatsApp integration
* Teacher login panel
* Real-time parent notifications
* Fine-tuned education LLM

---



---

⭐ If you like this project, consider giving it a star!
