# AI Document Intelligence & Research Assistant

An AI-powered document question-answering application that helps users understand PDF documents, retrieve relevant information, and receive answers grounded in document content with page citations.

## 🚀 Features

- **PDF Text Extraction:** Extract text from uploaded PDF documents.
- **Semantic Search:** Find relevant passages based on the meaning of a question.
- **AI-Powered Answers:** Generate answers using Google's Gemini API.
- **Page Citations:** Display page references and supporting passages for answers.
- **Document Chunking:** Divide document text into smaller passages for retrieval.
- **Interactive Web Interface:** Use the application through a Streamlit interface.

## 🛠️ Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- NumPy
- Google Gemini API

## ⚙️ How It Works

1. Upload a PDF document.
2. The application extracts its text.
3. The text is divided into smaller passages.
4. Sentence Transformers generates embeddings for semantic search.
5. The application retrieves relevant passages for your question.
6. Gemini generates an answer using the retrieved context.
7. The answer includes relevant page citations and supporting passages.

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Arjun1233333/AI-Document-Intelligence.git
cd AI-Document-Intelligence
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure your Gemini API key

Create a file named `secrets.toml` inside a `.streamlit` folder in the project directory.

Add your own Gemini API key:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Replace `YOUR_GEMINI_API_KEY` with your own key. Never commit or upload this secrets file to GitHub.

Alternatively, configure the `GEMINI_API_KEY` environment variable if supported by your application.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🔐 Security

- API keys should be stored in local secrets or environment variables.
- Do not upload `.streamlit/secrets.toml`, `.env` files, or other private credentials.
- Keep your API key private.

## 📁 Project Structure

```text
AI-Document-Intelligence/
├── .streamlit/
│   └── secrets.toml       # Local secrets; do not upload
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore              # Excludes private and unnecessary files
```

The `.streamlit/secrets.toml` file is created locally and should not be included in the public repository.

## 🎯 Project Objective

The goal of this project is to make document research easier by combining semantic retrieval and AI-generated answers with supporting document references.

## 👨‍💻 Author

**Arjun1233333**

GitHub: [Arjun1233333](https://github.com/Arjun1233333)

---

*This project is intended to assist with document research. Always verify important information against the original document.*
