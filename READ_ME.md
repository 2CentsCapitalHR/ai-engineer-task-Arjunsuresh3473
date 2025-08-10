# ADGM Document Review App

This web app helps review company legal documents for Abu Dhabi Global Market (ADGM) compliance. Upload your `.docx` files, and the app uses AI to detect any legal issues and adds comments inside the documents.

---

## Features

- Upload multiple `.docx` legal documents.
- AI-powered analysis using Google Gemini.
- Finds potential legal red flags and suggests fixes.
- Adds comments directly inside the documents.
- Download reviewed `.docx` files and summary reports.

---

## Setup Instructions

### 1. Install Python

Make sure Python 3.8 or higher is installed on your computer. You can download it from [python.org](https://www.python.org/downloads/).

### 2. Clone or Download the Project

Get the project files onto your computer.

### 3. Create and Activate a Virtual Environment (optional but recommended)

**On Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Required Packages

Make sure you have a file named `requirements.txt` in the project folder with these contents:

```
streamlit
python-docx
unidecode
langchain
google-generativeai
faiss-cpu
```

Then run:

```bash
pip install -r requirements.txt
```

### 5. Prepare Reference Documents

Create a folder `data/reference` inside the project folder. Add some ADGM-related `.docx` or `.txt` reference documents here. These help the AI understand ADGM rules better.

### 6. Get Google Gemini API Key

Obtain your Gemini API key from Google’s cloud platform and keep it handy.

### 7. Run the App

Run the following command from the project folder:

```bash
streamlit run app_streamlit.py
```

Open the URL it shows (usually `http://localhost:8501`) in your browser.

### 8. Use the App

- Enter your Gemini API key in the app.
- Upload your `.docx` documents.
- Click **Review Documents**.
- Download reviewed files and reports once analysis is done.

---

## Troubleshooting

- Make sure the `data/reference` folder exists with some documents.
- Verify your Gemini API key is correct.
- If FAISS installation fails, try installing `faiss-cpu` or check FAISS installation guides.

---

## Need Help?

Feel free to ask if you have any questions or need assistance setting up or using the app!


