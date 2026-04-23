# AI-Based Document Analysis System (Streamlit)

An interactive Streamlit app to upload documents (PDF, DOCX, TXT), ask natural language questions, and get document-grounded answers with relevant context.

## Step-wise Implementation

### Step 1: Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Run the app
```bash
streamlit run app.py
```

### Step 3: Upload document
- Supported types: `PDF`, `DOCX`, `TXT`
- The app extracts text with section metadata (page/paragraph/line).
- Upload history is persisted in `.upload_history.json`.

### Step 4: Ask questions
- Enter a natural language question in the input box.
- The system:
  - Splits text into chunks
  - Retrieves the most relevant chunk using TF-IDF similarity
  - Uses transformer QA model (if available) to extract an answer
  - Falls back to extractive answer if model is unavailable

### Step 5: View relevant content
- The app shows the most relevant passage with source citation.
- Query terms are highlighted for readability.
- Top 3 matching chunks are shown with similarity scores.

### Step 6: Optional enhancements already included
- Summary generation
- Keyword search with context snippets
- Upload history tracking (persistent)
- Light/Dark mode toggle
- Document preview tab
- Downloadable Q&A report (TXT/PDF)
- Multi-document answer comparison
- Comparison export (CSV/PDF)

## Tech Stack
- Python
- Streamlit
- pypdf, python-docx
- scikit-learn, numpy
- transformers, torch

## Notes
- First run may take longer while transformer models are downloaded.
- If model download fails, fallback logic still returns grounded answers.
