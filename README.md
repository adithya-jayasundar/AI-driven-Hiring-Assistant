#  AI Hiring Engine – Resume Ranking, Explainability & Search

An AI-powered assistant designed to automate and enhance the candidate screening process by parsing job descriptions and resumes, performing intelligent semantic search, and generating ranked reports.

---

##  Key Features

* **Job Description (JD) Parsing:** Uses an LLM to accurately extract essential requirements and skills from job descriptions.
* **Structured Resume Extraction:** Parses Resume PDFs (`PyPDF2`) into clean, structured data for processing.
* **Vector Indexing:** Creates high-dimensional multi-chunk embeddings (`Sentence Transformers - MiniLM`) and indexes them in a dedicated vector store (`ChromaDB`).
* **Hybrid Semantic Search:** Combines the power of:
    * **Dense Search** (semantic vector similarity)
    * **Sparse Search** (keyword matching)
    * **Fuzzy Search** (typo tolerance)
* **Advanced Candidate Ranking:** Scores candidates based on a comprehensive set of criteria:
    * Technical and functional **Skills**
    * Relevance and depth of **Experience**
    * **Recency** of past projects and roles
    * Relevance of portfolio **Projects**
    * Assessment of **Soft Skills**
* **LLM-based Explainability:** Generates detailed reports (`OpenRouter LLM - Mistral`) justifying the fit score and ranking of each candidate.
* **Report Generation:** Exports the full ranking and explainability report as a shareable PDF (`ReportLab`).
* **Intuitive UI:** A user-friendly, interactive frontend built with **Streamlit**.

---

##  Technology Stack

| Category | Technology / Library | Role |
| :--- | :--- | :--- |
| **Language** | Python | Core programming language. |
| **Frontend** | Streamlit | Creating the web-based user interface (`app.py`). |
| **LLM** | OpenRouter LLM (Mistral) | Parsing, extraction, and explainability report generation. |
| **Embeddings** | Sentence Transformers (MiniLM) | Generating semantic vector embeddings. |
| **Vector DB** | ChromaDB | High-performance storage and retrieval of vector embeddings. |
| **PDF Processing** | PyPDF2 | Extracting text content from uploaded Resume PDFs. |
| **Report Output** | ReportLab | Generating the final, structured PDF reports. |

---

##  Project Structure

| File | Description |
| :--- | :--- |
| `backend.py` | Contains the core business logic: LLM parsing, vector indexing, candidate scoring algorithm, and hybrid search implementation. |
| `app.py` | The Streamlit application entry point. Handles all user interface elements and input/output processing. |
| `requirements.txt` | Lists all necessary Python dependencies (libraries) required to run the project. |

---



### Workflow

1.  Input or upload a **Job Description** (JD).
2.  Upload candidate **Resume PDFs**.
3.  The system indexes the data and performs the **Hybrid Search**.
4.  View the **Ranked Candidates** in the Streamlit interface.
5.  Generate and download the **Explainability Report** PDF for selected candidates.
