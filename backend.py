
import os
import re
import json
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from PyPDF2 import PdfReader
import chromadb



OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = "mistralai/mistral-7b-instruct:free"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Embeddings
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# ChromaDB In-memory store
chroma_client = chromadb.Client()
COLLECTION_NAME = "hiring_engine_db"
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except:
    pass
collection = chroma_client.create_collection(name=COLLECTION_NAME)

NOW = datetime.now().year




class SkillItem(BaseModel):
    skill: str
    weight: float

class JobPostingSchema(BaseModel):
    job_title: str
    must_have: List[SkillItem]
    important: List[SkillItem]
    nice_to_have: List[SkillItem]
    implicit_traits: List[str]
    min_years_experience: float

class ResumeSkill(BaseModel):
    skill: str
    normalized_skill: str
    last_used_year: Optional[int] = None
    experience_years: Optional[float] = None

class ResumeSchema(BaseModel):
    candidate_name: str
    technical_skills: List[ResumeSkill]
    soft_skills: List[str]
    total_experience: float
    relevant_projects: List[str]




def llm_text(prompt: str, model: str = LLM_MODEL):
    """Standard response"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except:
        return ""

def extract_json_safe(text: str):
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return None

def llm_json(prompt: str):
    for _ in range(3):
        out = llm_text(prompt)
        js = extract_json_safe(out)
        if js:
            return js
    return None




def parse_with_llm(text: str, schema_class):
    schema_json = json.dumps(schema_class.model_json_schema(), indent=2)

    prompt = f"""
Extract STRICT JSON using this schema:
{schema_json}

Rules:
- No explanation, JSON only.
- Must-have=1.0, Important=0.8, Nice=0.6
- Normalize skills (AWS, React, Django...)
- Extract last_used_year & experience_years if available.

Input:
{text}
"""

    out = llm_json(prompt)
    if not out:
        return None

    try:
        return schema_class.model_validate(out)
    except Exception as e:
        print("Schema error:", e)
        return None





def norm(x: str) -> str:
    return " ".join(x.lower().split()) if x else ""

def chunk_text(text: str, size=300, overlap=40):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    out = []
    idx = 0
    while idx < len(words):
        out.append(" ".join(words[idx: idx + size]))
        idx += size - overlap
    return out

def extract_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([p.extract_text() or "" for p in reader.pages])




def make_doc_id(prefix: str, name: str):
    safe = "".join(c if c.isalnum() else "_" for c in name)[:30]
    return f"{prefix}_{safe}_{int(time.time())}"

def index_resume(resume: ResumeSchema, raw_text: str):
    doc_id = make_doc_id("resume", resume.candidate_name)
    chunks = chunk_text(raw_text)
    embeddings = embedder.encode(chunks)

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{doc_id}_{i}"],
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            metadatas=[{
                "doc_id": doc_id,
                "candidate_name": resume.candidate_name,
                "skills": ", ".join(s.normalized_skill for s in resume.technical_skills),
                "experience": resume.total_experience,
                "source": "resume"
            }]
        )
    return doc_id

def index_jd(jd: JobPostingSchema, raw_text: str):
    doc_id = make_doc_id("jd", jd.job_title)
    chunks = chunk_text(raw_text)
    embeddings = embedder.encode(chunks)

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{doc_id}_{i}"],
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            metadatas=[{
                "doc_id": doc_id,
                "job_title": jd.job_title,
                "source": "jd"
            }]
        )
    return doc_id




def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def sparse_score(q, t):
    q_tokens = q.lower().split()
    hits = sum(1 for w in q_tokens if w in t.lower())
    return hits / (len(q_tokens) + 1)

def fuzzy(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def hybrid_search(query: str, top_k=5):
    q_emb = embedder.encode([query])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=50,
        include=["documents", "metadatas", "embeddings"]
    )

    docs = []

    for text, meta, emb in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["embeddings"][0]
    ):
        emb_vec = np.array(emb)
        score = (
            0.45 * cosine(q_emb, emb_vec) +
            0.25 * sparse_score(query, text) +
            0.20 * fuzzy(query, text) +
            0.10
        )
        docs.append({
            "doc_id": meta["doc_id"],
            "text": text,
            "meta": meta,
            "score": round(score, 4)
        })

    # aggregate best per doc_id
    best = {}
    for d in docs:
        key = d["doc_id"]
        if key not in best or d["score"] > best[key]["score"]:
            best[key] = d

    return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]




def get_jd_groups(jd):
    return (
        [s.model_dump() for s in jd.must_have],
        [s.model_dump() for s in jd.important],
        [s.model_dump() for s in jd.nice_to_have]
    )

def resume_index(resume):
    out = {}
    for s in resume.technical_skills:
        sd = s.model_dump()
        out[norm(sd["normalized_skill"])] = sd
    return out

def score_candidate(jd: JobPostingSchema, resume: ResumeSchema):
    must, imp, nice = get_jd_groups(jd)
    r_ix = resume_index(resume)

    # SKILLS
    total_w = sum(x["weight"] for x in must + imp + nice)
    matched_w = 0
    missing_must = []

    for item in must + imp + nice:
        key = norm(item["skill"])
        if key in r_ix:
            matched_w += item["weight"]
        else:
            if item in must:
                missing_must.append(item["skill"])

    skill_raw = matched_w / total_w if total_w else 0
    if missing_must:
        skill_raw = max(0, skill_raw - 0.25)

    # EXPERIENCE
    exp_raw = min(1.0, resume.total_experience / jd.min_years_experience)

    # RECENCY
    years = [s.last_used_year for s in resume.technical_skills if s.last_used_year]
    if not years:
        rec_raw = 0.5
    else:
        age = NOW - max(years)
        rec_raw = 1.0 if age <= 2 else 0.6 if age <= 5 else 0.2

    # PROJECT RELEVANCE
    jd_tokens = set(norm(item["skill"]) for item in must + imp + nice)
    hits = sum(1 for p in resume.relevant_projects for t in jd_tokens if t in norm(p))
    proj_raw = hits / max(1, len(resume.relevant_projects))

    # SOFT SKILLS
    soft_raw = len(set(norm(t) for t in jd.implicit_traits)
                   & set(norm(s) for s in resume.soft_skills)) / max(1, len(jd.implicit_traits))

    final = (
        0.55 * skill_raw +
        0.20 * exp_raw +
        0.10 * rec_raw +
        0.10 * proj_raw +
        0.05 * soft_raw
    ) * 100

    return round(final, 2)




def rank_candidates(jd, resumes):
    scored = []
    for r in resumes:
        sc = score_candidate(jd, r)
        scored.append({
            "candidate_name": r.candidate_name,
            "final_score": sc
        })
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    for i, c in enumerate(scored, start=1):
        c["rank"] = i
    return scored




def explain_candidate(jd, resume, score, rank):
    prompt = f"""
Explain why this candidate received score {score} and rank {rank}.

Rules:
- Plain text only (NO markdown)
- Mention:
  1) Must-have match
  2) Important match
  3) Nice-to-have match
  4) Soft skills
  5) Experience
  6) Recency
  7) Project relevance
  8) Tier
  9) Ranking reason

JD:
{json.dumps(jd.model_dump(), indent=2)}

Resume:
{json.dumps(resume.model_dump(), indent=2)}
"""

    return llm_text(prompt)



__all__ = [
    "parse_with_llm",
    "index_resume",
    "index_jd",
    "hybrid_search",
    "score_candidate",
    "rank_candidates",
    "explain_candidate",
    "extract_pdf",
    "JobPostingSchema",
    "ResumeSchema",
]
