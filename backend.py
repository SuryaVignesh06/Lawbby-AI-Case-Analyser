# Lawbby Backend API
# FastAPI server for legal AI assistant

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import hashlib
from datetime import datetime
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Lawbby API",
    description="Legal AI Research Assistant for Indian Law",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHROMA_PERSIST_DIR = "./vectorstore"
COLLECTION_NAME = "lawbby_legal_docs"

# System Prompts
GATEKEEPER_PROMPT = """Is this query related to Law, Legal matters, Rights, Constitution, Crime, or Civil Disputes? Answer YES or NO only.
Query: {query}"""

LEGAL_PROMPT = """You are Lawbby, a Legal Research Assistant for Indian Law. 
Be professional and authoritative. No casual language.

CONTEXT FROM DATABASE:
{context}

USER QUESTION: {query}

Provide a clear, structured legal answer. Cite sections and cases when available."""

REFUSAL_MESSAGE = "I am a Legal Research Assistant. My expertise is limited to Indian Law, Constitutional Rights, Criminal and Civil matters. Please ask a law-related question."

# Enhanced Case Analysis Prompt for Gemini
CASE_ANALYSIS_PROMPT = """You are an expert Indian legal researcher. Analyze this case description:

CASE DESCRIPTION:
{case_description}

CONTEXT FROM KNOWLEDGE BASE:
{context}

Provide a COMPREHENSIVE legal analysis. Return ONLY valid JSON in this exact format:
{{
    "applicable_sections": [
        {{"section": "302", "act": "IPC", "title": "Punishment for Murder", "relevance": "Direct application to facts", "key_points": "Essential elements and punishment"}}
    ],
    "relevant_cases": [
        {{"case_name": "State v. Accused", "year": "2020", "court": "Supreme Court", "relevance": "Similar facts", "key_holding": "Court's decision and ratio"}}
    ],
    "legal_strategies": ["Strategy 1 with specific legal basis", "Strategy 2"],
    "procedural_steps": ["Step 1: File complaint under...", "Step 2: ..."],
    "key_defenses": ["Defense 1 based on...", "Defense 2"],
    "risk_assessment": "Brief assessment of case strength"
}}

Be SPECIFIC to Indian law. Include actual IPC/CrPC sections. Return ONLY the JSON, no other text."""

# Request/Response Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    success: bool
    is_legal: bool
    response: str
    sources: list = []

class DocumentRequest(BaseModel):
    text: str
    doc_type: str = "statute"
    section: str = ""
    act: str = ""

class StatsResponse(BaseModel):
    total_documents: int
    collection_name: str

# RAG Engine Class
class LawbbyRAG:
    def __init__(self):
        self.setup_gemini()
        self.setup_chromadb()
        self.load_sample_data()
    
    def setup_gemini(self):
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.model = None
    
    def setup_chromadb(self):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        # Main legal documents collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Lawbby Legal Documents"}
        )
        
        # Query cache collection for storing previous analyses
        self.query_cache = self.client.get_or_create_collection(
            name="lawbby_query_cache",
            metadata={"description": "Cached Query-Response Pairs"}
        )
    
    def get_embedding(self, text: str) -> list:
        if not GEMINI_API_KEY:
            return [0.0] * 768
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    def is_legal_query(self, query: str) -> bool:
        if not self.model:
            return True
        try:
            prompt = GATEKEEPER_PROMPT.format(query=query)
            response = self.model.generate_content(prompt)
            return "YES" in response.text.upper()
        except:
            return True
    
    def search(self, query: str, n_results: int = 5) -> list:
        if self.collection.count() == 0:
            return []
        
        try:
            # Try embedding-based search first
            if GEMINI_API_KEY:
                query_embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=query,
                    task_type="retrieval_query"
                )['embedding']
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
            else:
                # Fall back to text query
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
        except Exception as e:
            print(f"Embedding search failed: {e}, falling back to text search")
            # Fallback to text-based search using ChromaDB's built-in
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
            except Exception as e2:
                print(f"Text search also failed: {e2}")
                # Ultimate fallback - get all docs and filter manually
                all_docs = self.collection.get()
                query_lower = query.lower()
                docs = []
                if all_docs and all_docs['documents']:
                    for i, doc in enumerate(all_docs['documents']):
                        # Simple keyword matching
                        if any(word in doc.lower() for word in query_lower.split()):
                            docs.append({
                                "text": doc,
                                "metadata": all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                            })
                return docs[:n_results]
        
        docs = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                docs.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                })
        return docs
    
    def generate_response(self, query: str, docs: list) -> str:
        if not self.model:
            return "API key not configured. Please add GEMINI_API_KEY to .env file."
        
        context = "\n\n---\n\n".join([d["text"] for d in docs]) if docs else "No documents found."
        prompt = LEGAL_PROMPT.format(context=context, query=query)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def add_document(self, text: str, metadata: dict) -> str:
        doc_id = f"doc_{self.collection.count()}"
        try:
            embedding = self.get_embedding(text)
        except Exception as e:
            print(f"Embedding generation failed: {e}, using default embedding")
            embedding = [0.0] * 768  # Default embedding for fallback
        
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id
    
    def get_query_hash(self, query: str) -> str:
        """Generate a unique hash for a query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:16]
    
    def find_similar_query(self, query: str, threshold: float = 0.80) -> dict | None:
        """Find if a similar query was already analyzed (semantic cache lookup)"""
        if self.query_cache.count() == 0:
            return None
        
        try:
            # Generate embedding for the query
            if GEMINI_API_KEY:
                query_embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=query,
                    task_type="retrieval_query"
                )['embedding']
                
                results = self.query_cache.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                results = self.query_cache.query(
                    query_texts=[query],
                    n_results=1,
                    include=["documents", "metadatas", "distances"]
                )
            
            if results['documents'] and results['documents'][0]:
                # Check similarity (distance < threshold means similar)
                # ChromaDB uses L2 distance, lower is more similar
                distance = results['distances'][0][0] if results['distances'] else 1.0
                similarity = 1 / (1 + distance)  # Convert to similarity score
                
                print(f"Cache lookup - Distance: {distance:.3f}, Similarity: {similarity:.3f}")
                
                if similarity >= threshold:
                    metadata = results['metadatas'][0][0] if results['metadatas'] else {}
                    cached_response = metadata.get("response_json", "{}")
                    
                    # Increment hit count
                    try:
                        hit_count = metadata.get("hit_count", 0) + 1
                        doc_id = metadata.get("query_hash", "")
                        if doc_id:
                            self.query_cache.update(
                                ids=[f"query_{doc_id}"],
                                metadatas=[{**metadata, "hit_count": hit_count}]
                            )
                    except:
                        pass
                    
                    return {
                        "cached": True,
                        "similarity": similarity,
                        "original_query": results['documents'][0][0],
                        "response": json.loads(cached_response) if cached_response else {},
                        "hit_count": metadata.get("hit_count", 0)
                    }
        except Exception as e:
            print(f"Cache lookup failed: {e}")
        
        return None
    
    def generate_dynamic_analysis(self, case_description: str, context_docs: list) -> dict:
        """Generate comprehensive legal analysis using Gemini with RAG context"""
        if not self.model:
            return self._get_fallback_analysis(case_description)
        
        # Build context from retrieved documents
        context = "\n\n---\n\n".join([d["text"][:1500] for d in context_docs[:5]]) if context_docs else "No prior context available."
        
        prompt = CASE_ANALYSIS_PROMPT.format(
            case_description=case_description,
            context=context
        )
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up JSON response (remove markdown code blocks if present)
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            # Parse JSON response
            analysis = json.loads(response_text)
            
            # Ensure all expected fields exist
            analysis.setdefault("applicable_sections", [])
            analysis.setdefault("relevant_cases", [])
            analysis.setdefault("legal_strategies", [])
            analysis.setdefault("procedural_steps", [])
            analysis.setdefault("key_defenses", [])
            analysis.setdefault("risk_assessment", "Analysis pending additional review")
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return self._get_fallback_analysis(case_description)
        except Exception as e:
            print(f"Analysis generation error: {e}")
            return self._get_fallback_analysis(case_description)
    
    def _get_fallback_analysis(self, case_description: str) -> dict:
        """Fallback analysis when Gemini is unavailable"""
        # Use keyword matching to find relevant sections from database
        docs = self.search(case_description, n_results=10)
        
        applicable_sections = []
        relevant_cases = []
        
        for doc in docs:
            meta = doc.get("metadata", {})
            if meta.get("type") == "statute":
                text = doc["text"]
                lines = text.split("\n")
                title = lines[0].split(" - ")[1] if " - " in lines[0] else "Legal Section"
                applicable_sections.append({
                    "section": meta.get("section", ""),
                    "act": meta.get("act", ""),
                    "title": title,
                    "relevance": "Keyword match from database",
                    "key_points": lines[2] if len(lines) > 2 else ""
                })
            elif meta.get("type") == "case_law":
                text = doc["text"]
                lines = text.split("\n")
                case_name = lines[0].replace("LANDMARK CASE: ", "").split(" (")[0]
                relevant_cases.append({
                    "case_name": case_name,
                    "year": meta.get("year", "Unknown"),
                    "court": meta.get("court", "Supreme Court"),
                    "relevance": "Similar subject matter",
                    "key_holding": "See full case details"
                })
        
        return {
            "applicable_sections": applicable_sections[:5],
            "relevant_cases": relevant_cases[:3],
            "legal_strategies": [
                "Review applicable IPC/CrPC sections",
                "Analyze similar case precedents",
                "Consider available legal defenses",
                "Consult with legal counsel for detailed strategy"
            ],
            "procedural_steps": [
                "File FIR if criminal matter",
                "Gather documentary evidence",
                "Identify witnesses",
                "Engage legal representation"
            ],
            "key_defenses": [
                "Examine facts for available defenses",
                "Check for procedural violations"
            ],
            "risk_assessment": "Detailed analysis requires AI assistance - currently using cached knowledge only"
        }
    
    def store_analysis(self, query: str, analysis: dict) -> str:
        """Store query-response pair in cache for future retrieval (learning)"""
        query_hash = self.get_query_hash(query)
        doc_id = f"query_{query_hash}"
        
        # Check if already exists
        try:
            existing = self.query_cache.get(ids=[doc_id])
            if existing and existing['documents']:
                # Update hit count
                metadata = existing['metadatas'][0] if existing['metadatas'] else {}
                hit_count = metadata.get("hit_count", 0) + 1
                self.query_cache.update(
                    ids=[doc_id],
                    metadatas=[{**metadata, "hit_count": hit_count}]
                )
                return doc_id
        except:
            pass
        
        # Store new analysis
        try:
            if GEMINI_API_KEY:
                embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=query,
                    task_type="retrieval_document"
                )['embedding']
            else:
                embedding = [0.0] * 768
        except Exception as e:
            print(f"Embedding failed for cache: {e}")
            embedding = [0.0] * 768
        
        metadata = {
            "query_hash": query_hash,
            "timestamp": datetime.now().isoformat(),
            "response_json": json.dumps(analysis),
            "hit_count": 1,
            "type": "case_analysis"
        }
        
        try:
            self.query_cache.add(
                documents=[query],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
            )
            print(f"Stored analysis in cache: {doc_id}")
        except Exception as e:
            print(f"Failed to store in cache: {e}")
        
        return doc_id
    
    def get_enhanced_stats(self) -> dict:
        """Get stats including query cache"""
        return {
            "total_documents": self.collection.count(),
            "cached_queries": self.query_cache.count(),
            "collection_name": COLLECTION_NAME
        }
    
    def get_stats(self) -> dict:
        return self.get_enhanced_stats()
    
    def clear_all(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
    
    def load_sample_data(self):
        if self.collection.count() > 0:
            return
        
        sample_docs = [
            {"id": "ipc_302", "text": "Section 302 - Indian Penal Code (IPC)\nPUNISHMENT FOR MURDER\n\nWhoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.\n\nKey Elements:\n1. Intention to cause death\n2. Intention to cause bodily injury likely to cause death\n3. Knowledge that the act is likely to cause death", "metadata": {"type": "statute", "section": "302", "act": "IPC"}},
            {"id": "ipc_304", "text": "Section 304 - Indian Penal Code (IPC)\nCULPABLE HOMICIDE NOT AMOUNTING TO MURDER\n\nPart I: If done with intention of causing death or bodily injury likely to cause death - imprisonment for life, or up to 10 years, and fine.\nPart II: If done with knowledge that it is likely to cause death - imprisonment up to 10 years, or fine, or both.", "metadata": {"type": "statute", "section": "304", "act": "IPC"}},
            {"id": "ipc_420", "text": "Section 420 - Indian Penal Code (IPC)\nCHEATING AND DISHONESTLY INDUCING DELIVERY OF PROPERTY\n\nWhoever cheats and thereby dishonestly induces delivery of property shall be punished with imprisonment up to 7 years, and fine.\n\nEssentials:\n1. Deception\n2. Fraudulent inducement\n3. Delivery of property\n4. Dishonest intention", "metadata": {"type": "statute", "section": "420", "act": "IPC"}},
            {"id": "ipc_498a", "text": "Section 498A - Indian Penal Code (IPC)\nCRUELTY BY HUSBAND OR RELATIVES\n\nWhoever subjects a woman to cruelty shall be punished with imprisonment up to 3 years and fine.\n\nCruelty includes:\n(a) Conduct likely to drive the woman to suicide\n(b) Harassment for dowry demands\n\nThis is a cognizable and non-bailable offence.", "metadata": {"type": "statute", "section": "498A", "act": "IPC"}},
            {"id": "ipc_379", "text": "Section 379 - Indian Penal Code (IPC)\nPUNISHMENT FOR THEFT\n\nWhoever commits theft shall be punished with imprisonment up to 3 years, or fine, or both.\n\nEssentials of Theft (Section 378):\n1. Dishonest intention\n2. Movable property\n3. Without consent\n4. Movement of property", "metadata": {"type": "statute", "section": "379", "act": "IPC"}},
            {"id": "case_bachan", "text": """LANDMARK CASE: Bachan Singh v. State of Punjab (1980)
Supreme Court of India | 5-Judge Constitution Bench
Citation: AIR 1980 SC 898 | (1980) 2 SCC 684

FACTS OF THE CASE:
The appellant Bachan Singh was convicted for the murder of his wife, son, and daughter-in-law. He appealed against the death sentence arguing that the death penalty under Section 302 IPC was unconstitutional.

ISSUES BEFORE THE COURT:
1. Whether death penalty under IPC Section 302 violates Article 21 (Right to Life)?
2. What guidelines should govern imposition of death sentence?

SECTIONS AND LAWS APPLIED:
- Section 302 IPC (Punishment for Murder)
- Section 354(3) CrPC (Special reasons for death sentence)
- Article 21 of Constitution (Right to Life)
- Article 14 of Constitution (Right to Equality)

ARGUMENTS THAT SUCCEEDED:
1. Death penalty serves legitimate penological goals - deterrence, prevention, reformation
2. Parliament's policy choice to retain death penalty reflects collective wisdom
3. Adequate procedural safeguards exist through judicial review
4. 'Rarest of rare' doctrine balances retribution with reformation

HELD (4:1 Majority):
Death penalty is constitutional but must be imposed ONLY in the 'RAREST OF RARE' cases where the alternative option of life imprisonment is unquestionably foreclosed.

KEY PRECEDENT VALUE:
- Established the 'Rarest of Rare' doctrine
- Courts must give special reasons for death sentence
- Aggravating and mitigating circumstances must be weighed
- Life imprisonment is the rule; death is the exception

OUTCOME: Death penalty upheld as constitutional with guidelines.""", "metadata": {"type": "case_law", "case_name": "Bachan Singh v. State of Punjab", "year": "1980", "court": "Supreme Court", "citation": "AIR 1980 SC 898"}},

            {"id": "case_vishaka", "text": """LANDMARK CASE: Vishaka v. State of Rajasthan (1997)
Supreme Court of India | 3-Judge Bench
Citation: AIR 1997 SC 3011 | (1997) 6 SCC 241

FACTS OF THE CASE:
Bhanwari Devi, a social worker in Rajasthan, was gang-raped by upper-caste men for preventing a child marriage. The trial court acquitted the accused. A PIL was filed by Vishaka and other women's groups.

ISSUES BEFORE THE COURT:
1. What constitutes sexual harassment at workplace?
2. What remedies exist in absence of legislation?
3. Can international conventions fill legislative vacuum?

SECTIONS AND LAWS APPLIED:
- Article 14 (Equality before law)
- Article 15 (Prohibition of discrimination)
- Article 19(1)(g) (Right to practice profession)
- Article 21 (Right to life and dignity)
- Article 32 (Writ jurisdiction)
- CEDAW (UN Convention on Women's Rights)

ARGUMENTS THAT SUCCEEDED:
1. Right to work with dignity is a fundamental right under Article 21
2. Sexual harassment violates fundamental rights of women
3. In absence of domestic law, international conventions can be relied upon
4. Courts can lay down guidelines to fill legislative vacuum

HELD (Unanimously):
The Court laid down comprehensive guidelines (Vishaka Guidelines) for prevention of sexual harassment at workplace, binding on all employers until legislation is enacted.

VISHAKA GUIDELINES INCLUDE:
1. Definition of sexual harassment
2. Duty of employer to prevent harassment
3. Complaints Committee mechanism
4. Third-party harassment coverage
5. Criminal proceedings where necessary

KEY PRECEDENT VALUE:
- Judicial activism to protect women's rights
- International law can supplement domestic law
- Led to Sexual Harassment of Women at Workplace Act, 2013 (POSH Act)

OUTCOME: Landmark guidelines established; led to POSH Act 2013.""", "metadata": {"type": "case_law", "case_name": "Vishaka v. State of Rajasthan", "year": "1997", "court": "Supreme Court", "citation": "AIR 1997 SC 3011"}},

            {"id": "case_kesav", "text": """LANDMARK CASE: Kesavananda Bharati v. State of Kerala (1973)
Supreme Court of India | 13-Judge Bench (Largest Ever)
Citation: AIR 1973 SC 1461 | (1973) 4 SCC 225

FACTS OF THE CASE:
Kesavananda Bharati, head of a religious mutt in Kerala, challenged the Kerala Land Reforms Act which acquired his mutt's property. The case became a vehicle to examine Parliament's power to amend the Constitution.

ISSUES BEFORE THE COURT:
1. Is Parliament's power to amend the Constitution unlimited?
2. Can Parliament alter fundamental rights completely?
3. What are the limits of Article 368?

SECTIONS AND LAWS APPLIED:
- Article 368 (Power to amend Constitution)
- Article 13 (Laws inconsistent with fundamental rights)
- Article 31 (Right to Property - then a fundamental right)
- Part III (Fundamental Rights)
- Kerala Land Reforms Act, 1963

ARGUMENTS THAT SUCCEEDED:
1. Constitution has certain basic features that cannot be destroyed
2. Amending power is different from constituent power
3. 'Amendment' cannot mean 'destruction'
4. Fundamental rights are part of basic structure

HELD (7:6 Majority):
Parliament CAN amend any part of the Constitution including Fundamental Rights, BUT cannot alter the 'BASIC STRUCTURE' of the Constitution.

BASIC STRUCTURE INCLUDES:
1. Supremacy of the Constitution
2. Republican and democratic form of government
3. Secular character of the Constitution
4. Separation of powers
5. Federal character
6. Sovereignty and integrity of India
7. Judicial review
8. Rule of law

KEY PRECEDENT VALUE:
- Basic Structure Doctrine - cornerstone of constitutional law
- Limits parliamentary supremacy
- Protects core constitutional values
- Applied to strike down 39th Amendment, 42nd Amendment (parts), NJAC Act

OUTCOME: Property acquisition upheld; Basic Structure Doctrine established.""", "metadata": {"type": "case_law", "case_name": "Kesavananda Bharati v. State of Kerala", "year": "1973", "court": "Supreme Court", "citation": "AIR 1973 SC 1461"}},
            {"id": "contract_10", "text": "Section 10 - Indian Contract Act, 1872\nWHAT AGREEMENTS ARE CONTRACTS\n\nAll agreements are contracts if made by:\n1. Free consent of parties\n2. Parties competent to contract\n3. Lawful consideration\n4. Lawful object\n5. Not expressly declared void", "metadata": {"type": "statute", "section": "10", "act": "Contract Act"}},
            {"id": "crpc_154", "text": "Section 154 - Code of Criminal Procedure (CrPC)\nFIRST INFORMATION REPORT (FIR)\n\nEvery information of cognizable offence given orally shall be reduced to writing and signed by informant. Copy must be given free of cost.\n\nNote: Zero FIR can be filed at any police station. Police cannot refuse to register FIR.", "metadata": {"type": "statute", "section": "154", "act": "CrPC"}}
        ]
        
        for doc in sample_docs:
            self.add_document(doc["text"], doc["metadata"])

# Initialize RAG
rag = LawbbyRAG()

# API Endpoints
@app.get("/")
def root():
    return {"message": "Lawbby Legal AI API", "status": "running"}

@app.get("/api/stats", response_model=StatsResponse)
def get_stats():
    return rag.get_stats()

@app.get("/api/sections")
def get_sections():
    """Get list of all available sections"""
    sections_data = []
    
    try:
        # Get all documents from ChromaDB collection
        all_docs = rag.collection.get()
        
        if all_docs and all_docs['documents']:
            for i, doc_text in enumerate(all_docs['documents']):
                metadata = all_docs['metadatas'][i] if all_docs['metadatas'] and i < len(all_docs['metadatas']) else {}
                
                # Only include statute-type documents
                if metadata.get("type") == "statute":
                    # Parse title from document text
                    lines = doc_text.split("\n")
                    title = "Legal Section"
                    if lines and " - " in lines[0]:
                        title = lines[0].split(" - ", 1)[1] if len(lines[0].split(" - ", 1)) > 1 else "Legal Section"
                    
                    # Parse summary (first paragraph after title)
                    summary = ""
                    paragraphs = doc_text.split("\n\n")
                    if len(paragraphs) > 0:
                        # Get first line after the header
                        first_para_lines = paragraphs[0].split("\n")
                        if len(first_para_lines) >= 3:
                            summary = first_para_lines[2]
                        elif len(paragraphs) > 1:
                            summary = paragraphs[1][:200]
                    
                    sections_data.append({
                        "section": metadata.get("section", ""),
                        "act": metadata.get("act", ""),
                        "title": title.strip(),
                        "summary": summary.strip(),
                        "text": doc_text
                    })
        
        # Sort by section number
        sections_data.sort(key=lambda x: (x.get("act", ""), x.get("section", "")))
        
    except Exception as e:
        print(f"Error fetching sections: {e}")
        return {"sections": [], "error": str(e)}
    
    return {"sections": sections_data}

@app.post("/api/analyze-case")
def analyze_case(request: dict):
    """Enhanced RAG: Analyze case with semantic cache + dynamic Gemini analysis"""
    case_description = request.get("case_description", "").strip()
    
    if not case_description or len(case_description) < 10:
        raise HTTPException(status_code=400, detail="Case description too short")
    
    # Check if legal query
    is_legal = rag.is_legal_query(case_description)
    
    if not is_legal:
        return {
            "applicable_sections": [],
            "past_cases": [],
            "legal_strategies": [],
            "message": REFUSAL_MESSAGE,
            "source": "rejected"
        }
    
    # STEP 1: Check semantic cache for similar queries
    cached_result = rag.find_similar_query(case_description, threshold=0.75)
    
    if cached_result and cached_result.get("cached"):
        print(f"Cache HIT! Similarity: {cached_result.get('similarity', 0):.2f}")
        cached_response = cached_result.get("response", {})
        
        # Transform cached response to frontend format
        applicable_sections = []
        for section in cached_response.get("applicable_sections", []):
            applicable_sections.append({
                "section": section.get("section", ""),
                "act": section.get("act", ""),
                "title": section.get("title", "Legal Section"),
                "relevance": section.get("relevance", "From cached analysis"),
                "text": section.get("key_points", "See detailed analysis")
            })
        
        past_cases = []
        for case in cached_response.get("relevant_cases", []):
            past_cases.append({
                "case_name": case.get("case_name", ""),
                "year": case.get("year", "Unknown"),
                "similarity": case.get("relevance", "Similar to previous query"),
                "facts": "From cached similar analysis",
                "sections_used": ["See cached details"],
                "legal_approach": case.get("key_holding", ""),
                "outcome": "See full case",
                "ratio": case.get("key_holding", "")
            })
        
        return {
            "applicable_sections": applicable_sections,
            "past_cases": past_cases,
            "legal_strategies": cached_response.get("legal_strategies", []),
            "procedural_steps": cached_response.get("procedural_steps", []),
            "key_defenses": cached_response.get("key_defenses", []),
            "risk_assessment": cached_response.get("risk_assessment", ""),
            "source": "cache",
            "cache_similarity": cached_result.get("similarity", 0),
            "hit_count": cached_result.get("hit_count", 0)
        }
    
    # STEP 2: No cache hit - search knowledge base for context
    print("Cache MISS - generating fresh analysis")
    context_docs = rag.search(case_description, n_results=10)
    
    # STEP 3: Generate dynamic analysis using Gemini with RAG context
    analysis = rag.generate_dynamic_analysis(case_description, context_docs)
    
    # STEP 4: Store the new analysis in cache for future queries (learning)
    rag.store_analysis(case_description, analysis)
    
    # STEP 5: Transform to frontend format
    applicable_sections = []
    for section in analysis.get("applicable_sections", []):
        applicable_sections.append({
            "section": section.get("section", ""),
            "act": section.get("act", ""),
            "title": section.get("title", "Legal Section"),
            "relevance": section.get("relevance", "AI-generated analysis"),
            "text": section.get("key_points", "")
        })
    
    # Also include matching docs from knowledge base
    for doc in context_docs:
        meta = doc.get("metadata", {})
        if meta.get("type") == "statute":
            text = doc["text"]
            lines = text.split("\n")
            title = lines[0].split(" - ")[1] if " - " in lines[0] else "Legal Section"
            # Check if not already in list
            existing_sections = [s.get("section") for s in applicable_sections]
            if meta.get("section") not in existing_sections:
                applicable_sections.append({
                    "section": meta.get("section", ""),
                    "act": meta.get("act", ""),
                    "title": title,
                    "relevance": "From knowledge base",
                    "text": text
                })
    
    past_cases = []
    for case in analysis.get("relevant_cases", []):
        past_cases.append({
            "case_name": case.get("case_name", ""),
            "year": case.get("year", "Unknown"),
            "similarity": case.get("relevance", "AI-identified relevance"),
            "facts": "AI-analyzed facts",
            "sections_used": ["See analysis"],
            "legal_approach": case.get("key_holding", ""),
            "outcome": "See full case",
            "ratio": case.get("key_holding", "")
        })
    
    # Also include matching case law from knowledge base
    for doc in context_docs:
        meta = doc.get("metadata", {})
        if meta.get("type") == "case_law":
            text = doc["text"]
            lines = text.split("\n")
            case_name = lines[0].replace("LANDMARK CASE: ", "").split(" (")[0]
            # Check if not already in list
            existing_cases = [c.get("case_name") for c in past_cases]
            if case_name not in existing_cases:
                past_cases.append({
                    "case_name": case_name,
                    "year": meta.get("year", "Unknown"),
                    "similarity": "From knowledge base",
                    "facts": "See case details",
                    "sections_used": ["Cited in judgment"],
                    "legal_approach": "Legal strategy in judgment",
                    "outcome": "See details",
                    "ratio": text
                })
    
    return {
        "applicable_sections": applicable_sections[:8],  # Limit to 8
        "past_cases": past_cases[:5],  # Limit to 5
        "legal_strategies": analysis.get("legal_strategies", []),
        "procedural_steps": analysis.get("procedural_steps", []),
        "key_defenses": analysis.get("key_defenses", []),
        "risk_assessment": analysis.get("risk_assessment", ""),
        "source": "fresh",
        "knowledge_base_docs": len(context_docs)
    }

@app.post("/api/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    query = request.query.strip()
    
    if not query or len(query) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    
    # Check if legal query
    is_legal = rag.is_legal_query(query)
    
    if not is_legal:
        return QueryResponse(
            success=True,
            is_legal=False,
            response=REFUSAL_MESSAGE,
            sources=[]
        )
    
    # Search and generate
    docs = rag.search(query, n_results=5)
    response = rag.generate_response(query, docs)
    
    sources = []
    for doc in docs:
        meta = doc.get("metadata", {})
        sources.append({
            "type": meta.get("type", "document"),
            "section": meta.get("section", ""),
            "act": meta.get("act", meta.get("case_name", "")),
            "text": doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"]
        })
    
    return QueryResponse(
        success=True,
        is_legal=True,
        response=response,
        sources=sources
    )

@app.post("/api/documents")
def add_document(request: DocumentRequest):
    metadata = {"type": request.doc_type}
    if request.section:
        metadata["section"] = request.section
    if request.act:
        metadata["act"] = request.act
    
    doc_id = rag.add_document(request.text, metadata)
    return {"success": True, "doc_id": doc_id}

@app.delete("/api/documents")
def clear_documents():
    rag.clear_all()
    rag.load_sample_data()
    return {"success": True, "message": "Database cleared and reloaded with samples"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
