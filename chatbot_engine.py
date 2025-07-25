import chromadb
from sentence_transformers import SentenceTransformer
import pdfplumber, docx, json
from transformers import pipeline
from bs4 import BeautifulSoup
import pandas as pd

chroma_client = chromadb.Client()
model = SentenceTransformer("all-MiniLM-L6-v2")

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def extract_text(file_path):
    ext = file_path.lower()

    if ext.endswith(".pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                all_text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                print("📄 Extracted PDF Text:\n", all_text[:500]) 
                return all_text
        except Exception as e:
            print("❌ PDF Parse Error:", e)
            return ""

    elif ext.endswith('.docx'):
        try:
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print("DOCX parse error:", e)
            return ""

    elif ext.endswith('.json'):
        try:
            with open(file_path, encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=2)
        except Exception as e:
            print("JSON parse error:", e)
            return ""

    elif ext.endswith('.txt'):
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print("TXT parse error:", e)
            return ""

    elif ext.endswith('.html'):
        try:
            with open(file_path, encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                return soup.get_text(separator="\n")
        except Exception as e:
            print("HTML parse error:", e)
            return ""

    elif ext.endswith('.xlsx'):
        try:
            df = pd.read_excel(file_path)
            return df.to_string(index=False)
        except Exception as e:
            print("XLSX parse error:", e)
            return ""

    else:
        print("Unsupported file format:", ext)
        return ""

def process_file_and_embed(file_path, bot_id, prompt):
    print(f"\n📂 Processing file: {file_path} for bot: {bot_id}")

    text = extract_text(file_path)
    if not text.strip():
        print("❌ No text extracted from file.")
        return

    print(f"✅ Extracted text length: {len(text)} characters")

    # Chunking the text into pieces of 500 characters each
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    print(f"✅ Total chunks created: {len(chunks)}")

    # Embed and store each chunk in Chroma
    try:
        collection = chroma_client.get_or_create_collection(name=bot_id)

        for i, chunk in enumerate(chunks):
            embedding = model.encode([chunk])[0].tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{bot_id}_{i}"]
            )

        print(f"✅ Embedded and saved {len(chunks)} chunks into ChromaDB for bot: {bot_id}")

    except Exception as e:
        print("❌ Error during embedding or saving:", e)

def answer_from_bot(bot_id, query):
    collection = chroma_client.get_or_create_collection(bot_id)
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    
    # context = "\n".join(results['documents'][0])
    context = "\n".join(results['documents'][0])
    context = context[:2000]  

    print("Context Retrieved:", context)
    try:
        answer = qa_pipeline(question=query, context=context)
        print("Context Retrieved:", context)

        return answer['answer']
    except Exception as e:
        print("Error:", e)
        return "I'm sorry, I couldn't find a clear answer."

# http://localhost:5000/chat/fast_cart