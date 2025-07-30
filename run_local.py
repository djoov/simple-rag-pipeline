import os
import glob
import ollama
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Konfigurasi ---
DOCUMENT_SOURCE_FOLDER = "sample_data/source/"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3:mini"

def load_and_chunk_documents(folder_path):
    """Membaca semua dokumen dari folder dan memecahnya menggunakan LangChain."""
    print(f"🔍 Membaca dokumen dari: {folder_path}")
    
    document_files = glob.glob(os.path.join(folder_path, "*"))
    all_docs = []

    for doc_path in document_files:
        try:
            print(f"   - Memproses {os.path.basename(doc_path)}...")
            loader = PyPDFLoader(doc_path)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"   ⚠️ Gagal memproses {os.path.basename(doc_path)}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunked_docs = text_splitter.split_documents(all_docs)
    
    final_chunks = [doc.page_content for doc in chunked_docs]
            
    print(f"✅ Selesai, total {len(final_chunks)} potongan teks ditemukan.\n")
    return final_chunks

def get_embeddings(chunks, model_name):
    """Membuat embedding untuk setiap potongan teks menggunakan Ollama."""
    print(f"🧠 Membuat embedding menggunakan model '{model_name}'...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            response = ollama.embed(model=model_name, input=chunk)
            
            # Mengambil data dari kunci "embeddings" (jamak) sesuai output error.
            if "embeddings" in response:
                # Respons dari model ini adalah list di dalam list, jadi kita ambil elemen pertamanya.
                embeddings.append(response["embeddings"][0])
                # Menghapus print di dalam loop agar output lebih bersih
                # print(f"   - Embedding untuk potongan {i+1}/{len(chunks)} selesai.")
            else:
                print(f"   ⚠️ Ollama tidak mengembalikan 'embeddings' untuk potongan {i+1}. Respon: {response}")
        except Exception as e:
            print(f"   ⚠️ Gagal membuat embedding untuk potongan {i+1}: {e}")
            
    print("✅ Semua embedding berhasil dibuat.\n")
    return np.array(embeddings)

def find_most_relevant_chunks(query, document_embeddings, all_chunks, top_k=5):
    """Mencari potongan teks yang paling relevan dengan pertanyaan."""
    print("🔎 Mencari potongan yang paling relevan...")
    
    query_embedding_response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    
    # Mengambil data dari kunci "embeddings" (jamak) dan elemen pertamanya.
    query_embedding = np.array(query_embedding_response["embeddings"][0])
    
    # Menghitung kemiripan kosinus
    similarities = np.dot(document_embeddings, query_embedding) / (np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding))
    
    # Mengambil indeks dari potongan yang paling mirip
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_chunks = [all_chunks[i] for i in top_indices]
    print("✅ Potongan yang relevan ditemukan.\n")
    return relevant_chunks

def generate_answer(query, context_chunks, model_name):
    """Menghasilkan jawaban menggunakan LLM (gemma:2b) dengan konteks yang relevan."""
    print(f"✍️  Menghasilkan jawaban menggunakan model '{model_name}'...")
    
    context = "\n\n".join(context_chunks)
    
    system_prompt = "Anda adalah asisten AI yang menjawab pertanyaan hanya berdasarkan konteks yang diberikan. Jika jawaban tidak ada di dalam konteks, katakan 'Saya tidak dapat menemukan jawaban dari dokumen yang diberikan.'"
    user_prompt = f"Konteks:\n{context}\n\nPertanyaan: {query}\n\nJawaban:"
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        print("✅ Jawaban berhasil dibuat.\n")
        return response['message']['content']
    except Exception as e:
        print(f"⚠️ Gagal menghasilkan jawaban: {e}")
        return "Gagal saat mencoba menghasilkan jawaban dari model lokal."

# --- Alur Kerja Utama ---

if __name__ == "__main__":
    chunks = load_and_chunk_documents(DOCUMENT_SOURCE_FOLDER)
    
    if not chunks:
        print("Tidak ada dokumen yang bisa diproses. Program berhenti.")
    else:
        doc_embeddings = get_embeddings(chunks, EMBEDDING_MODEL)
        
        # Menambahkan pengecekan untuk memastikan embedding berhasil dibuat
        if doc_embeddings.size == 0:
            print("❌ Gagal membuat embedding. Pastikan Ollama berjalan dan model 'nomic-embed-text' sudah diunduh. Program berhenti.")
        else:
            while True:
                user_question = input("Masukkan pertanyaan Anda (atau ketik 'keluar' untuk berhenti): ")
                
                if user_question.lower() == 'keluar':
                    break
                
                if not user_question.strip():
                    continue

                relevant_context = find_most_relevant_chunks(user_question, doc_embeddings, chunks)
                
                answer = generate_answer(user_question, relevant_context, LLM_MODEL)
                
                print("--- Jawaban ---")
                print(answer)
                print("---------------\n")