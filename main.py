import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- 1. Define a collection of documents (sentences) ---
documents = [
    "Kedi ve köpekler evcil hayvanlardır.",
    "Büyük kediler ormanda yaşar.",
    "Yavru kedi çok sevimliydi.",
    "Araba hızlı bir şekilde ilerliyordu.",
    "Bisiklet sürmek sağlıklı bir aktivitedir.",
    "Kedi maması almak için markete gittim."
]

# --- 2. Define a query ---
query = "Evcil hayvanlarla ilgili bir şeyler arıyorum."

print("--- Vektör Tabanlı Anlamsal Arama Örneği ---")
print(f"Dokümanlar: {len(documents)} adet")
print(f"Sorgu: '{query}'\n")

# --- 3. Load a pre-trained Sentence Transformer model ---
# This model converts text into high-dimensional numerical vectors (embeddings).
# 'all-MiniLM-L6-v2' is a good balance of speed and accuracy for many tasks.
print("Model yükleniyor...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model yüklendi.\n")

# --- 4. Generate embeddings for documents and the query ---
# Each sentence is transformed into a vector that captures its semantic meaning.
# This is the core 'embedding' process discussed in the article.
print("Embeddings oluşturuluyor...")
document_embeddings = model.encode(documents, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)
print("Embeddings oluşturuldu.\n")

# --- 5. Calculate cosine similarity between the query and all documents ---
# Cosine similarity measures the angle between two vectors.
# A higher similarity score (closer to 1) means the vectors are more aligned,
# indicating greater semantic similarity between the texts. This demonstrates
# the 'similarity' concept central to vector databases.
cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]

# --- 6. Sort documents by similarity score ---
# We pair each document with its score and sort them in descending order.
results = []
for i, score in enumerate(cosine_scores):
    results.append({'document': documents[i], 'score': score.item()})

results = sorted(results, key=lambda x: x['score'], reverse=True)

# --- 7. Print the results ---
print("Sorguya en benzer dokümanlar (anlamsal sıralama):")
for res in results:
    # The score indicates how semantically close the document is to the query.
    print(f"- Puan: {res['score']:.4f} | Doküman: '{res['document']}'")

print("\n--- Açıklama ---")
print("Bu örnek, metinleri anlamsal vektörlere dönüştürerek (embedding),")
print("geleneksel anahtar kelime eşleşmesinin ötesinde, sorgunun niyetine")
print("en uygun dokümanları nasıl bulabileceğimizi göstermektedir.")
print("Vektör veritabanları bu tür işlemleri büyük ölçekte ve verimli bir şekilde yapar.")
