# from pythainlp.corpus.common import thai_stopwords
# from pythainlp.tokenize import word_tokenize
# from fuzzywuzzy import fuzz

# # ข้อความที่ต้องการตรวจสอบ
# text = "ฉันอยากได้ค่าจาก ystat ช่วยหาให้หน่อยได้ไหม"

# # คำที่ต้องการเปรียบเทียบ
# second_brain = {'products': ['y-stat', 'graph', 'ภาพ', 'image']}

# # ใช้ pythainlp แบ่งคำในภาษาไทย
# words = word_tokenize(text, engine='newmm')

# # ลบ stopwords ที่มาจาก pythainlp
# filtered_words = [word for word in words if word not in thai_stopwords()]

# # รวมคำหลังจากลบ stopwords
# filtered_text = ' '.join(filtered_words)

# # ฟังก์ชันเปรียบเทียบความคล้ายคลึงกัน
# def find_similarity(text, keywords):
#     similarities = [(keyword, fuzz.ratio(text.lower(), keyword.lower())) for keyword in keywords]
#     return similarities

# # เรียกใช้ฟังก์ชันและแสดงผลลัพธ์
# similarity_results = find_similarity(filtered_text, second_brain['products'])

# print("ความคล้ายคลึงกันของแต่ละคำในรายการ:")
# for keyword, similarity in similarity_results:
#     print(f"{keyword}: {similarity}%")
    




# from transformers import AutoTokenizer, AutoModel
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # โหลด tokenizer และ model สำหรับ LaBSE
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
# model = AutoModel.from_pretrained('sentence-transformers/LaBSE')

# # ฟังก์ชันสำหรับการแปลงประโยคเป็นเวคเตอร์
# def get_sentence_embedding(sentence):
#     inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         embeddings = model(**inputs).pooler_output
#     return embeddings

# # ประโยคตัวอย่าง (ภาษาไทยและภาษาอังกฤษ)
# sentence1 = "สวัสดีครับ"
# sentence2 = "Hello"

# # แปลงประโยคเป็นเวคเตอร์
# embedding1 = get_sentence_embedding(sentence1)
# embedding2 = get_sentence_embedding(sentence2)

# # คำนวณความคล้ายคลึงโดยใช้ cosine similarity
# cosine_sim = cosine_similarity(embedding1.numpy(), embedding2.numpy())

# print(f"Cosine similarity: {cosine_sim[0][0]}")




from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# โหลด tokenizer และ model สำหรับ LaBSE
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
model = AutoModel.from_pretrained('sentence-transformers/LaBSE')

# ฟังก์ชันสำหรับการแปลงประโยคเป็นเวคเตอร์
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    return embeddings

# ประโยคตัวอย่าง (ภาษาไทยและภาษาอังกฤษ)
sentence1 = "ฉันอยากได้รูปแสดงกราฟ"

# ข้อมูลใน second_brain
second_brain = {'products': ['y-stat', 'graph', 'ภาพ', 'image']}

# แปลงประโยค sentence1 เป็นเวคเตอร์
embedding1 = get_sentence_embedding(sentence1).numpy()

# คำนวณความคล้ายคลึงกับทุกประโยคใน second_brain['products']
similarities = {}
for product in second_brain['products']:
    embedding_product = get_sentence_embedding(product).numpy()
    cosine_sim = cosine_similarity(embedding1, embedding_product)
    similarities[product] = cosine_sim[0][0]

# แสดงผลลัพธ์
for product, sim in similarities.items():
    print(f"Cosine similarity between '{sentence1}' and '{product}': {sim}")
