# from pythainlp.corpus.common import thai_stopwords
# from pythainlp.tokenize import word_tokenize
# from fuzzywuzzy import fuzz

# # ข้อความที่ต้องการตรวจสอบ
# text = "ฉันอยากได้ค่าจาก ystat ช่วยหาให้หน่อยได้ไหม"

# # คำที่ต้องการเปรียบเทียบ
# second_brain = {'products': ['y-stat', 'graph', 'ภาพ', 'image', 'วัน']}

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
# sentence1 = "ฉันอยากได้รูปแสดงกราฟ"

# # ข้อมูลใน second_brain
# second_brain = {'products': ['y-stat', 'graph', 'ภาพ', 'image']}

# # แปลงประโยค sentence1 เป็นเวคเตอร์
# embedding1 = get_sentence_embedding(sentence1).numpy()

# # คำนวณความคล้ายคลึงกับทุกประโยคใน second_brain['products']
# similarities = {}
# for product in second_brain['products']:
#     embedding_product = get_sentence_embedding(product).numpy()
#     cosine_sim = cosine_similarity(embedding1, embedding_product)
#     similarities[product] = cosine_sim[0][0]

# # แสดงผลลัพธ์
# for product, sim in similarities.items():
#     print(f"Cosine similarity between '{sentence1}' and '{product}': {sim}")



from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize
from fuzzywuzzy import fuzz

# Synonym dictionary
synonyms = {
    'รูปภาพ': ['ภาพ', 'image', 'รูป', 'ภาพถ่าย'],
    'วันที่': ['วัน', 'วันที่', 'date'],
    'กราฟ': ['graph', 'Graph', 'กราฟ'],
    # Add more synonyms as needed
}

# Known stock names and symbols
stock_names_symbols = {
    'AMD': ['Advanced Micro Devices', 'AMD'],
    'INTC': ['Intel', 'INTC'],
    'AAPL': ['Apple', 'AAPL'],
    # Add more stock names and symbols as needed
}


# Initialize second_brain and input1
second_brain = {'products': ['y-stat', 'รูปภาพ', 'กราฟ', 'วันที่']}
input1 = 'ฉันอยากได้รูปภาพแสดง graph ค่าy AMD ของวันที่ 20 เดือน7 2023'
input2 = 'ขอรูปภาพค่า y วันที่ 21 เดือน 7 2023 AMD'
input3 = 'ขอรูปภาพ Teerawat ทำท่าตัว y'
input4 = 'ขอกราฟ AMD วันนี้'
input5 = 'ขอภาพรถ 7 คัน ที่เรียงกันเป็นรูปตัว y ที่หน้าบ. AMD ในปี 2023'

# Tokenize words and remove stopwords
stopwords = thai_stopwords()
# input1_tokens = [word for word in word_tokenize(input1) if word not in stopwords]
# input1_tokens = [word for word in word_tokenize(input2) if word not in stopwords]
# input1_tokens = [word for word in word_tokenize(input3) if word not in stopwords]
# input1_tokens = [word for word in word_tokenize(input4) if word not in stopwords]
input1_tokens = [word for word in word_tokenize(input5) if word not in stopwords]
products_tokens = [word for word in second_brain['products'] if word not in stopwords]

# Expand products_tokens with synonyms
expanded_products_tokens = products_tokens[:]
for word in products_tokens:
    if word in synonyms:
        expanded_products_tokens.extend(synonyms[word])

# Check for stock names and symbols
def find_stocks_in_input(input_tokens, stock_dict):
    found_stocks = set()
    for token in input_tokens:
        for stock, names_symbols in stock_dict.items():
            if token in names_symbols:
                found_stocks.add(stock)
    return found_stocks

# Find stocks in the input
found_stocks = find_stocks_in_input(input1_tokens, stock_names_symbols)

# Calculate the percentage of matching words, considering synonyms and fuzzy matching
matching_words = set()
for word in input1_tokens:
    # Direct match
    if word in expanded_products_tokens:
        matching_words.add(word)
    else:
        # Fuzzy match
        for prod_word in expanded_products_tokens:
            if fuzz.partial_ratio(word, prod_word) > 80:  # threshold can be adjusted
                matching_words.add(word)
                break

### Calculate the threshold
matched_pct = len(matching_words) / len(input1_tokens) * 100

n_product_key_words = len(second_brain['products'])
matched_n_product_words_pct = len(matching_words)/n_product_key_words * 100

n_found_stocks = len(found_stocks)
matched_n_found_stocks_pct = n_found_stocks * 100

list_matched_pcts = [matched_pct, matched_n_product_words_pct, matched_n_found_stocks_pct]
list_weights = [0.2, 0.4, 0.4]
threshold_pct = round(sum([matched_pct*weight for matched_pct, weight in zip(list_matched_pcts, list_weights)]), 2)

# Display the matching words and percentage
print(f"Matching words: {matching_words}")
print(f"len(matching_words): {len(matching_words)}")
print(f"Matching percentage: {matched_pct:.2f}%")
print(f"Found stocks: {found_stocks}")
print(f"len(found_stocks): {len(found_stocks)}")
print(f"threshold_pct: {threshold_pct}")

if threshold_pct > 50:
    print('True')
else:
    print('False')
