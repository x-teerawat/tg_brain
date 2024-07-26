from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize
from fuzzywuzzy import fuzz

# ข้อความที่ต้องการตรวจสอบ
text = "ฉันอยากได้ค่าจาก ystat ช่วยหาให้หน่อยได้ไหม"

# คำที่ต้องการเปรียบเทียบ
second_brain = {'products': ['y-stat', 'graph', 'ภาพ', 'image']}

# ใช้ pythainlp แบ่งคำในภาษาไทย
words = word_tokenize(text, engine='newmm')

# ลบ stopwords ที่มาจาก pythainlp
filtered_words = [word for word in words if word not in thai_stopwords()]

# รวมคำหลังจากลบ stopwords
filtered_text = ' '.join(filtered_words)

# ฟังก์ชันเปรียบเทียบความคล้ายคลึงกัน
def find_similarity(text, keywords):
    similarities = [(keyword, fuzz.ratio(text.lower(), keyword.lower())) for keyword in keywords]
    return similarities

# เรียกใช้ฟังก์ชันและแสดงผลลัพธ์
similarity_results = find_similarity(filtered_text, second_brain['products'])

print("ความคล้ายคลึงกันของแต่ละคำในรายการ:")
for keyword, similarity in similarity_results:
    print(f"{keyword}: {similarity}%")
