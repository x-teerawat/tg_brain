from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize
from fuzzywuzzy import fuzz

# Synonym dictionary
synonyms = {
    'รูปภาพ': ['ภาพ', 'image', 'รูป', 'ภาพถ่าย'],
    'วันที่': ['วัน', 'วันที่', 'date'],
    'กราฟ': ['graph', 'Graph', 'กราฟ'],
    'y-stat': ['ystat', 'y', 'staty', 'stat-y']
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
input1 = 'ฉันอยากได้รูปภาพแสดง graph ค่าy AMD ของวันที่ 20 เดือน7 2024'
input2 = 'ขอรูปภาพค่า y วันที่ 21 เดือน 7 2024 AMD'
input3 = 'ขอรูปภาพ Teerawat ทำท่าตัว y'
input4 = 'ขอกราฟ AMD วันนี้'
input5 = 'ขอภาพรถ 7 คัน ที่เรียงกันเป็นรูปตัว y ที่หน้าบ. AMD ในปี 2024'

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
    
import re
from pythainlp.tokenize import word_tokenize

# List of Thai month names
thai_months = [
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
]

# Function to extract and format date from input
def extract_dates(input_text):
    # Tokenize the input text
    tokens = word_tokenize(input_text)

    # Regular expressions for various date formats
    date_patterns = [
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',  # DD-MM-YYYY or DD/MM/YYYY
        r'(\d{1,2})\s*เดือน\s*(\d{1,2}|\w+)\s*(\d{4})',  # DD เดือน MM YYYY
        r'(\d{1,2})\s*(\d{1,2}|\w+)\s*(\d{4})'  # DD MM YYYY (with or without Thai month names)
    ]

    # Container for formatted dates
    formatted_dates = []

    for pattern in date_patterns:
        for match in re.finditer(pattern, input_text):
            day, month, year = match.groups()

            # Convert Thai month names to numbers if needed
            if month.isdigit():
                month = int(month)
            else:
                month = thai_months.index(month) + 1 if month in thai_months else None

            if month:
                formatted_date = f"{int(day):02d}/{int(month):02d}/{year}"
                formatted_dates.append(formatted_date)

    # Checking for relative dates (e.g., วันนี้, เมื่อวานนี้)
    relative_dates = {
        "วันนี้": "today",
        "เมื่อวานนี้": "yesterday",
        # More relative dates can be added here
    }

    for token in tokens:
        if token in relative_dates:
            formatted_dates.append(relative_dates[token])

    return formatted_dates

# Example input
input1 = 'ฉันอยากได้รูปภาพแสดง graph ค่าy AMD ของวันที่ 20 เดือน7 2023'
input2 = 'ขอรูปภาพค่า y วันที่ 21 เดือน 7 2023 AMD'
input3 = 'ขอรูปภาพ Teerawat ทำท่าตัว y'
input4 = 'ขอกราฟ AMD วันนี้'
input5 = 'ขอภาพรถ 7 คัน ที่เรียงกันเป็นรูปตัว y ที่หน้าบ. AMD ในปี 2023'
formatted_dates = extract_dates(input1)
# formatted_dates = extract_dates(input2)
# formatted_dates = extract_dates(input3)
# formatted_dates = extract_dates(input4)
# formatted_dates = extract_dates(input5)
print(f"Formatted Dates: {formatted_dates}")
