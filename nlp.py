from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)  # Flask uygulaması oluşturuluyor

model_path = "C:\\Users\\Furkan\\Desktop\\content\\outputs"
tokenizer_path = "C:\\Users\\Furkan\\Desktop\\content\\outputs"
# Model ve tokenizer yükleniyor
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data['text']
    # Metni sınıflandırıp sonucu döndür
    result = nlp(text)
    # Sonucun içerisinde belirli bir afet türünü arayarak 'yes' veya 'no' döndür
    response = 'yes' if result[0]['label'] == 'LABEL_0' else 'no'
    return jsonify({'prediction': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

