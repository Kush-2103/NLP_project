from flask import Flask, render_template, request, redirect, url_for
import os
import PyPDF2
import re
from transformers import BartTokenizer, BartForConditionalGeneration
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

app = Flask(__name__, template_folder="templateFiles", static_folder="staticFiles")


app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def remove_string(content, string_to_remove):
    content = re.sub(string_to_remove, '', content)
    return content.strip()

def extract_pdf_content(pdf_path, start_page, end_page):
    content_array = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        if start_page < 1 or end_page > num_pages:
            print("Invalid page range.")
            return content_array

        for page_number in range(start_page - 1, end_page):
            page = reader.pages[page_number]
            content = page.extract_text()

            content = remove_string(content, r'\nJUNE 2023\nDRDO NEWSLETTERCOVER STORY\n')

            content_array.append(content)

    return content_array

def preprocess_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def summarize_text(text):

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(inputs, num_beams=4, max_length=400, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return summary

def generate_wordcloud(text):

    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')

    wordcloud_image_path = 'static/wordcloud.png'
    plt.savefig(wordcloud_image_path)
    plt.close()

    return wordcloud_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        start_page = 1
        end_page = 5
        content_array = extract_pdf_content(file_path, start_page, end_page)

        preprocessed_array = [preprocess_text(text) for text in content_array]

        summaries = [summarize_text(text) for text in preprocessed_array]
        combined_text = ' '.join(summaries)

        wordcloud_image_path = generate_wordcloud(combined_text)

        return render_template('index.html', summaries=summaries, wordcloud_image_path=wordcloud_image_path)

    else:
        return redirect(request.url)

if __name__ == '__main__':
    app.run()
