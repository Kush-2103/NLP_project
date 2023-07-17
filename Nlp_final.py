pdf_path = '/content/NL_June2023.pdf'

import PyPDF2
import re
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')

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

start_page = 4
end_page = 24

page_content_array = extract_pdf_content(pdf_path, start_page, end_page)

def preprocess_text(text):
    try:
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        # text = text.encode('latin-1', 'ignore').decode('latin-1')
    except UnicodeEncodeError:
        print("Warning: Skipping word due to encoding issue")

    return text
preprocessed_array = [preprocess_text(text) for text in page_content_array]

from transformers import BartTokenizer, BartForConditionalGeneration

def summarize_text(text):

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(inputs, num_beams=4, max_length=400, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return summary

text = preprocessed_array[1]
summary = summarize_text(text)

print("Summary:")
print(summary)

len(summary)

from transformers import pipeline
models = [
    {
        "model": "distilbert-base-cased-distilled-squad",
    },
    {
        "model": "bert-large-uncased-whole-word-masking-finetuned-squad",
    },
    {
        "model": "google/electra-base-discriminator",
    },

]

model_pipelines = []

for model_info in models:
    model = model_info["model"]
    qa_pipeline = pipeline(
        "question-answering",
        model=model,
    )
    model_pipelines.append(qa_pipeline)

context = " ".join(preprocessed_array)

question = "MOU was signed by whom?"
answers = []
for qa_pipeline in model_pipelines:
    result = qa_pipeline(question=question, context=context)
    answers.append(result["answer"])

aggregated_answer = max(set(answers), key=answers.count)

print("Aggregated Answer:")
print(aggregated_answer)

import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
additional_stop_words = ["director", "dr", "shri", "may", "june", "sc","april"]
stop_words.update(additional_stop_words)

def preprocess_text_word(text):

    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)

    return text
preprocessed_array_word = [preprocess_text_word(text) for text in page_content_array]
# preprocessed_array_word[0]

combined_text = ' '.join(preprocessed_array_word)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Combined Word Cloud')

plt.show()

import spacy
nlp = spacy.load('en_core_web_sm')

def perform_ner(text):
    doc = nlp(text)

    named_entities = []
    for entity in doc.ents:
        named_entities.append((entity.text, entity.label_))

    return named_entities

ner_results = []
for text in preprocessed_array:
    entities = perform_ner(text)
    ner_results.append(entities)

output_file = "ner_results.txt"
with open(output_file, 'w') as file:
    for i, entities in enumerate(ner_results):
        file.write(f"Named Entities in preprocessed_array[{i}]:\n")
        for entity in entities:
            file.write(f"{entity[0]} - {entity[1]}\n")
        file.write("\n")

print(f"NER results saved to {output_file}")

!pip install fpdf2

def preprocess_text(text):
    try:
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.encode('latin-1', 'ignore').decode('latin-1')
    except UnicodeEncodeError:
        print("Warning: Skipping word due to encoding issue")

    return text
preprocessed_array = [preprocess_text(text) for text in page_content_array]

from fpdf import FPDF, FPDFException

class CustomPDF(FPDF):
    # def header(self):
    #     self.set_font("Arial", "B", 16)
    #     self.cell(0, 10, "Summary of Pages", ln=True)
    #     self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True)
        self.ln(10)

    def chapter_body(self, content):
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, content)
        self.ln(10)

def create_summary_pdf(pdf_path, start_page, end_page):
    pdf = CustomPDF()

    try:

        pdf.add_page()
        for i, text in enumerate(preprocessed_array):
            summary = summarize_text(text)

            pdf.chapter_title(f"Summary of Page {i + start_page}")
            pdf.chapter_body(summary)

        pdf.output("summary.pdf")
    except FPDFException as e:
        print(f"An error occurred while generating the PDF: {e}")

pdf_path = '/content/NL_June2023.pdf'
start_page = 4
end_page = 24

create_summary_pdf(pdf_path, start_page, end_page)


