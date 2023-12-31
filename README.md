## PDF Content Extraction, Summarization, and Analysis

This code demonstrates a Python script for extracting content from a PDF file, performing text summarization and question-answering, generating a word cloud, performing named entity recognition (NER), and creating a summary PDF using the fpdf lirary. It utilizes various libraries and models to accomplish these tasks. 

### Steps Performed

Steps performed for the project-

 - PDF **content extraction** using pyPDF2 library
 - **Tokenization**, stop word removal, and pre-processing steps.
 - The code utilizes the **BART model** for text summarization. BART (Bidirectional and Auto-Regressive Transformers) is a sequence-to-sequence transformer model developed by Facebook AI. It is designed for various natural language processing tasks, including text summarization. The code uses the `transformers` library to load the pre-trained BART model (`facebook/bart-large-cnn`) and tokenizer (`facebook/bart-large-cnn`) for text summarization.
 - **PDF also generated** through the use of fpdf library.[Summarised PDF](https://github.com/Kush-2103/NLP_project/blob/main/summary_pdf.pdf)
 - **WordCloud Generation** 
 - ![Word Cloud](https://github.com/Kush-2103/NLP_project/blob/main/static/WordCloud.png)
 
 - Used the ensemble method for **QnA model**. I used 3 models namely *distilbert-base-cased-distilled-squad*, *bert-large-uncased-whole-word-masking-finetuned-squad, google/electra-base-discriminator*. Answer will be selected using the Vote count method.
 

### Usage

1.  Install the required dependencies using the command above or by adding them to your `requirements.txt` file and running `pip install -r requirements.txt`.
2.  Set the `pdf_path` variable to the path of the PDF file you want to process.
3.  Define the `start_page` and `end_page` variables to specify the range of pages to extract content from.
4.  Run the script.

Please note that you might need to adjust the code according to your specific requirements, such as changing the models used for question-answering or modifying the preprocessing steps.

### References

-   [PyPDF2 Documentation](https://pythonhosted.org/PyPDF2/)
-   [Hugging Face Transformers Library](https://huggingface.co/transformers/)
-   [NLTK Documentation](https://www.nltk.org/)
-   [WordCloud Documentation](https://amueller.github.io/word_cloud/)
-   [Matplotlib Documentation](https://matplotlib.org/)
-   [spaCy Documentation](https://spacy.io/)
-   [fpdf Documentation](https://pyfpdf.readthedocs.io/)
