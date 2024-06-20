
from flask import Flask, render_template, request, redirect, url_for
import os
from PyPDF2 import PdfReader
import re
import nltk
import numpy as np
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    text_content = ""
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text_content += page.extract_text()
    text_content = re.sub(r'\s+', ' ', text_content).strip()
    return text_content

def preprocess_sentence(sentence, remove_stop_words=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    if remove_stop_words:
        sentence = remove_stopwords(sentence)
    return sentence

def get_word_vector(word, model):
    sample_vector = model['computer']
    try:
        vector = model[word]
    except KeyError:
        vector = [0] * len(sample_vector)
    return vector


def compute_phrase_embedding(phrase, embedding_model):
    sample_vector = get_word_vector('example', embedding_model)
    phrase_vector = np.zeros(len(sample_vector))
    word_count = 0
    for word in phrase.split():
        word_count += 1
        phrase_vector += np.array(get_word_vector(word, embedding_model))
    return phrase_vector.reshape(1, -1)

def find_best_matching_sentence(question_embedding, sentence_embeddings):
    highest_similarity = -1
    best_match_index = -1
    for index, embedding in enumerate(sentence_embeddings):
        similarity = cosine_similarity(embedding, question_embedding)[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_index = index
    return best_match_index


def clean_and_tokenize_sentences(sentences, remove_stop_words=False):
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = preprocess_sentence(sentence, remove_stop_words)
        cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences


glove_model = gensim.models.KeyedVectors.load('Downloads/Flask_QA_System/glovemodel.mod')  

def process_pdf_and_question(pdf_path, question):
    pdf_text = extract_text_from_pdf(pdf_path)
    sentences = nltk.sent_tokenize(pdf_text)
    clean_sentences_with_stopwords = clean_and_tokenize_sentences(sentences, remove_stop_words=True)
    original_sentences = clean_and_tokenize_sentences(sentences, remove_stop_words=False)
    
    sentence_embeddings = [compute_phrase_embedding(sentence, glove_model) for sentence in clean_sentences_with_stopwords]
    question_embedding = compute_phrase_embedding(question, glove_model)
    best_sentence_index = find_best_matching_sentence(question_embedding, sentence_embeddings)
    
    return original_sentences[best_sentence_index]


@app.route('/', methods=['GET', 'POST'])
def handle_upload():
    if request.method == 'POST':
        if request.form.get('btn') == 'index':
            uploaded_file = request.files['upload']
            uploaded_file.save(os.path.join('uploads', uploaded_file.filename))
            global uploaded_pdf_path
            uploaded_pdf_path = os.path.join('uploads', uploaded_file.filename)
            return redirect(url_for('display_qa'))
        elif request.form.get('btn') == 'QuesAns':
            user_question = request.form.get('question')
            answer = process_pdf_and_question(uploaded_pdf_path, user_question)
            return render_template('question_answer.html', answer=answer, question=user_question)
    return render_template('upload_pdf.html')

@app.route('/ques_ans/', methods=['GET', 'POST'])
def display_qa():
    return render_template('question_answer.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
