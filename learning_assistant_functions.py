from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import TokenTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from learning_assistant_prompts import question_template, rag_template
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import spacy
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import GPT4AllEmbeddings
import json
from io import BytesIO
import tempfile
from typing import List
from langchain.schema import Document
from langchain.llms import BaseLLM
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import pytextrank
# def load_data(data):
#     pdf_file = BytesIO(data)
#     loader = PyPDFium2Loader(pdf_file)
#     return loader.load()
class SpacyEmbeddings(Embeddings):
    def __init__(self, model_name: str = "en_core_web_md"):
        self.nlp = spacy.load(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.nlp(text).vector.tolist() for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self.nlp(text).vector.tolist()

def load_data(data):
    # Create a temporary file to save the PDF content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(data)
        temp_file_path = temp_file.name

    # Use PyPDFLoader to load the PDF
    loader = PyPDFLoader(temp_file_path)
    documents= loader.load()
    return " ".join([doc.page_content for doc in documents])
def load_spacy_model():
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("textrank")
    return nlp


def summarize_with_pytextrank(documents: List[Document], summary_ratio: float = 0.1, min_sentences: int = 3, max_sentences: int = 20) -> Document:
    nlp = load_spacy_model()
    
   
    full_text = " ".join([doc.page_content for doc in documents])

    doc = nlp(full_text)

    total_sentences = len(list(doc.sents))
    
    target_sentences = max(min_sentences, min(int(total_sentences * summary_ratio), max_sentences))
    

    tr = doc._.textrank
    summary_sentences = [sent.text for sent in tr.summary(limit_sentences=target_sentences)]
    summary_text = " ".join(summary_sentences)
    
    return Document(page_content=summary_text, metadata={
        "source": "summary"
    })
# def load_data(data):
#     pdf_file = BytesIO(data)
#     pdf_reader = PdfReader(pdf_file)
#     text = " ".join(page.extract_text() for page in pdf_reader.pages)
#     return text

# def split_text(text,chunk_size,chunk_overlap):
#     splitter = TokenTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
#     return splitter.create_documents(text)
def split_text(text, chunk_size, chunk_overlap):

    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")


    words = text.split()
   
    chunks = []
    chunk_words = []
    word_count = 0

    for word in words:
        chunk_words.append(word)
        word_count += 1

        if word_count >= chunk_size:
            chunks.append(' '.join(chunk_words))

            chunk_words = chunk_words[-chunk_overlap:]
            word_count = len(chunk_words)
    if chunk_words:
        chunks.append(' '.join(chunk_words))

    documents = [Document(page_content=chunk) for chunk in chunks]

    return documents

def initialize_llm(api_key):
    #return ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
    return ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2, google_api_key=api_key
    # other params...
)
# def generate_questions(llm,text,nquestions):
#     processed_text = split_text(load_data(text),chunk_size=1000, chunk_overlap=200)
#     summarize=  load_summarize_chain(llm,"stuff")
#     summary = summarize.run(processed_text)
#     question_generation_chain = LLMChain(llm=llm,prompt=question_template)
#     questions = question_generation_chain.run(summary=summary,nquestions=nquestions)
#     try:
#         question_list= json.loads(questions)
#         assert isinstance(questions,list)
#     except json.JSONDecodeError:
#          question_list = [q.strip() for q in questions.split('\n') if q.strip()]
#          return question_list[:nquestions]
def generate_questions(llm,text,nquestions):
    processed_text = split_text(load_data(text),chunk_size=1000, chunk_overlap=200)
   
    
    

    summary = summary_document = summarize_with_pytextrank(processed_text, summary_ratio=0.1, min_sentences=3, max_sentences=100)

    question_generation_chain = LLMChain(llm=llm,prompt=question_template)
    questions = question_generation_chain.run(summary=summary.page_content,nquestions=nquestions)
    questions = questions.replace("```json", "").replace("```", "").strip()
    
    try:
        question_list= json.loads(questions)
        
        assert isinstance(question_list,list)
        return question_list
    except json.JSONDecodeError:
         question_list = [q.strip() for q in questions.split('\n') if q.strip()]
         return question_list[2:nquestions+2]

def create_retrieval_qa_chain(llm,document):
    embeddings = SpacyEmbeddings()
    loaded_docs = split_text(load_data(document),chunk_size=1000, chunk_overlap=200)
    db = FAISS.from_documents(loaded_docs, embeddings)
    return RetrievalQA.from_chain_type(llm,retriever=db.as_retriever(),chain_type_kwargs={"prompt":rag_template})




