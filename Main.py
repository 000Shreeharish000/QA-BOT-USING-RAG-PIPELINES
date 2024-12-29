import os
import pinecone
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import signal
import json

# Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Pinecone
PINECONE_API_KEY = "pcsk_279afX_QMNMfUTXqEfoeveEx5M1b4jKmnc1pSJP3FYYBJrH5yYkuBUqswAQF2wXGsJLPZ3"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index configuration
INDEX_NAME = "custom-faq"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connecting to index
index = pc.Index(INDEX_NAME)

#  model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the model
model_name = "facebook/opt-1.3b" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# text generation pipeline
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device=0 if torch.cuda.is_available() else -1)

# Loading documents from JSON file
with open('documents.json', 'r') as file:
    data = json.load(file)
    documents = [{"id": str(i), "text": f"{faq['question']} {faq['answer']}"} for i, faq in enumerate(data["FAQs"])]

def upload_documents():
    try:
        for doc in documents:
            embedding = sentence_model.encode(doc['text']).tolist()
            index.upsert([(doc['id'], embedding, {"text": doc['text']})])
    except Exception as e:
        print(f"Error uploading documents: {e}")

def retrieve_context(query, top_k=3):
    try:
        query_embedding = sentence_model.encode(query).tolist()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        contexts = [match['metadata']['text'] for match in results['matches']]
        return contexts
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

def is_relevant(query, contexts):
    # Simple relevance check based on keyword presence
    query_keywords = set(query.lower().split())
    context_keywords = set(" ".join(contexts).lower().split())
    return any(keyword in context_keywords for keyword in query_keywords)

def generate_answer(query, contexts):
    try:
        if not is_relevant(query, contexts):
            return "Sorry, the content is irrelevant. You can ask questions relevant to:\n- ABOUT THE COMPANY\n- WHAT WE PROVIDE\n- DETAILS ABOUT OUR SOFTWARE\nand so on..."

        context_text = "\n".join(contexts)
        prompt = f"""Context:
{context_text}

Question: {query}
Answer:"""

        # checking the length
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_length = input_ids.shape[1]

        if input_length > 2048:
            input_ids = input_ids[:, :2048]  # token limt

        response = llm(prompt, max_length=256, num_return_sequences=1, truncation=True)
        answer = response[0]['generated_text'].split("Answer:")[-1].strip()
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer at this time."

def rag_pipeline(query):
    try:
        contexts = retrieve_context(query)
        if not contexts:
            return "No relevant information found."
        return generate_answer(query, contexts)
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return "An error occurred while processing your query."

def handle_greeting(query):
    greetings = ["hi", "hello", "hey", "howdy"]
    if query.lower() in greetings:
        return "Hey! How can I help you?"
    return None

# Timeout 
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60) 

if __name__ == "__main__":
    try:
        upload_documents()
        print("Welcome to Velotriz Chatbot! Velotriz India's No 1 Software Company.")
        while True:
            test_query = input("Feel Free to Ask Your doubts (or type 'exit' to quit): ")
            if test_query.lower() == 'exit':
                print("Thank you for contacting us. I hope we cleared your queries.")
                break
            greeting_response = handle_greeting(test_query)
            if greeting_response:
                print(f"{greeting_response}\n")
            else:
                answer = rag_pipeline(test_query)
                print(f"Answer: {answer}\n")
    except TimeoutException:
        print("The operation timed out.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        signal.alarm(0) 
