import requests
import os

import tracetune as tt

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# use ColBERTv2 retrieval model that's publically hosted
def retrieve(query: str, k: int = 2):
    payload = {"query": query, "k": k}
    res = requests.get("http://20.102.90.50:2017/wiki17_abstracts", params=payload, timeout=10)

    topk = res.json()["topk"][:k]
    topk = [{**d, "long_text": d["text"]} for d in topk]
    return topk[:k]

def call_openai_api(prompt, model="gpt-4o", temperature=0.7):
    """Calls the OpenAI API with a given prompt and returns the response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an AI assistant that generates search queries and answers questions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()["choices"][0]["message"]["content"]


def generate_query(context, question):
    prompt = (
        "You will take as input a context and a question. "
        "Your task is to generate a search query that would help find relevant information."
        f"Context:\n{context}\n\nQuestion: {question}"
        "Now generate a search query that would help find relevant information."
    )
    return call_openai_api(prompt)

def generate_answer(context, question):
    prompt = (
        "You will take as input a context and a question. "
        "Your task is to generate an accurate and concise answer."
        f"Context:\n{context}\n\nQuestion: {question}"
        "Now answer the question accurately and concisely."
    )
    return call_openai_api(prompt)


@tt.trace(llm_fns=["generate_query", "generate_answer"])
def multi_hop(question, hops=2):
    context = []

    for hop in range(hops):
        query = generate_query(context=context, question=question)
        context += retrieve(query=query)

    answer = generate_answer(context=context, question=question)
    return answer

traced_multi_hop = multi_hop("How many storeys are in the castle that David Gregory inherited?")

print("====== Trace =======")
print(traced_multi_hop.trace_string)
print("====================")
print()
print("Program Output:", traced_multi_hop.output)
