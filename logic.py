import os
import time
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm
from groq import Groq
import csv

import pandas as pd


pipe = pipeline("automatic-speech-recognition", model="facebook/s2t-medium-librispeech-asr")
caption = pipeline('image-to-text',model="Salesforce/blip-image-captioning-base")
model = "sentence-transformers/all-MiniLM-L6-v2"

def split_chunks_documents(text, chunk_size, overlap):
    return [text[i:min(i + chunk_size, len(text))] for i in range(0, len(text), chunk_size - overlap)]


# Improved Function for Efficiency and Readability
def create_and_save_embeddings(embed_model, chunks, vector_path):
    """
    Computes embeddings for document chunks, stores them in a dictionary,
    and saves the dictionary as a PyTorch file.

    Args:
        embed_model (str): Name of the SentenceTransformer model to use.
        chunks (list): List of text chunks to create embeddings for.
        vector_path (str, optional): vector_path to save the PyTorch file. Defaults to "embeddings.pt".

    Returns:
        None
    """

    embedder = SentenceTransformer(embed_model)
    embeddings = []

    with tqdm(total=len(chunks), desc="Computing Embeddings...") as pbar:  # Improved Progress Bar
        for chunk in chunks:
            embedding = embedder.encode(chunk, convert_to_tensor=True).to("cpu")
            embeddings.append(embedding)
            pbar.update(1)  # Update progress bar for each chunk

    embeddings_dict = dict(zip(chunks, embeddings))  # Concise dictionary creation

    torch.save(embeddings_dict, vector_path)

def retrieve_relevant_docs(embed_model, embeddings_path, query, top_k):
    """
    Retrieves the most relevant documents (chunks) from a pre-computed embedding dictionary based on a query.

    Args:
      embed_model (str): Name of the SentenceTransformer model used for embeddings.
      embeddings_path (str): Path to the PyTorch file containing the document embeddings dictionary.
      query (str): The query string to search for relevant documents.
      top_k (int): Number of top most relevant documents to retrieve.

    Returns:
      list: List of the top-k most relevant document chunks (strings).
    """

    # Load SentenceTransformer model and embeddings dictionary
    embedder = SentenceTransformer(embed_model)
    embeddings_dict = torch.load(embeddings_path)

    # Encode query and ensure top_k doesn't exceed available documents
    query_encoded = embedder.encode(query, convert_to_tensor=True)
    top_k = min(top_k, len(embeddings_dict))  # Limit top_k to available documents

    # Efficient similarity computation with batching (if applicable)
    if len(embeddings_dict) > 1000:  # Consider batching for large datasets
        all_scores = []
        for chunk_batch in tqdm(torch.split(list(embeddings_dict.keys()), 1000), desc="Computing Similarity..."):
            chunk_embeddings = torch.stack([embeddings_dict[chunk] for chunk in chunk_batch])
            batch_scores = util.cos_sim(query_encoded.unsqueeze(0), chunk_embeddings)  # Batch query encoding
            all_scores.extend(batch_scores.squeeze().tolist())  # Extract scores
    else:
        # For smaller datasets, iterate through each document
        scores = []
        for chunk in tqdm(embeddings_dict.keys(), desc="Computing Similarity..."):
            cos_score = util.cos_sim(query_encoded, embeddings_dict[chunk])[0]
            scores.append(cos_score)

    # Retrieve top-k documents based on similarity scores
    scores = torch.tensor(scores if len(scores) > 0 else all_scores)  # Handle empty score list
    top_k_chunk_indices = torch.topk(scores, k=top_k).indices.tolist()
    top_k_chunks = [list(embeddings_dict.keys())[i] for i in top_k_chunk_indices]

    return top_k_chunks


def concat_top_docs(relevant_chunks):
    """
    Creates a single string by concatenating relevant document chunks with clear formatting.

    Args:
      relevant_chunks (list): List of strings containing the relevant document chunks.

    Returns:
      str: A single string combining all relevant document chunks with numbering and newlines.
    """

    # Efficient string formatting with enumerate
    stuffed_chunk = "\n\n".join([f"Document {i+1}: {chunk}" for i, chunk in enumerate(relevant_chunks)])
    return stuffed_chunk


prompt = """
Considering the user question {question} and the provided context, identify the most relevant parts of the context that can be helpful in answering the question.

relevant_context: {context}

Based on the identified relevant context and the user question, answer the question in a comprehensive and informative way, providing additional insights and explanations when possible. Aim for a balance between conciseness and clarity.

answer:

confidence_score: (between 0.0 and 1.0, indicating how certain the model is about the answer)
"""

def answer_question(large_language_model_name, conversation_history, user_query, groq_api_key):
    """
    Queries a large language model (LLM) using Groq for question answering.

    Args:
      large_language_model_name (str): Name of the LLM model to use (e.g., "lamda-7b").
      conversation_history (str): Contextual information for the question.
      user_query (str): The question to ask the LLM.
      groq_api_key (str, optional): Your Groq API key (if not set as environment variable).

    Returns:
      str: The answer generated by the LLM.
    """

    # Load API key from environment variable (if available)
    if groq_api_key is None:
        groq_api_key = os.getenv("GROQ_API_KEY")

    # Early check for missing API key
    if not groq_api_key:
        raise ValueError("Missing Groq API key. Please set GROQ_API_KEY environment variable or provide the key as an argument.")

    # Create Groq client and construct LLM input
    groq_client = Groq(api_key = groq_api_key)
    llm_input = prompt.format(context=conversation_history, question=user_query)

    # Send request and extract answer
    chat_completion = groq_client.chat.completions.create(
      messages=[{"role": "user", "content": llm_input}],
      model=large_language_model_name
    )
    return chat_completion.choices[0].message.content

def csv_to_dict_list(file,csv=None):
    chunk_size = 20  # 20% of total length
    data_chunks = []
    if file.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
          
    total_rows = len(df)
    chunk_length = int(chunk_size * total_rows / 100)

    for chunk_start in range(0, total_rows, chunk_length):
        chunk_end = min(chunk_start + chunk_length, total_rows)
        chunk_df = df.iloc[chunk_start:chunk_end]
        chunk_data = []

        for _, row in chunk_df.iterrows():
            row_dict = row.to_dict()
            formatted_row = "\n".join(f"{key}: {val}" for key, val in row_dict.items())
            chunk_data.append(formatted_row)

        data_chunks.append('\n'.join(chunk_data))

    return data_chunks

if __name__ == "__main__":
    main()
