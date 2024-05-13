import gradio as gr
import re
import librosa # For managing audio file
import fitz
import time
from logic import split_chunks_documents
from logic import create_and_save_embeddings
from logic import retrieve_relevant_docs
from logic import concat_top_docs
from logic import answer_question
from logic import csv_to_dict_list


model = "sentence-transformers/all-MiniLM-L6-v2"
groq_api_key = GROQ_API_KEY

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def run_audio(model_selected,history,question,uploaded_file):
    top_k_value = 3
    vector_path_audio = "vector_data/test.pt"   
    if len(history)>2:
        top_docs_audio = retrieve_relevant_docs(model, vector_path_audio, question, top_k_value)
        response = answer_question(model_selected, top_docs_audio, question, groq_api_key=groq_api_key)
    else:     
        audio, rate = librosa.load(uploaded_file[0], sr=16000)
        out = pipe(audio)['text']
        chunk_audio = split_chunks_documents(out, 500, 50)
        create_and_save_embeddings(model, chunk_audio, vector_path_audio)
        top_docs_audio = retrieve_relevant_docs(model, vector_path_audio, question, top_k_value)
        response = answer_question(model_selected, top_docs_audio, question, groq_api_key=groq_api_key)
    return response 

def url_scrape(model_selected,history,question,uploaded_file):
    top_k_value = 3
    vector_path_url = "vector_data/test2.pt" 
    pattern = "(?P<url>https?://[^\s]+)"
    url = re.findall(pattern, uploaded_file)
    if len(url) > 1:
        question = re.sub(pattern,"" ,uploaded_file)  
        
    if len(history)>1:
        top_docs_audio = retrieve_relevant_docs(model, vector_path_url, question, top_k_value)
        response = answer_question(model_selected, top_docs_audio, question, groq_api_key="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")
    else:      
        text_content = ""
        for i in url:
            o = get_text_from_webpage(i)
            text_content += o + "\n\n"

        chunks_url = split_chunks_documents(text_content, 500, 50)
        create_and_save_embeddings(model, chunks_url, vector_path_url)
        top_docs_url = retrieve_relevant_docs(model, vector_path_url, question, top_k_value)
        response = answer_question(model_selected, top_docs_url, question, groq_api_key=groq_api_key)
    return response 

def run_pdf(model_selected,history,question,uploaded_file):
    top_k_value = 3
    vector_path = "vector_data/test3.pt"
    if len(history)>2:
        top_docs = retrieve_relevant_docs(model,vector_path, question, top_k_value)
        all_top_docs = concat_top_docs(top_docs)
        response = answer_question(model_selected, all_top_docs , question ,groq_api_key=groq_api_key)
    else:
        doc = fitz.open(uploaded_file[0]) # open a document
        for page in doc: # iterate the document pages
            text = page.get_text().encode("utf8").decode("utf-8") # get plain text (is in UTF-8)
        chunks = split_chunks_documents(text, 500, 50)
        create_and_save_embeddings(model, chunks, vector_path)
        top_docs = retrieve_relevant_docs(model,vector_path, question, top_k_value)
        all_top_docs = concat_top_docs(top_docs)
        response = answer_question(model_selected, all_top_docs , question ,groq_api_key=groq_api_key)
    return response

def run_image(model_selected,history,question,uploaded_file):
    top_k_value = 3
    vector_path = "vector_data/test4.pt"
    if len(history)>2:
        top_docs = retrieve_relevant_docs(model,vector_path, question, top_k_value)
        all_top_docs = concat_top_docs(top_docs)
        response = answer_question(model_selected, all_top_docs , question ,groq_api_key=groq_api_key)
    else:
        out = caption(uploaded_file[0])[0]['generated_text']
        chunk_image = split_chunks_documents(out, 500, 50)
        create_and_save_embeddings(model, chunk_image, vector_path)
        top_docs = retrieve_relevant_docs(model, vector_path, question, top_k_value)
        response = answer_question(model_selected, top_docs, question, groq_api_key=groq_api_key)
    return response

def run_doc(model_selected,history,question,uploaded_file):
    top_k_value = 3
    vector_path = "vector_data/test5.pt"
    if len(history)>2:
        top_docs = retrieve_relevant_docs(model,vector_path, question, top_k_value)
        all_top_docs = concat_top_docs(top_docs)
        response = answer_question(model_selected, all_top_docs , question ,groq_api_key=groq_api_key)
    else:
        chunk = csv_to_dict_list(uploaded_file[0])
        create_and_save_embeddings(model, chunk, vector_path)
        top_docs = retrieve_relevant_docs(model, vector_path, question, top_k_value)
        all_top_docs = concat_top_docs(top_docs)
        response = answer_question(model_selected, all_top_docs, question, groq_api_key=groq_api_key)
    return response
    
def bot(history,model_choice):
    
    model_selected = model_choice
    question = history[-1][0]
    uploaded_file = history[0][0]

    if "http" in uploaded_file:
        type_of_file = "http"
    else:
        type_of_file = uploaded_file[0].split('.')[-1] 
    
    if type_of_file in ["wav","mp3","mp4"]:
        response = run_audio(model_selected,history,question,uploaded_file)
    elif type_of_file == "http":
        response = url_scrape(model_selected,history,question,uploaded_file)
    elif type_of_file == "pdf":
        response = run_pdf(model_selected,history,question,uploaded_file)
    elif type_of_file in ["png","jpg","jpeg"]:
        response = run_image(model_selected,history,question,uploaded_file)
    elif type_of_file in ["xls","csv","xlsx"]:
        response = run_doc(model_selected,history,question,uploaded_file)
    else:
        response = "PLEASE CHOOSE THE SOURCE FOR INPUT"
    
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.03)
        yield history
        

with gr.Blocks() as demo:
    gr.Markdown("## Multimodal RAG LLM")
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )
    
    with gr.Column(scale=1):
        model_choice = gr.Radio(label="Choose Model", choices=["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], info="Select the model you want to use.")
    
        chat_input = gr.MultimodalTextbox(interactive=True, file_types=["file"], placeholder="Upload file...", show_label=False)

        chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
        bot_msg = chat_msg.then(bot, [chatbot,model_choice], chatbot, api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
    

    chatbot.like(print_like_dislike, None, None)
    
# Launch the app
demo.launch()
