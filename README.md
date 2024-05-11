<div>
<h1>GroqLLMRagApp</h1>
     <h2>Description</h2>
    <p>This project implements a Retrieval-Augmented Generative (RAG) conversational AI chatbot powered by Groq and Gradio. It utilizes Sentence Transformers for efficient document retrieval and leverages pre-trained large language models (LLMs) through Groq for comprehensive question answering. The chatbot can handle various input modalities, including text documents, audio files, URLs, images, and CSV/Excel files.</p>
    <h2>Key Features</h2>
    <ul>
        <li>Multimodal interaction: Text, audio, URL, image, and CSV/Excel file uploads</li>
        <li>Contextual awareness for improved understanding of user queries</li>
        <li>Retrieval of relevant document segments to support responses</li>
        <li>Integration with various LLM models via Groq API</li>
        <li>Confidence score for answer reliability (to be implemented)</li>
    </ul>
    <h2>Requirements</h2>
    <ul>
        <li>Python 3.9 (<a href="https://www.python.org/downloads/">https://www.python.org/downloads/</a>)</li>
        <li>Transformers library (<a href="https://huggingface.co/docs/transformers/en/index">https://huggingface.co/docs/transformers/en/index</a>)</li>
        <li>SentenceTransformers library (<a href="https://huggingface.co/sentence-transformers">https://huggingface.co/sentence-transformers</a>)</li>
        <li>Groq account and API key (<a href="https://groq.com/">https://groq.com/</a>)</li>
        <li>Gradio library (<a href="https://www.gradio.app/guides/quickstart">https://www.gradio.app/guides/quickstart</a>)</li>
        <li>Additional libraries for specific modalities:</li>
            <ul>
                <li>librosa for audio (pip install librosa)</li>
                <li>fitz for PDF (pip install fitz-py)</li>
                <li>pandas for CSV/Excel (pip install pandas)</li>
            </ul>
    </ul>
    <h2>Installation</h2>
    <pre>pip install transformers sentence-transformers groq gradio librosa fitz pandas  # For all functionalities</pre>
    <p>Alternatively, you can install only the required libraries based on the modalities you want to support.</p>
    <h2>Usage</h2>
    <ol>
        <li>Clone this repository or download the project files.</li>
        <li>Install the required libraries (see Requirements).</li>
        <li>Configure your Groq API key (obtain from your Groq account).</li>
            <ul>
                <li>You can set the <code>GROQ_API_KEY</code> environment variable.</li>
                <li>Alternatively, you can modify the `answer_question` function in `logic.py` to provide the API key as an argument.</li>
            </ul>
        <li>Prepare your input documents (text files, audio files, URLs, images, or CSV/Excel files).</li>
            <ul>
                <li>For text files, ensure each line is a separate document.</li>
                <li>For audio files, ensure they are in a supported format (e.g., WAV, MP3, MP4).</li>
                <li>For URLs, ensure they are valid links.</li>
                <li>For images, ensure they are in a supported format (e.g., PNG, JPG, JPEG).</li>
                <li>For CSV/Excel files, make sure the relevant text column is identified.</li>
            </ul>
        <li>(Optional) If using CSV/Excel files, modify the `csv_to_dict_list` function in `logic.py` to adjust column names or formatting as needed.</li>
        <li>Run the Gradio app:</li>
        <pre>python groq_app.py</pre>
        <li>Open <code>http://localhost:1274</code> in your web browser to interact with the chatbot.</li>
    </ol>
    <h2>Explanation of Files</h2>
    <ul>
        <li><code>logic.py</code>: Contains core logic for document processing, retrieval, and question answering using RAG, Groq, and handling different modalities.</li>
        <li><code>groq_app.py</code>: Handles the Gradio app interface and interaction with `logic.py`. It defines functions like `print_like_dislike`, `add_message`, and functions for processing different modalities like audio, URLs, PDFs, images, and CSV/Excel files.</li>
    </ul>
    <h2>Additional Notes</h2>
    <ul>
        <li>The current implementation assumes pre-computed document embeddings. You may need to modify the code to generate embeddings from scratch if necessary.</li>
        <li>Consider adding error handling and logging for a more robust user experience.</li>
        <li>The confidence score for answer reliability is not yet implemented but can be integrated in the future.</li>
    </ul>
    <h2>Contributing</h2>
    <p>Feel free to submit pull requests for improvements or bug fixes.</p>
</div>