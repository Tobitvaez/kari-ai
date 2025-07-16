from flask import Flask, request, render_template_string
import os

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
except ModuleNotFoundError as e:
    raise ImportError("[ERROR] Falta un módulo requerido: {}".format(e))

# API Key (reemplaza por tu clave real si vas a usarlo)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# === PDF ===
pdf_path = "documentos/mis_apuntes.pdf"
documents = []
if os.path.exists(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        raw_documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(raw_documents)
    except Exception as e:
        print("[ERROR] Procesando PDF:", e)
else:
    print("[ERROR] No se encontró el PDF:", pdf_path)

qa_chain = None
if documents:
    try:
        vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.5),
            retriever=retriever,
            memory=memory,
            verbose=False
        )
    except Exception as e:
        print(f"[ERROR] Error configurando LangChain: {e}")

app = Flask(__name__)
html_template = """
<!DOCTYPE html>
<html><head><title>Kari AI</title></head>
<body>
    <h1>Kari</h1>
    <form method="post">
        <input type="text" name="question" placeholder="Escribe tu pregunta" autofocus>
        <input type="submit" value="Enviar">
    </form>
    {% if response %}
        <p><strong>Kari:</strong> {{ response }}</p>
    {% endif %}
</body></html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    response = None
    if request.method == "POST":
        question = request.form.get("question")
        if qa_chain:
            try:
                response = qa_chain.run(question)
            except Exception as e:
                response = f"[Error] {e}"
        else:
            response = "Base de conocimiento no cargada."
    return render_template_string(html_template, response=response)

if __name__ == "__main__":
    app.run(debug=True)
