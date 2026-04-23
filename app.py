import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ---------------- LLM (Groq) ----------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- PINECONE ----------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX", "company-rag")

# 🔥 IMPORTANT: No embeddings here (fix for memory issue)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name
)

# ---------------- RBAC ----------------
ROLE_ACCESS = {
    "Finance": ["finance"],
    "Marketing": ["marketing"],
    "HR": ["hr", "general"],
    "Engineering": ["engineering"],
    "C-Level": ["engineering", "finance", "hr", "marketing", "general"],
    "Employee": ["general"]
}

# ---------------- LLM ANSWER ----------------
def generate_answer(docs, question):
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are FinSolve AI Assistant.

Answer clearly and professionally based only on the context below.
If answer is not found, say "No data available".

Context:
{context}

Question:
{question}
"""
    return llm.invoke([HumanMessage(content=prompt)]).content


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question")
        role = data.get("role", "Employee")

        if not question:
            return jsonify({"error": "Question missing"}), 400

        # 🔐 RBAC filtering
        allowed_domains = ROLE_ACCESS.get(role, [])

        docs = vectorstore.similarity_search(
            question,
            k=10,  # 🔥 better retrieval
            filter={
                "allowed_roles": {"$in": [role]},
                "domain": {"$in": allowed_domains}
            }
        )

        if not docs:
            return jsonify({"answer": "No relevant data found."})

        # 🤖 Generate response
        answer = generate_answer(docs, question)

        return jsonify({"answer": answer})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Server error"}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)