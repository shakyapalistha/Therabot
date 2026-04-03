"""
app.py — Flask server.
 
"""
 
from flask import Flask, send_from_directory, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from main import get_response, stream_response, is_safe_query
import os
import logging
import time
 

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
 

# APP SETUP
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
 
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app, resources={r"/chat*": {"origins": "*"}})
 

# CRISIS DETECTION
CRISIS_KEYWORDS = {"suicide", "kill myself", "end my life", "self harm", "hurt myself"}
 
CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this way — you're not alone. "
    "Please reach out to a trusted person or a mental health professional immediately. "
    "If you're in immediate danger, contact your local emergency services or a crisis helpline."
)
 
def is_crisis(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in CRISIS_KEYWORDS)
 
 
# STATIC FILES
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")
 
@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)
 
 
# CHAT  — buffered
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
 
    query = data.get("message", "").strip()
 
    if not query:
        return jsonify({"response": "No message received."}), 400
 
    if not is_safe_query(query):
        return jsonify({"response": "Please provide a valid question so I can help you better."}), 400
 
    if is_crisis(query):
        return jsonify({"response": CRISIS_RESPONSE})
 
    t0 = time.time()
    try:
        # FIX: single call — retrieval + rerank + format + LLM all happen inside
        result = get_response(query)
        logger.info("Chat answered in %.2fs", time.time() - t0)
        return jsonify({"response": result})
 
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        return jsonify({
            "response": "I'm sorry, something went wrong. Please try again in a moment."
        }), 500
 

# CHAT  — streaming
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
 
    query = data.get("message", "").strip()
 
    if not query:
        return jsonify({"response": "No message received."}), 400
 
    if not is_safe_query(query):
        return jsonify({"response": "Please provide a valid question so I can help you better."}), 400
 
    if is_crisis(query):
        return jsonify({"response": CRISIS_RESPONSE})
 
    def generate():
        try:
            # FIX: stream_response() handles retrieval + rerank + format + streaming
            yield from stream_response(query)
        except Exception as e:
            logger.error("Stream error: %s", e, exc_info=True)
            yield "\nSomething went wrong. Please try again."
 
    return Response(
        stream_with_context(generate()),
        mimetype="text/plain",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
        },
    )
 
 
# HEALTH CHECK
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200
 
 
# RUN
if __name__ == "__main__":
    app.run(
        port=5000,
        debug=False,
        threaded=True,
    )