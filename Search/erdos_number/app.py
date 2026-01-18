from flask import Flask, request, jsonify, render_template
import uuid
import threading
from typing import Dict, Any

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# In-memory job storage (for simplicity in first version)
jobs: Dict[str, Dict[str, Any]] = {}

from search import bidirectional_wiki_search

def sanitize_title(title: str) -> str:
    """Extracts the Wikipedia title from a URL if provided, and normalizes it."""
    title = title.strip()
    if "wikipedia.org/wiki/" in title:
        title = title.split("wikipedia.org/wiki/")[-1].split("?")[0].split("#")[0]
        from urllib.parse import unquote
        title = unquote(title).replace("_", " ")
    return title

@app.route("/api/search", methods=["POST"])
def start_search():
    data = request.json
    start_title = sanitize_title(data.get("start", ""))
    end_title = sanitize_title(data.get("end", ""))
    
    if not start_title or not end_title:
        return jsonify({"error": "Missing start or end title"}), 400
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "searching",
        "start": start_title,
        "end": end_title,
        "path": None,
        "processed_nodes": 0,
        "error": None
    }
    
    def run_search(jid, s, e):
        try:
            path = bidirectional_wiki_search(s, e)
            if path:
                jobs[jid]["status"] = "complete"
                jobs[jid]["path"] = path
            else:
                jobs[jid]["status"] = "failed"
                jobs[jid]["error"] = "No path found within depth limit."
        except Exception as ex:
            jobs[jid]["status"] = "failed"
            jobs[jid]["error"] = str(ex)

    thread = threading.Thread(target=run_search, args=(job_id, start_title, end_title))
    thread.start()
    
    return jsonify({"job_id": job_id}), 202

@app.route("/api/search/<job_id>", methods=["GET"])
def get_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(job)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
