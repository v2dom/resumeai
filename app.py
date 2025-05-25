from flask import Flask, render_template, request
from utils.parse_resume import extract_text as extract_resume
from utils.evaluate import get_similarity_score, get_ai_feedback
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume = request.files["resume"]
        desired_title = request.form.get("desired_title", "").strip()

        resume_text = extract_resume(resume)

        score = get_similarity_score(resume_text, desired_title) if desired_title else None
        feedback = get_ai_feedback(resume_text, desired_title)

        return render_template("index.html", score=score, feedback=feedback, show_score=bool(desired_title))

    return render_template("index.html", score=None, feedback=None, show_score=False)


if __name__ == "__main__":
    app.run(debug=True)
