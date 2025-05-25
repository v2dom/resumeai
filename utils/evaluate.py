import os
import openai
import markdown
from flask import Flask, render_template, request, url_for
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

app = Flask(__name__)

@app.route("/resumeai", methods=["GET", "POST"])
def resumeai():
    score = None
    feedback = None

    if request.method == "POST":
        resume = request.files["resume"]
        desired_title = request.form.get("desired_title", "").strip()
        resume_text = resume.read().decode("utf-8")  # Replace this if you use custom extract logic

        if desired_title:
            score = get_similarity_score(resume_text, desired_title)

        feedback_raw = get_ai_feedback(resume_text, desired_title)
        feedback = markdown.markdown(feedback_raw)

        return render_template("index.html", score=score, feedback=feedback, show_score=bool(desired_title))

    return render_template("index.html", score=None, feedback=None, show_score=False)

def get_similarity_score(resume_text, job_title=None):
    if job_title:
        comparison_text = f"This resume is applying for the role: {job_title}."
    else:
        comparison_text = resume_text

    embeddings = model.encode([resume_text, comparison_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(score * 100, 2)

def get_ai_feedback(resume_text, desired_title=None):
    if desired_title:
        prompt = f"""You're an expert technical recruiter.
A resume and a target job title are below.
Assess how qualified the candidate is for the target role based on their experience, skills, education, and certifications.

Format your response with:
1. Summary of Fit
2. Relevant Experience or Achievements
3. Missing Qualifications or Gaps
4. 3 Specific Improvement Suggestions

Desired Job Title: {desired_title}

Resume:
{resume_text}
"""
    else:
        prompt = f"""You're an expert technical recruiter.
A resume is provided below. Assess the overall quality and clarity of the resume, then determine what type of technical role or field the candidate would likely excel in based on their skills, experience, and education.

Format your response with:

1. Overall Resume Quality
2. Recommended Career Path or Technical Field
3. Key Strengths and Capabilities
4. 3 Actionable Suggestions to Improve Resume Impact

Resume:
{resume_text}
"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    markdown_content = response.choices[0].message.content.strip()
    return markdown.markdown(markdown_content)

if __name__ == "__main__":
    app.run(debug=True)
