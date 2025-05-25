def extract_job_text(text):
    # Clean job post, normalize formatting
    return text.replace('\n', ' ').strip()