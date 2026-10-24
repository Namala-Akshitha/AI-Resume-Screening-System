
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "google/flan-t5-small"

# A fallback vocabulary for skills commonly used in AI/data roles.
KNOWN_SKILLS = [
    "machine learning",
    "deep learning",
    "data visualization",
    "natural language processing",
    "computer vision",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "python",
    "sql",
    "java",
    "c++",
    "fastapi",
    "flask",
    "django",
    "langchain",
    "hugging face",
    "transformers",
    "nlp",
    "statistics",
    "data analysis",
    "data science",
    "power bi",
    "tableau",
    "excel",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "linux",
    "html",
    "css",
    "javascript",
    "rest api",
    "mongodb",
    "mysql",
    "postgresql",
    "chroma",
    "langgraph",
]


def read_file(path: str | Path) -> str:
    """Read a non-empty UTF-8 text file."""
    file_path = Path(path)

    if not file_path.is_absolute():
        file_path = BASE_DIR / file_path

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = file_path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"File is empty: {file_path}")

    return content


def build_chain():
    """Build the local FLAN-T5 model chain."""
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline
    from langchain_core.prompts import PromptTemplate

    model_pipeline = pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        max_new_tokens=100,
        do_sample=False,
    )

    prompt = PromptTemplate.from_template(
        "Extract technical skills explicitly mentioned in this text. "
        "Return only skill names separated by commas. "
        "If there are none, return NONE.\n\nText:\n{text}\n\nSkills:"
    )

    return prompt | HuggingFacePipeline(pipeline=model_pipeline)


def extract_skills(chain, text: str) -> set[str]:
    """Extract skills with the model and use a text-based fallback."""
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    response = str(chain.invoke({"text": text})).strip()
    response = response.removeprefix("Skills:").strip()

    # First, accept recognized skills explicitly returned by the model.
    model_skills = set()

    if response.upper() != "NONE":
        for skill in KNOWN_SKILLS:
            if re.search(
                r"(?<!\w)" + re.escape(skill) + r"(?!\w)",
                response,
                re.IGNORECASE,
            ):
                model_skills.add(skill)

    # If the model returns NONE or unusable text, scan the original text.
    if not model_skills:
        model_skills = {
            skill
            for skill in KNOWN_SKILLS
            if re.search(
                r"(?<!\w)" + re.escape(skill) + r"(?!\w)",
                text,
                re.IGNORECASE,
            )
        }

    return model_skills


def screen_resume(chain, job_description: str, resume_text: str) -> str:
    """Compare extracted skills and calculate skill overlap."""
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    if not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    required_skills = extract_skills(chain, job_description)
    candidate_skills = extract_skills(chain, resume_text)

    matched = sorted(required_skills & candidate_skills)
    missing = sorted(required_skills - candidate_skills)

    if required_skills:
        score = round(len(matched) / len(required_skills) * 100)
        score_text = f"{score}%"
    else:
        score_text = "Unavailable (no required skills detected)"

    return (
        f"Required skills: {', '.join(sorted(required_skills)) or 'None detected'}\n"
        f"Skills found in resume: {', '.join(sorted(candidate_skills)) or 'None detected'}\n"
        f"Matching skills: {', '.join(matched) or 'None'}\n"
        f"Missing skills: {', '.join(missing) or 'None'}\n"
        f"Fit score (based on extracted skill overlap): {score_text}\n"
        "Note: This score measures skill overlap only. Review resumes manually."
    )


def main() -> None:
    job_description = read_file("job_description.txt")

    resume_files = [
        "resumes/strong_resume.txt",
        "resumes/average_resume.txt",
        "resumes/weak_resume.txt",
    ]

    print("AI Resume Screening System")
    print("=" * 60)
    print("Model:", MODEL_NAME)
    print("Scores measure skill overlap, not overall candidate quality.\n")

    chain = build_chain()

    for relative_path in resume_files:
        print(f"Processing: {relative_path}")
        print("-" * 60)

        try:
            resume = read_file(relative_path)
            print(screen_resume(chain, job_description, resume))
        except Exception as exc:
            print(f"Could not screen this resume: {exc}")

        print()


if __name__ == "__main__":
    main()
