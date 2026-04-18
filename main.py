from dotenv import load_dotenv
load_dotenv()

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Local Hugging Face model
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_new_tokens=300
)


llm = HuggingFacePipeline(pipeline=pipe)

# Prompt
template = """
You are an AI Resume Screening Assistant.

Job Description:
{jd}

Resume:
{resume}

Tasks:
1. Extract skills
2. Compare with job description
3. Give score out of 100
4. Explain score

Output:
Skills:
Matched Skills:
Fit Score:
Explanation:
"""

prompt = PromptTemplate.from_template(template)

chain = prompt | llm

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():

    jd = read_file("job_description.txt")

    resumes = [
        "resumes/strong_resume.txt",
        "resumes/average_resume.txt",
        "resumes/weak_resume.txt"
    ]

    print("AI Resume Screening System")
    print("="*60)

    for file in resumes:
        resume = read_file(file)

        print("\nProcessing:", file)
        print("-"*60)

        result = chain.invoke({
            "jd": jd,
            "resume": resume
        })

        print(result)

if __name__ == "__main__":
    main()
