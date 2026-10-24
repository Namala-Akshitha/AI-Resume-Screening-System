# AI Resume Screening System

A Python project that compares plain-text resumes against a job description using Hugging Face FLAN-T5-small, LangChain, and a Python-based skill-matching fallback.

## Features

- Loads a job description and sample resumes from text files.
- Integrates the FLAN-T5-small model through LangChain.
- Uses a predefined technical-skill vocabulary as a fallback when model extraction is empty or unusable.
- Identifies matching and missing skills.
- Calculates a skill-overlap percentage.
- Includes unit tests for core functionality.

## Technology Stack

- Python
- Hugging Face Transformers
- FLAN-T5-small
- LangChain
- PyTorch
- unittest

## Project Structure

```text
AI-Resume-Screening-System/
├── main.py
├── requirements.txt
├── README.md
├── job_description.txt
├── resumes/
│   ├── strong_resume.txt
│   ├── average_resume.txt
│   └── weak_resume.txt
├── tests/
│   └── test_main.py
└── Screenshots/
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

On Windows Git Bash:

```bash
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

The first run may download the model. An internet connection is required for the initial download.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

The unit tests use a mock model for testing core logic and do not need to download FLAN-T5.

## Fit Score

The score is calculated as:

`(Matching required skills / Total required skills) × 100`

For example, matching 5 out of 6 required skills gives approximately 83%.

## Limitations

- Processes plain-text files rather than PDF or DOCX resumes.
- Uses a predefined skill vocabulary as a fallback, so unlisted skills may be missed.
- Model-generated skill extraction may be incomplete or incorrect.
- The fit score measures skill overlap only; it does not measure overall candidate quality.
- Results should not be used as the sole basis for hiring decisions.

## Future Improvements

- Support PDF and DOCX resumes.
- Expand and validate the skill vocabulary.
- Improve extraction accuracy and test against varied resumes.
- Add a web interface and more comprehensive tests.