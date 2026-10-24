

import unittest
from unittest.mock import Mock

import main


class TestResumeScreening(unittest.TestCase):
    def test_read_existing_resume(self):
        text = main.read_file("resumes/strong_resume.txt")
        self.assertIn("Python", text)

    def test_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            main.read_file("does-not-exist.txt")

    def test_empty_job_description_rejected(self):
        chain = Mock()

        with self.assertRaises(ValueError):
            main.screen_resume(chain, "  ", "Resume text")

        chain.invoke.assert_not_called()

    def test_empty_resume_rejected(self):
        chain = Mock()

        with self.assertRaises(ValueError):
            main.screen_resume(chain, "Job description", "  ")

        chain.invoke.assert_not_called()

    def test_extract_skills(self):
        chain = Mock()
        chain.invoke.return_value = "Python, SQL, pandas"

        result = main.extract_skills(chain, "Some resume text")

        self.assertEqual(result, {"python", "sql", "pandas"})

    def test_screen_resume_calculates_skill_overlap(self):
        chain = Mock()
        chain.invoke.side_effect = [
            "Python, SQL, pandas",
            "Python, SQL",
        ]

        result = main.screen_resume(
            chain,
            "Job description",
            "Resume text",
        )

        self.assertIn("Matching skills: python, sql", result)
        self.assertIn("Missing skills: pandas", result)
        self.assertIn("Fit score (based on extracted skill overlap): 67%", result)
        self.assertEqual(chain.invoke.call_count, 2)


if __name__ == "__main__":
    unittest.main()
