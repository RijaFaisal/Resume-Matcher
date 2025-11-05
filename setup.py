from setuptools import setup, find_packages

setup(
    name="resume-matcher",
    version="1.0.0",
    description="AI-Powered Resume Matching System using ML",
    author="Resume Matcher Team",
    author_email="team@resumematcher.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pandas",
        "torch",
        "transformers",
        "sentence-transformers",
        "fastapi",
        "uvicorn[standard]",
        "streamlit",
        "mlflow",
        "prometheus-fastapi-instrumentator",
        "evidently",
        "datasets",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "ruff>=0.2.0",
            "black>=24.1.0",
            "isort>=5.13.0",
            "pre-commit>=3.5.0",
            "pip-audit>=2.6.0",
        ],
        "dvc": [
            "dvc>=3.0.0",
            "dvc-s3>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)

