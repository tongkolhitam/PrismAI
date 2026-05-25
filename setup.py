from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prismai",
    version="1.0.0",
    author="PrismAI Team",
    description="Multi-Modal Content Intelligence Platform powered by MiMo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tongkolhitam/PrismAI",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.25.0",
        "openai>=1.6.0",
        "Pillow>=10.0.0",
        "SpeechRecognition>=3.10.0",
        "pydub>=0.25.1",
        "python-multipart>=0.0.6",
        "structlog>=23.2.0",
        "redis>=5.0.0",
        "cachetools>=5.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Framework :: FastAPI",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
