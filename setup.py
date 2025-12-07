from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anomalyflow",
    version="1.0.0",
    author="Hrishikesh",
    author_email="hrishikesh@example.com",
    description="Advanced Anomaly Detection System with Real-Time Streaming",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HRISHIKESH-hackoff/AnomalyFlow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "pylint>=3.0",
        ],
        "viz": [
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
        ],
    },
)
