from setuptools import setup, find_packages

setup(
    name="gcp-robotics",
    version="2.0.0",
    description="Golden Codex Protocol 2.0 — GCP-ROBOTICS Core Engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="iAeternum / Metavolve Labs",
    author_email="research@iaeternum.ai",
    url="https://iaeternum.ai/robotics",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0",
        "Pillow>=9.0",
        "numpy>=1.24",
    ],
    extras_require={
        "llm": ["anthropic>=0.18"],
        "ros2": [],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
