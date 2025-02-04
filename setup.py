from setuptools import find_packages, setup

LATEST_VERSION = "0.10.10"

exclude_packages = [
    "selenium",
    "webdriver",
    "fastapi",
    "fastapi.*",
    "uvicorn",
    "jinja2",
    "icis-researcher",
    "langgraph"
]

with open(r"README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    reqs = [line.strip() for line in f if not any(pkg in line for pkg in exclude_packages)]

setup(
    name="icis-researcher",
    version=LATEST_VERSION,
    description="ICIS Researcher is an autonomous agent designed for comprehensive web research on any task",
    package_dir={'icis_researcher': 'icis_researcher'},
    packages=find_packages(exclude=exclude_packages),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="James Melvin",
    author_email="james.melvin@lexisnexisrisk.com",
    license="MIT",
    install_requires=reqs,


)