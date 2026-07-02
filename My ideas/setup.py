"""
setup.py
--------
Minimal pip-installable setup for the ltr package.

Install (editable / development mode):
    pip install -e .

Install (standard):
    pip install .
"""

from setuptools import setup, find_packages

setup(
    name             = "ltr",
    version          = "0.1.0",
    description      = "Learning-to-Rank with PyTorch — MLP scorer, weighted BCE loss, NDCG evaluation",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = [
        "torch>=1.12",
        "numpy>=1.21",
        "scipy>=1.7",
        "tqdm>=4.62",
    ],
)
