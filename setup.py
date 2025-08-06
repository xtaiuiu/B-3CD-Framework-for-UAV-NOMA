from setuptools import setup, find_packages

setup(
    name="your_project",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy~=1.26.4",
        "scipy~=1.16.1",
        "matplotlib~=3.10.5",
        "cvxpy~=1.7.1",
        "pandas~=2.3.1",
        "mealpy~=3.0.2",
        "setuptools~=80.9.0",
        "GPyOpt~=1.2.6",
        "pyinstrument~=5.0.3",
    ],
)