from setuptools import setup, find_packages

setup(
    name="your_project",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "GPy~=1.13.2",
        "numpy",
        "scipy",
        "matplotlib",
        "cvxpy",
        "pandas",
        "mealpy",
        "setuptools",
        "GPyOpt",
        "pyinstrument",
    ],
)