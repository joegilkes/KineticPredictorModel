import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="KineticPredictorModel",
    version="1.0",
    author="Idil Ismail",
    author_email="idil.ismail@warwick.ac.uk",
    description="Package for producting models to predict activation energies of reactions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/idilismail/KineticPredictorModel", 
    project_urls={
        "Bug Tracker": "https://github.com/idilismail/KineticPredictorModel/issues", 
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.20',
        'matplotlib>=3.5',
        'pandas',
        'scipy',
        'seaborn',
        'scikit-learn>=0.21',
        'openbabel>=3.1',
        'rdkit-pypi',
    ]
)
