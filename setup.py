from distutils.core import setup
import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sli',
    version='0.0.1',
    author="Chandan Singh",
    description="Sensible local interpretations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/sensible-local-interpretations",
    packages=setuptools.find_packages(),
    install_requires=[
        'lime',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',        
        'seaborn',
        'shap',
        'tqdm',
        'lcp'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
