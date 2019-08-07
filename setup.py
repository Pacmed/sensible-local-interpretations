from distutils.core import setup

setup(
    name='sli',
    version='0.0.1',
    packages=['sli',],
    long_description=open('README.txt').read(),
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
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)