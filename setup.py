from distutils.core import setup

from setuptools import find_packages

__author__ = "Arthur Fortes"

setup(
    name='CaseRecommender',
    packages=find_packages(),
    version='0.0.4',
    license='GPL3 License',
    description='A recommender systems CaseRecommender for python',
    author='Arthur Fortes da Costa',
    author_email='fortes.arthur@gmail.com',
    url='https://github.com/ArthurFortes/CaseRecommender',
    download_url='https://github.com/ArthurFortes/CaseRecommender/tarball/0.0.3',
    keywords=['recommender systems', 'CaseRecommender', 'collaborative filtering', 'content-based filtering'],
    install_requires=["scipy", "numpy"],
    classifiers=[],
)
