from distutils.core import setup
from setuptools import find_packages

__author__ = "Arthur Fortes"

setup(
    name='CaseRecommender',
    packages=find_packages(),
    version='0.0.20',
    license='GPL3 License',
    description='A recommender systems framework for Python',
    author='Arthur Fortes',
    author_email='fortes.arthur@gmail.com',
    url='https://github.com/ArthurFortes/CaseRecommender',
    download_url='https://github.com/ArthurFortes/CaseRecommender/tarball/0.0.20',
    keywords=['recommender systems', 'framework', 'collaborative filtering', 'content-based filtering'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
    ],
)
