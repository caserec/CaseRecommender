""""
    Setup for Case Recommender

"""

# © 2018. Case Recommender (MIT License)

from distutils.core import setup
from setuptools import find_packages

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'

setup(
    name='CaseRecommender',
    packages=find_packages(),
    version='1.0.0',
    license='MIT License',
    description='A recommender systems framework for Python',

    author='Arthur Fortes',
    author_email='fortes.arthur@gmail.com',

    url='https://github.com/ArthurFortes/CaseRecommender',
    download_url='https://github.com/ArthurFortes/CaseRecommender/tarball/1.0',

    keywords=['recommender systems', 'framework', 'collaborative filtering', 'content-based filtering'],

    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
    ],
)
