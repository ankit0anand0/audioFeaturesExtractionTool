"""
Author: Ankit Anand
Created on: 12/12/24
"""

from setuptools import setup, find_packages

setup(
	name="afet-pkg",
	version="2024.0.0",
	packages=find_packages(where='.', include=['src*']),
	python_requires = ">=3.12.0, <3.13.0",
	
	author="Ankit Anand",
	author_email="ankit0.anand0@gmail.com",
	install_requires = [	],
)