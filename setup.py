from setuptools import setup

version = '0.1.0'

classifiers = [
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Information Analysis',
]

dependencies = [
    'numpy>=1.14.5',
    'scipy>=1.1.1'
]

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name = 'flowline',
    version = version,
    description = 'Extract flowlines from a velocity field.',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    author = 'Michael Field',
    author_email = 'mjfield2@outlook.com',
    url = 'https://github.com/mjfield2/flowline_extraction',
    license = 'MIT',
    classifiers = classifiers,
    py_modules = ['flowline'],
    python_requires = '>=3',
    install_requires = dependencies,
)
