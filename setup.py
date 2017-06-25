try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'Description': 'A simple Finite Element Modeling (FEM) library',
    'author': 'Greg Hamel (MrJarv1s)',
    'url': 'https://github.com/MrJarv1s/FEMur',
    'download_url': 'https://github.com/MrJarv1s/FEMur',
    'author_email': 'hamegreg@gmail.com',
    'version': '0.1',
    'install_requires': ['nose', 'numpy', 'scipy', 'sympy'],
    'packages': ['FEMur'],
    'scripts': [],
    'name': 'FEMur',
    'classifiers': [
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering"
    ]
}

setup(**config)
