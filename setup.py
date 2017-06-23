try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'Description': 'A simple Finite Element Modelling (FEM) library',
    'author': 'Greg Hamel (MrJarv1s)',
    'url': 'https://github.com/MrJarv1s/FEMur',
    'download_url': 'https://github.com/MrJarv1s/FEMur',
    'author_email': 'hamegreg@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['FEMur'],
    'scripts': [],
    'name': 'FEMur',
}

setup(**config)
