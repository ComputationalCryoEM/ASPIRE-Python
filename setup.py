import os
from setuptools import setup, find_namespace_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
	
setup(
    name='aspire',
    version='0.5.3',

    description='Algorithms for Single Particle Reconstruction',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',	
    license="GPLv3",
    url='https://github.com/ComputationalCryoEM/ASPIRE-Python',
    author='Joakim Anden, Yoel Shkolnisky, Itay Sason, Robbie Brook, Vineet Bansal, Junchao Xia',
    author_email='devs.aspire@gmail.com',

    install_requires=[
        'importlib_resources>=1.0.2',
        'mrcfile',
        'python-box',
        'finufftpy',
        'console_progressbar',
        'pyfftw',
        'click',
        'matplotlib',
        'numpy',
        'pandas>=0.23.4',
        'scipy==0.19.1',
        'tqdm',
        'scikit-learn',
        'scikit-image'
    ],

    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
    package_data={'aspire': ['config.ini']},

    zip_safe=True,
    test_suite='tests',
	
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ]
)
