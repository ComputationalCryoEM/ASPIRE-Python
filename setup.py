from setuptools import setup, find_namespace_packages

setup(
    name='aspire',
    version='0.4.0',

    description='Algorithms for Single Particle Reconstruction',
    url='https://github.com/ComputationalCryoEM/ASPIRE-Python',
    author='Joakim Anden, Yoel Shkolnisky, Itay Sason, Robbie Brook, Vineet Bansal, Junchao Xia',
    author_email='devs.aspire@gmail.com',

    install_requires=[
        'importlib_resources>=1.0.2'
    ],

    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
    package_data={'aspire': ['config.ini']},

    zip_safe=True,
    test_suite='tests',
)
