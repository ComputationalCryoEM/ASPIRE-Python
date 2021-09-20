import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="aspire",
    version="0.8.0",
    data_files=[
        ("", ["src/aspire/config.ini"]),
        ("", ["src/aspire/logging.conf"]),
    ],
    include_package_data=True,
    description="Algorithms for Single Particle Reconstruction",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="GPLv3",
    url="https://github.com/ComputationalCryoEM/ASPIRE-Python",
    author="Joakim Anden, Ayelet Heimowitz, Vineet Bansal, Robbie Brook, Itay Sason, Yoel Shkolnisky, Garrett Wright, Junchao Xia",
    author_email="devs.aspire@gmail.com",
    install_requires=[
        "click",
        "finufft==2.0.0",
        "importlib_resources>=1.0.2",
        "joblib",
        "matplotlib",
        "mrcfile",
        "numpy==1.16",
        "pandas==0.25.3",
        "pyfftw",
        "PyWavelets",
        "pillow",
        "scipy==1.4.0",
        "scikit-learn",
        "setuptools>=0.41",
        "tqdm",
    ],
    # Here we can call out specific extras,
    #   for example gpu packages which may not install for all users,
    #   or developer tools that are handy but not required for users.
    extras_require={
        "gpu": ["pycuda", "cupy", "cufinufft==1.2"],
        "dev": [
            "black",
            "bumpversion",
            "check-manifest",
            "flake8>=3.7.0",
            "isort",
            "jupyter",
            "pyflakes",
            "pydocstyle",
            "parameterized",
            "pytest",
            "pytest-cov",
            "pytest-random-order",
            "sphinxcontrib-bibtex",
            "sphinx-rtd-theme>=0.4.2",
            "snakeviz",
            "tox",
        ],
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"aspire": ["config.ini"]},
    zip_safe=True,
    test_suite="tests",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    entry_points={
        "console_scripts": [
            "aspire = aspire.__main__:main_entry",
        ]
    },
)
