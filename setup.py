import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="aspire",
    version="0.6.2",
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
        "finufft",
        "importlib_resources>=1.0.2",
        "joblib",
        "jupyter",
        "matplotlib",
        "mrcfile",
        "numpy==1.16",
        "numpydoc==0.7.0",
        "pandas==0.25.3",
        "pyfftw",
        "pillow",
        "pytest",
        "pytest-cov",
        "scipy==1.4.0",
        "scikit-learn",
        "scikit-image==0.16.2",
        "setuptools>=0.41",
        "sphinxcontrib-bibtex",
        "sphinx-rtd-theme>=0.4.2",
        "tqdm",
    ],
    # Here we can call out specific extras,
    #   for example gpu packages which may not install for all users,
    #   or developer tools that are handy but not required for users.
    extras_require={
        "gpu": ["pycuda", "cupy", "cufinufft>=1.1"],
        "dev": [
            "black",
            "bumpversion",
            "check-manifest",
            "flake8>=3.7.0",
            "isort",
            "pyflakes",
            "pydocstyle",
            "pytest-random-order",
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
)
