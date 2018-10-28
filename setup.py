import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aspire",
    version="0.1",
    author="Yoel Shkolnisky, Amit Zinger, Itay Sason, Joakim Andén, Robbie Brook",
    author_email="devs.aspire@gmail.com",
    description="ASPIRE - Algorithms for Single Particle REconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PrincetonUniversity/ASPIRE-Python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License",
        "Operating System :: OS Independent",
    ],
)
