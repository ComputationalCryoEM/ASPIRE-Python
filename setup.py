import os
import pip
import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

# setuptools.setup(
#     name="aspire",
#     version="0.1",
#     author="Yoel Shkolnisky, Amit Zinger, Itay Sason, Joakim Andén, Robbie Brook",
#     author_email="devs.aspire@gmail.com",
#     description="Aspire application",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/PrincetonUniversity/ASPIRE-Python",
#     packages=setuptools.find_packages(),
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: custom, see file 'LICENSE'",
#         "Operating System :: OS Independent",
#     ],
# )

os.system('apt install fftw3')


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])


if __name__ == '__main__':
    install('pipenv')

    # create requirements.txt
    os.system('pipenv lock -r > requirements.txt')
    with open('requirements.txt', 'r') as f:
        reqs = f.readlines()
        for req in reqs:
            install(req)
