# ASPIRE-Python

## Installation
#### Requirements (Linux)
Conda for **Python3** (either 
[Anaconda](https://www.anaconda.com/download/#linux)
or 
[Miniconda](https://conda.io/miniconda.html), click on the right distribution to view Conda's installation instructions)

- gcc: `sudo apt install --upgrade gcc`
- g++: `sudo apt install --upgrade g++`
- fftw3: `sudo apt install libfftw3-bin  libfftw3-dev`

These commands were tested on Ubuntu 16.04. On other Linux distro's, use the substitute command for 'apt install' (yum, apt-get, etc.)

#### Creating Conda environment
Run `conda env create -f environment.yml`

###### Common errors:
- `"Prefix already exists .../envs/aspire"` - please delete the directory shown and rerun.

#### Activating Conda environment
Run: `source activate aspire`
Depending on your Conda distribution, in some cases you should run: `conda activate aspire`

To deactivate the environment run: `source deactivate` (or `conda deactivate`, respectively)

#### Installing finufftpy
As you're in the root directory (under ASPIRE-Python, so you can `ls` and see _aspire.py_)
run `./install.sh`

Assuming no errors, you can now use the Aspire tool.

## Usage
Aspire is command-line-interface (CLI) which allows you to run atomic actions on stack of CRYO projections (MRC files).
The Pythonic version will simply be run with: python aspire.py
As you run that, you'll get a usage help message, showing you the various available commands.

It is important to note that at each command level you should supply the flags/options of that level, not more, not less.
E.g. you want to run Aspire in debug mode and with maximum verbosity:
`python aspire.py --debug -v crop demmo.mrc 42` (crop stack in demmo.mrc file to projections of 42x42 px squares)

If you place the '-v' or '--debug' in the end of the line, Aspire would assume these are flags for crop command (which aren't clear and Aspire won't run).

###### Common errors:
- `ModuleNotFoundError: No module named 'click'` -  You're outside Conda's environment! Run 'source activate aspire' (or create a new env if you skipped the previous step 'Creating Conda environment'.



