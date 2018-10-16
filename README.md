![alt ](http://spr.math.princeton.edu/sites/spr.math.princeton.edu/files/ASPIRE_1.jpg)

ASPIRE-Python
=============

Installation
------------

#### Requirements (Linux)
Conda for **Python3** (either 
[Anaconda](https://www.anaconda.com/download/#linux)
or 
[Miniconda](https://conda.io/miniconda.html), click on the right distribution to view Conda's installation instructions)

#### Creating Conda environment
Run `conda env create -f environment.yml`

###### Common errors:
- `"Prefix already exists .../envs/aspire"` - please delete the directory shown and rerun.

#### Activating Conda environment
Run: `source activate aspire`  
Depending on your Conda distribution, in some cases you should run: `conda activate aspire`

To deactivate the environment run: `source deactivate` (or `conda deactivate`, respectively)

#### Installing finufftpy (optional)
For using certain commands, such as *classify*, you'll need to compile and install *finufftpy*:  
Make sure you're in the root directory (under ASPIRE-Python, so you see _aspire.py_ when running `ls`)  
Run `make finufftpy`  
You might need to install some prerequisites, follow the instructions.

###### common errors
If you skipped any of the previous steps, the script will complain about the missing part and terminate.
If you encounter problems during compilation process, please upgrade the following Linux packages:
- gcc: `sudo apt install --upgrade  gcc  g++`
- fftw3: `sudo apt install  libfftw3-bin  libfftw3-dev`

These commands were tested on Ubuntu 16TLS. For other Linux distros please use the substitute command for 'apt install' (yum, apt-get, brew, etc.)

#### Downloading data files (optional)
For some commands, you'll need to have certain binary files available for Aspire.  
To download them simply run `make download`  
You can choose to firstly not download binaries and then Aspire would ask you to do so before running any commnad which needs them.


Assuming no errors, you can now use Aspire tool.

## Usage
Aspire is a command-line-interface (CLI) application allowing you to run atomic actions on stack of CRYO projections (MRC files).
To invoke the tool simply run: `python aspire.py`.  
You'll see a help message showing you the various available commands.

#### Arguments, options and flags
Arguments are mandatory inputs. When running 'compare' command, you must provide 2 MRC files to compoare.  
Options are, like their name suggests, optional inputs. For example, _aspire.py_ accepts option '-v 2' for setting verbosity level to 2. All options have a default value set for them.  
Flags are optional values which tells Aspire to activate/deactivate certain behaviour. A good example would be '_--debug_'. All flags also have a default value pre-set for them, '--no-debug' in case of the 'debug' flag.  

Aspire CLI is built in layers. A layer is basically a command which can be followed by another command.
The most basic command is `aspire.py` itself, the base layer. It accepts its own flags such as '_--help_', '_--debug_' or '_-v N_'. Each of these optional flags will be directed into the root layer.

Then we can call Aspire with a consequtive subcommand such as '_compare_', and provide another layer of arguments, options and flags. In case of '_compare_' these can be '_compare MRC1 MRC2 --max-error=0.123_'.  

It is important to note that each command has to followed by its own options/arguments of that specific level, not more, not less.

##### Examples
1. If you want to view the help message for each command, please place '--help' **after** the command:  
`python aspire.py compare --help`.  
`python aspire.py --help compare` will only present the help message for the highest layer.

2. Crop a stack of projections of an mrc file to squares of 42x42 px, in debug mode and with maximum verbosity:  
`python aspire.py --debug -v 3 crop demmo.mrc 42`

3. If you place the '-v' or '--debug' in the end of the line, Aspire would assume these are flags for crop command (which aren't clear and Aspire won't run). For `python aspire.py compare --debug` you'll get a message '_Error: no such option: --debug_'

###### Common errors:
- `ModuleNotFoundError: No module named 'click'` -  You're outside Conda's environment! Run 'source activate aspire' (or create a new env if you skipped the previous step 'Creating Conda environment'.
