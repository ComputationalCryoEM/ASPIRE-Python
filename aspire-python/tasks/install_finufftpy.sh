CONDA_MISSING_ERR_MSG="You're running outside conda environment! please run 'conda env create -f environment.yml' first, then activate it with 'source activate aspire'."

# make sure you're in the right directory
if [ ! -f 'environment.yml' ] ||  ! grep -q aspire ./environment.yml
then
  echo "Can't find 'environment.yml' for Aspire! Change directory to project's root dir."
  exit 1
fi

# make sure Conda env is activated
pip -V >/dev/null 2>&1 || (echo $CONDA_MISSING_ERR_MSG ; exit)
if [[ `which pip` != *"envs/aspire/bin/pip"* ]]; then

  echo "You seem to not be running from within aspire environment!"

  if [[ "$(conda env list |cut -d ' ' -f1 |grep aspire)" == "aspire" ]]; then
    echo "Please activate aspire environment! ('source activate aspire' or 'conda activate aspire')"
    exit 1
  fi

  echo $CONDA_MISSING_ERR_MSG
  exit 1
fi

# stop here if finufftpy is already installed
python -c "import finufftpy" >/dev/null 2>&1 && echo "finufftpy is already installed." && exit 0

# make sure we have finufft repo locally
finufft_dir='./src/finufft'
if [ ! -d $finufft_dir ]
then
  pip install -e git+https://github.com/flatironinstitute/finufft.git#egg=finufft
fi

# make sure all requirements are met for finufft
required_packages=( gcc g++ libfftw3-bin libfftw3-dev )
for package in "${required_packages[@]}"; do
    dpkg -s $package >/dev/null 2>&1
    if [[ $? -ne 0 ]]; then
        echo "Please install required linux package $package (e.g. 'sudo apt install $package')"
        exit 2
    fi
done

# compile finufft headers and object files
cwd=`pwd`
cd $finufft_dir

# clean whatever finufftpy did so far
make clean
git clean -f -d

# build finufft lib
make lib

# install finufftpy
pip install .

cd $cwd

# test finufftpy can be imported
if [[ "$(python -c 'import finufftpy' 2>&1 && echo 0)" == "0" ]]; then
    echo "OK"
else
    echo "finufftpy could not be installed!"
    echo "Try upgrading the following Linux packages (e.g. 'sudo apt install --upgrade <package>') and run again."
    for package in "${required_packages[@]}"; do
        echo "$package"
    done
fi
