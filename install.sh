CONDA_MISSING_ERR_MSG="You're running outside conda environment! please run 'conda create -f environment.yml' first, then activate it with 'conda activate aspire'."

set -e

# make sure conda env is created and that we have finufft repo locally
finufft_dir='./src/finufft'
if [ ! -d $finufft_dir ]
then
  echo "Directory ./src/finufft doesn't exist! please run this script from under ASPIRE-Python."
  echo $CONDA_MISSING_ERR_MSG
  exit 1
fi


# make sure conda virtualenv is activated
python -V 2>/dev/null || (echo $CONDA_MISSING_ERR_MSG ; exit)

# make sure we're running within virutalenv
if [[ `which python` != *"envs/aspire/bin/python"* ]]; then
  echo "You seem to not be running from within aspire environment!"
  echo $CONDA_MISSING_ERR_MSG
  exit
fi

cwd=`pwd`
cd $finufft_dir

# clean whatever finufftpy did so far
git clean -fd

# re/build finufft lib
make lib

# install finufftpy
pip install .

cd $cwd
