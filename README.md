Code accompanying the paper

> Tamir Bendory Yuehaw Khoo, Joe Kileel, Oscar Mickelin, and Amit Singer. "Autocorrelation analysis for cryo-EM with sparsity constraints: improved sample complexity and projection-based algorithms." Proceedings of the National Academy of Sciences 120, no. 18 (2023): e2216507120. https://www.pnas.org/doi/abs/10.1073/pnas.2216507120

Arxiv preprint available at https://arxiv.org/abs/2209.10531.

To run:

1. Download the package

2. Navigate to the downloaded folder in a terminal

3. Run the commands:
```
conda env create -f environment.yml -n sparsekam
conda activate sparsekam
pip install -e .
```

4. Navigate to sparse-Kam

6. Download datafiles from https://github.com/oscarmickelin/sparse-Kam-matrices and place them into the folder sparse-Kam/precomputed_matrices/

6. Run the simulation script with:

```
python -i run_simulation.py
```
