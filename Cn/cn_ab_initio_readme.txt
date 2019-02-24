Cn ab-initio python code (n>=2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
- config_symm.py
	has a class named 'AbinitioSymmConfig' which holds parameters
		is_calc_using_gt: whether or not to calculate common-lines and self-common-lines using ground-truth rotation matrices
		output_folder: relative path for saving reconstructed volume (must exist before running)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
- example_cn.py: example code which loads example (committed) mat files and calls main impl function abinitio_cn.py
	Input: since generating projections from a cn ghost volume is not supported yet, make sure you have the following (committed) mat files in your running directory, or else generate them using generate_cn_images.m
		projs
		rots_gt
		n_symm
		max_shift
		shift_step
		inplane_rot_res_deg
	Output: reconstrcuted volume + various reconstruction stats (mse, angular error)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
- abinitio_cn.py: main implementation function. 
	Input: projection-images (i.e., class averages) + underlying symmetry order + optional parameters
	Output: reconstrcuted volume saved as 
		1. an npy file 'AbinitioSymmConfig.output_folder/vol.npy' and,
		2. translated to a mat file 'AbinitioSymmConfig.output_folder/python_output.mat'
	Since read/write mrc functionality is not supported yet in python, the following matlab code 
	will create an mrc volume from python_output.mat which can then be rendered using chimera:
	********
	path_folder = '/a/home/cc/math/gabipragier/ASPIRE-Python/Cn/output/';
	python_output = load(fullfile(path_folder,'python_output.mat'));
	vol = python_output.vol;
	recon_mrc_fname = fullfile(path_folder,'python_reconst.mrc');
	WriteMRC(vol,1,recon_mrc_fname);
	********
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
TODO: 
	Cn code, n>4: the brute force procedure in which for any pairs of images all pairwise candidate rotations are considered runs extremely slow. Pssible solutions:
		1. Using the gpu might suffice. 
		2. Maybe this is a cache issue and a better organization of the candidate rotatoin matrices will make a difference
	C2 code: 
		- finding two pairs of common-lines in each image (see aspire/abinitio/C2/cryo_clmatrix_c2_gpu.m)
		- voting in which all common-lines are used (see aspire/abinitio/C2/cryo_generateRij.m)
	C_{odd} - when ground-truth rotations are supplied, the mse is sometimes lousy even in a clean setting. NEVERTHELESS: the reconstruction is perfect so there is a minor bug... 