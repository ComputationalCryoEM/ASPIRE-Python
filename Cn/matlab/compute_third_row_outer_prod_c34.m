function [vijs,viis,npf,projs,refq] = compute_third_row_outer_prod_c34(n_symm,npf,max_shift,shift_step,recon_mat_fname,...
    projs,verbose,is_remove_non_rank1,non_rank1_remov_percent,refq)

if exist('refq','var') && ~isempty(refq)
    is_simulation = true;
else
    refq = [];
    is_simulation = false;
end


if ~exist('verbose','var')
    verbose = 0;
end

if exist('recon_mat_fname','var') && ~isempty(recon_mat_fname)
    do_save_res_to_mat = true;
else
    do_save_res_to_mat = false;
end

[~,n_theta,n_images] = size(npf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 1  : detect a single pair of common-lines between each pair of images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_message('Detecting common-lines');
clmatrix = cryo_clmatrix(npf,n_images,verbose,max_shift,shift_step); 
if do_save_res_to_mat
    log_message('Saving clmatrix to %s',recon_mat_fname);
    save(recon_mat_fname,'clmatrix','-append');
end
if is_simulation
    cl_detection_rate_c3_c4(n_symm,clmatrix,n_theta,refq);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 2  : detect self-common-lines in each image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if n_symm == 3
    is_handle_equator_ims = false;	
else
    is_handle_equator_ims = true;
end
sclmatrix = cryo_self_clmatrix_gpu_c3_c4(n_symm,npf,max_shift,shift_step,verbose,is_handle_equator_ims,refq);
if do_save_res_to_mat
    log_message('Saving sclmatrix to %s',recon_mat_fname);
    save(recon_mat_fname,'sclmatrix','-append');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 3  : calculate self-relative-rotations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Riis = estimate_all_Riis_c3_c4(n_symm,sclmatrix,n_theta,refq);
if do_save_res_to_mat
    log_message('Saving Riis to %s',recon_mat_fname);
    save(recon_mat_fname,'Riis','-append');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 4  : calculate relative-rotations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rijs = cryo_estimate_all_Rijs_c3_c4(n_symm,clmatrix,n_theta,refq);
if do_save_res_to_mat
    log_message('Saving Rijs to %s',recon_mat_fname);
    save(recon_mat_fname,'Rijs','-append');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 5  : inner J-synchronization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_message('Local J-synchronization');
[vijs,viis,im_inds_to_remove,pairwise_inds_to_remove,...
    npf,projs,refq] = local_sync_J_c3_c4(n_symm,Rijs,Riis,npf,projs,is_remove_non_rank1,non_rank1_remov_percent,verbose,refq);
if do_save_res_to_mat
    log_message('Saving npf to %s',recon_mat_fname);
    log_message('Saving vijs to %s',recon_mat_fname);
    log_message('Saving viis to %s',recon_mat_fname);
    log_message('Saving im_inds_to_remove to %s',recon_mat_fname);
    log_message('Saving pairwise_inds_to_remove to %s',recon_mat_fname);
    save(recon_mat_fname,'npf','vijs','viis','im_inds_to_remove','pairwise_inds_to_remove','-append');
end

end



function [sclmatrix,correlations,shifts] = cryo_self_clmatrix_gpu_c3_c4(n_symm,npf,max_shift,shift_step,verbose,is_handle_equator_ims,refq)
% Input parameters:

%   n_symm                  Either 3 (for c_3) or 4 (for c_4)
%   npf                     A 3D array where each image npf(:,:,i) corresponds to the Fourier
%                           transform of projection i.
%   max_shift               The maximum spatial shift that each image is
%                           assumed to have. Default:15
%   shift_step              (Optional) Default:0.5
%   is_handle_equator_ims   (Optional) whether to handle equator images 
%                           seperatelly or not. Defualt:true 
%   equator_res_fact        (Optional) Angular resolution factor that each image
%                           should undergo. For example if res_factor=10 then all
%                           values along angles 0-9 are concatenated (and same for 10-19,20-29, etc)
%   equator_fraction        (Optional) the fraction of input images to decalare as
%                           equator images. Default:0.1
%   refq                    (Optional) A 2d table where the i-th column is the
%                           quaternion that corresponds to the beaming direction of
%                           the i-th image.
% Output parameters:
%   sclmatrix               A 2xnImages table where the i-th column holds
%                           the indexes of the first and third
%                           self-common-line in image i. 
%   correlations            An array of length nImages whose i-th entry
%                           holds the correlation between the two self common-lines that were found
%                           in image i
%   shifts                  An array of length nImages whose i-th entry
%                           holds the shift found for image i

log_message('detecting self-common-lines');
if n_symm ~= 3 && n_symm ~= 4
    error('n_symm may be either 3 or 4');
end

if ~exist('equator_fraction','var')
    equator_fraction = 0.1;
end

if ~exist('equator_res_fact','var')
    equator_res_fact = 10;
end

if ~exist('is_handle_equator_ims','var')
    if n_symm == 4
        is_handle_equator_ims = true;
    else
        is_handle_equator_ims = false;
    end
end

if ~exist('shift_step','var')
    shift_step = 0.5;
end

if ~exist('max_shift','var')
    max_shift = 15;
end

log_message('detecting self-common-lines');

[n_r,n_theta,nImages] = size(npf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: detect self-common-lines for all images
% Step 2: detect all equator images (only for c4)
% Step 3: detect self-common-lines for all equator images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for the angle between self-common-lines. In theory it is
% [60,180] but all 180 apart lines are perfectly correlated so we set it to
% be smaller.
if n_symm == 3
    min_angle_diff = 60*pi/180;
    max_angle_diff = 165*pi/180;
else % i.e. n_symm == 4
    min_angle_diff = 90*pi/180;
    max_angle_diff = 160*pi/180;
end

% the self-common-line matrix holds per image two indeces that represent
% the two self common-lines in the image
sclmatrix = zeros(2,nImages);
correlations = zeros(1,nImages);
shifts = zeros(1,nImages);

% the angular difference between each two self common lines in a given
% image is [90,180], so create a mask.
[X,Y] = meshgrid(1:n_theta/2,1:n_theta);
diff = Y-X;
unsigned_angle_diff = acos(cos(diff.*2*pi./n_theta));

good_diffs = unsigned_angle_diff > min_angle_diff & ...
    unsigned_angle_diff < max_angle_diff;

shift_phases = calc_shift_phases(n_r,max_shift,shift_step);
[n_r_,nshifts] = size(shift_phases);
assert(n_r == n_r_);
g_shift_phases = gpuArray(single(shift_phases));

msg = [];
for i=1:nImages
    
    t1 = clock;
    
    npf_i = npf(:,:,i);
    % each image is conjugate symmetric so need only consider half of the lines
    pi_half = npf_i(:,1:n_theta/2);
    
    g_npf_i   = gpuArray(single(npf_i));
    g_pi_half = gpuArray(single(pi_half));
    
    % generate all shifted copies of the image
    g_pi_half_shifted = zeros([size(pi_half),nshifts],'gpuArray');
    for s=1:nshifts
        g_pi_half_shifted(:,:,s) = bsxfun(@times,g_pi_half,g_shift_phases(:,s));
    end
    
    g_pi_half_shifted = reshape(g_pi_half_shifted,n_r,n_theta/2*nshifts);
    
    % ignoring dc-term
    g_npf_i(1,:) = 0;
    g_pi_half_shifted(1,:) = 0;
    
    % nomalize each ray to be norm 1
    norms = sqrt(sum((abs(g_npf_i)).^2));
    g_npf_i = bsxfun(@rdivide,g_npf_i,norms);
    
    % nomalize each ray to be norm 1
    norms = sqrt(sum((abs(g_pi_half_shifted)).^2));
    g_pi_half_shifted = bsxfun(@rdivide,g_pi_half_shifted,norms);
    
    Corr = g_npf_i.'*g_pi_half_shifted;
    corr = gather(Corr);
    corr = reshape(corr,[n_theta,n_theta/2,nshifts]);
    
    corr = bsxfun(@times,corr,good_diffs);
    
    [correlation,idx] = max(real(corr(:)));
    
    [sc1, scl3, shift] = ind2sub([n_theta, n_theta/2, nshifts],idx);
    sclmatrix(:,i) = [sc1, scl3]';
    correlations(i) = correlation;
    shifts(i) = shift; %TODO: need to "translate" shift 2*max_shift+1 = ...
    
    %%%%%%%%%%%%%%%%%%% debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if verbose 
        t2 = clock;
        t = etime(t2,t1);
        bs = char(repmat(8,1,numel(msg)));
        fprintf('%s',bs);
        msg = sprintf('k=%3d/%3d t=%7.5f',i,nImages,t);
        fprintf('%s',msg);
    end
    %%%%%%%%%%%%%%%%%%% end of debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

log_message('\nself-common-lines correlation (median)=%.2f',median(correlations));

if is_handle_equator_ims
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % step 2  : detect equator images
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     max_shift = 15;
    inds_eq_images = detect_equator_images_orig(npf,5,shift_step,equator_res_fact,...
        equator_fraction,refq);
    
  
    %inds_eq_images = detect_equator_images(npf,max_shift,shift_step,equator_res_fact,...
     %   equator_fraction,refq);
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % Step 3: detect self-common-lines for all equator images
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    log_message('Detecting self-common-lines for equator images');
    % Problem: the angles between the first and third self-cl for equaot images
    % is 180 degrees. As these are conjugate symmetric for any image, these
    % cannot be detected.
    % Observation1: ALL equator images intersect at the global z-axis. So do in
    % particualr any pair of equator images Ri and gRi.
    % Observation2: each pair of equator images have two pairs of common-lines
    % (not four)
    % Solution: find the self-common-lines for each equator image by searching the
    % median common-line with all other equator images
    %
    % clmatrix_eq = commonlines_gaussian(npf_eq,params.max_shift,1); %TODO: extract shift info for debug
    nEquators = numel(inds_eq_images);
%     scl_equators = zeros(2,nEquators);
    clmatrix_eq = cryo_clmatrix_gpu(npf(:,:,inds_eq_images),...
                                    nEquators,0,max_shift,shift_step);
    % we'll take the median line, so make sure all lines are in [0,180]
    clmatrix_eq = mod(clmatrix_eq-1,n_theta/2)+1; 
    sclmatrix(1,inds_eq_images) = median(clmatrix_eq,2)';
    sclmatrix(2,inds_eq_images) = ...
        mod(sclmatrix(1,inds_eq_images)+n_theta/2-1,n_theta)+1; % we **know** they are 180 degrees apart.
end

% analysis against ground-truth
if exist('refq','var') && ~isempty(refq)
    scl_detection_rate(n_symm,sclmatrix,n_theta,refq);
end

end


function inds_eq_images = detect_equator_images_orig(npf,max_shift,shift_step,res_factor,fraction,refq)
%
% Finds the images who's corresponding beaming direction is close to
% (*,#,eps)^T where eps is a small number and *,# are any two numbers. It
% is based on the fact that equator images of a C4 symmetric molecule
% have a reflection symmetry about the horizontal axis. That is, im(:,theta) = im(:,-theta)
%
%
% Input parameters:
%   npf             A 3d table containing the 2-d fourier transform of each
%                   projection image (nImages is the size of the third dimension)
%   max_shift       (Optional) The maximum spatial shift that each image is
%                   assumed to have. Default:15
%   res_factor      (Optional) Angular resolution factor that each image
%                   should undergo. For example if res_factor=10 then all
%                   values along angles 0-9 are concatenated (and same for 10-19,20-29, etc)
%   fraction        (Optional) the fraction of input images 
%                   to decalare as equator images. Defualt=0.1
%   refq            (Optional) A 2d table where the i-th column is the
%                   quaternion that corresponds to the beaming direction of
%                   the i-th image.
%
%
% Output parameters:
%   inds_eq_images  The indexes of equator images found. The number of
%                   indexes is fraction*nImages
%

log_message('Detecting equator images');

if ~exist('removal_frac','var')
    fraction = 0.1;
end

if ~exist('res_factor','var')
    res_factor = 10;
end

[n_r,n_theta,nImages] = size(npf);

nshifts = (2*max_shift+1)^2;
shifted_deltas = zeros(n_r,n_r,nshifts);
i_shift = 1;
for s_x = -max_shift:max_shift
    for s_y  = -max_shift:max_shift
        shifted_deltas(ceil(n_r/2)+s_x,...
            ceil(n_r/2)+s_y,...
            i_shift) = 1;
        i_shift = i_shift + 1;
    end
end
[phases,~] = cryo_pft(shifted_deltas, n_r, n_theta, 'single');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 3: detect the equotor images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

images_score = zeros(1,nImages);
Phases = gpuArray(single(phases));
msg = [];

for i_img = 1:nImages
    
    t1 = clock;
    
    pim = npf(:,:,i_img);
    g_pim = gpuArray(single(pim));
    g_pim_shifted = bsxfun(@times,g_pim,Phases);
    
    g_norms = sqrt(sum((abs(g_pim_shifted)).^2,2));
    g_pim_shifted = bsxfun(@rdivide,g_pim_shifted,g_norms);
%     pim_shifted = gather(g_pim_shifted);
    
    g_pim_shifted(1,:,:) = 0; %ignore the dc term;
    
    % Equator images of a C4 symmetric molecule have the proprty that each
    % of its images has a reflection symmetry about the horizontal axis. That is,
    % im(:,theta) = im(:,-theta)
    
    %flip all images with respect to theta
    g_pim_shifted_flipped = flipdim(g_pim_shifted,2);
    
%     % to find the translation we compute the cross power spectrum
%     g_pim_shifted            = gpuArray(single(pim_shifted));
%     Pim_shifted_flipped      = gpuArray(single(pim_shifted_flipped));
    
    Cross_pwr_spec = fft(g_pim_shifted ,[],2).*conj(fft(g_pim_shifted_flipped,[],2));
    Inv_cross_pwr_spec = ifft(Cross_pwr_spec,[],2);
    inv_cross_pwr_spec = gather(Inv_cross_pwr_spec);
    
    [nr,nc,~] = size(inv_cross_pwr_spec);
    inv_cross_pwr_spec = reshape(inv_cross_pwr_spec,nr*res_factor,nc/res_factor,[]);
    [shifts_score,~] = max(real(sum(inv_cross_pwr_spec,1)));
    images_score(i_img) = max(shifts_score);
    
    %%%%%%%%%%%%%%%%%%% debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    t2 = clock;
    t = etime(t2,t1);
    bs = char(repmat(8,1,numel(msg)));
    fprintf('%s',bs);
    msg = sprintf('k=%3d/%3d t=%7.5f',i_img,nImages,t);
    fprintf('%s',msg);
    
    %%%%%%%%%%%%%%%%%%% end of debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end

[~, sorted_inds] = sort(images_score,'descend');

inds_eq_images = sorted_inds(1:floor(fraction*numel(sorted_inds)));


%%%%%%%%%%%%%%%%%%% debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check how well we did in detecting equator images
if exist('refq','var') && ~isempty(refq)
    assert(nImages == size(refq,2));
    is_eq_gt = false(1,nImages);
    for k = 1:nImages
        Rk_gt  = q_to_rot(refq(:,k))';
        if( abs(Rk_gt(3,3)) < cosd(85))
            is_eq_gt(k) = true;
        end
    end
    TPR = 100*sum(is_eq_gt(inds_eq_images))/sum(is_eq_gt);
    log_message('True-positive-rate of equator images=%.2f%% (#detected_equators/#total_equators)',TPR)
    %     figure; plot(vv(is_eq),'g*'); hold on; plot(vv(~is_eq),'r*');
end

%%%%%%%%%%%%%%%%%%% end of debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% if ismember(nImages,inds_eq_images)
%     printf('\nTop-view image was accidently detected as equator image.');
%     inds_eq_images = inds_eq_images(inds_eq_images~=params.K);
% end

% printf('\nRemoving %d images (%.2f%%) in order eliminate equator images',numel(inds_eq_images),numel(inds_eq_images)/params.K*100);
% masked_projs(:,:,inds_eq_images) = [];
% noisy_projs(:,:,inds_eq_images) = [];
% npf(:,:,inds_eq_images) = [];
% params.K = params.K - numel(find(inds_eq_images));
% if ~params.real_data
%     params.refq(:,inds_eq_images) = [];
% end

end




function inds_eq_images = detect_equator_images(npf,max_shift,shift_step,res_factor,fraction,refq)
%
% Finds the images who's corresponding beaming direction is close to
% (*,#,eps)^T where eps is a small number and *,# are any two numbers. It
% is based on the fact that equator images of a C4 symmetric molecule
% have a reflection symmetry about the horizontal axis. That is, im(:,theta) = im(:,-theta)
%
%
% Input parameters:
%   npf             A 3d table containing the 2-d fourier transform of each
%                   projection image (nImages is the size of the third dimension)
%   max_shift       (Optional) The maximum spatial shift that each image is
%                   assumed to have. Default:15
%   res_factor      (Optional) Angular resolution factor that each image
%                   should undergo. For example if res_factor=10 then all
%                   values along angles 0-9 are concatenated (and same for 10-19,20-29, etc)
%   fraction        (Optional) the fraction of input images 
%                   to decalare as equator images. Defualt=0.1
%   refq            (Optional) A 2d table where the i-th column is the
%                   quaternion that corresponds to the beaming direction of
%                   the i-th image.
%
%
% Output parameters:
%   inds_eq_images  The indexes of equator images found. The number of
%                   indexes is fraction*nImages
%

log_message('Detecting equator images');

if ~exist('removal_frac','var')
    fraction = 0.1;
end

if ~exist('res_factor','var')
    res_factor = 10;
end

[n_r,n_theta,nImages] = size(npf);

% nshifts = (2*max_shift+1)^2;

shift_phases = calc_shift_phases(n_r,max_shift,shift_step);
g_shift_phases = gpuArray(single(shift_phases));
[~,nshifts] = size(shift_phases);

% shifted_deltas = zeros(n_r,n_r,nshifts);
% i_shift = 1;
% for s_x = -max_shift:max_shift
%     for s_y  = -max_shift:max_shift
%         shifted_deltas(ceil(n_r/2)+s_x,...
%             ceil(n_r/2)+s_y,...
%             i_shift) = 1;
%         i_shift = i_shift + 1;
%     end
% end
% [phases,~] = cryo_pft(shifted_deltas, n_r, n_theta, 'single');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 3: detect the equotor images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

images_score = zeros(1,nImages);
% Phases = gpuArray(single(phases));
msg = [];

for i_img = 1:nImages
    
    t1 = clock;
    
    pim = npf(:,:,i_img);
    g_pim = gpuArray(single(pim));
    
    g_pim_shifted = zeros([n_r,n_theta,nshifts],'gpuArray');
    for s=1:nshifts
        g_pim_shifted(:,:,s) = bsxfun(@times,g_shift_phases(:,s),g_pim);
    end
%     g_pim_shifted = bsxfun(@times,g_pim,Phases);
    
    g_norms = sqrt(sum((abs(g_pim_shifted)).^2,2));
    g_pim_shifted = bsxfun(@rdivide,g_pim_shifted,g_norms);
%     pim_shifted = gather(g_pim_shifted);
    
    g_pim_shifted(1,:,:) = 0; %ignore the dc term;
    
    % Equator images of a C4 symmetric molecule have the proprty that each
    % of its images has a reflection symmetry about the horizontal axis. That is,
    % im(:,theta) = im(:,-theta)
    
    %flip all images with respect to theta
    g_pim_shifted_flipped = flipdim(g_pim_shifted,2);
    
%     % to find the translation we compute the cross power spectrum
%     g_pim_shifted            = gpuArray(single(pim_shifted));
%     Pim_shifted_flipped      = gpuArray(single(pim_shifted_flipped));
    
    Cross_pwr_spec = fft(g_pim_shifted ,[],2).*conj(fft(g_pim_shifted_flipped,[],2));
    Inv_cross_pwr_spec = ifft(Cross_pwr_spec,[],2);
    inv_cross_pwr_spec = gather(Inv_cross_pwr_spec);
    
    [nr,nc,~] = size(inv_cross_pwr_spec);
    inv_cross_pwr_spec = reshape(inv_cross_pwr_spec,nr*res_factor,nc/res_factor,[]);
    [shifts_score,~] = max(real(sum(inv_cross_pwr_spec,1)));
    images_score(i_img) = max(shifts_score);
    
    %%%%%%%%%%%%%%%%%%% debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    t2 = clock;
    t = etime(t2,t1);
    bs = char(repmat(8,1,numel(msg)));
    fprintf('%s',bs);
    msg = sprintf('k=%3d/%3d t=%7.5f',i_img,nImages,t);
    fprintf('%s',msg);
    
    %%%%%%%%%%%%%%%%%%% end of debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end

[~, sorted_inds] = sort(images_score,'descend');

inds_eq_images = sorted_inds(1:floor(fraction*numel(sorted_inds)));


%%%%%%%%%%%%%%%%%%% debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check how well we did in detecting equator images
if exist('refq','var') && ~isempty(refq)
    assert(nImages == size(refq,2));
    is_eq_gt = false(1,nImages);
    for k = 1:nImages
        Rk_gt  = q_to_rot(refq(:,k))';
        if( abs(Rk_gt(3,3)) < cosd(85))
            is_eq_gt(k) = true;
        end
    end
    TPR = 100*sum(is_eq_gt(inds_eq_images))/sum(is_eq_gt);
    log_message('True-positive-rate of equator images=%.2f%% (#detected_equators/#total_equators)',TPR)
    %     figure; plot(vv(is_eq),'g*'); hold on; plot(vv(~is_eq),'r*');
end

%%%%%%%%%%%%%%%%%%% end of debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% if ismember(nImages,inds_eq_images)
%     printf('\nTop-view image was accidently detected as equator image.');
%     inds_eq_images = inds_eq_images(inds_eq_images~=params.K);
% end

% printf('\nRemoving %d images (%.2f%%) in order eliminate equator images',numel(inds_eq_images),numel(inds_eq_images)/params.K*100);
% masked_projs(:,:,inds_eq_images) = [];
% noisy_projs(:,:,inds_eq_images) = [];
% npf(:,:,inds_eq_images) = [];
% params.K = params.K - numel(find(inds_eq_images));
% if ~params.real_data
%     params.refq(:,inds_eq_images) = [];
% end

end



function detec_rate = scl_detection_rate(n_symm,sclmatrix,n_theta,refq)
%
% Calculates the detection rate of self-common-lines. It is invariant to a
% possible J-ambiguity (independently for each image), as well as invariant
% to an RigRi<->Rig^{3}Ri ambiguity (independently for each image)
% 
% Input parameters:
%   sclmatrix  A 2xnImages table where the i-th column holds
%              the indexes of the first and third
%              self-common-line in image i. 
%   n_theta    The angular resolution of the images. E.g.,n_theta=360 
%              means that there are 360 lines in each images 
%   refq       A 4-by-n table. The i-th column represent the quaternion of
%              that corresponds to the rotation matrix of the i-th image
%
% Output parameters:
%   detec_rate The detection rate in [0,1] of self common-lines against the
%               ground truth
              
angle_tol_err = 10/180*pi;
% Two issues:
% 1. DOF: cannot tell the difference between first\third self-common-line
% 2. Ambiguity: handedness - At this stage the handedness is independent for each image
nImages = size(sclmatrix,2);
% sclmatrix_correct = zeros(size(sclmatrix));
sclmatrix_gt = detectScls_gt(n_symm,n_theta,refq); % clmatrix_gt is a 2*n matrix

sclmatrix_diff1 = sclmatrix_gt - sclmatrix;
sclmatrix_diff2 = sclmatrix_gt - flipud(sclmatrix); % we cannot (and need not) tell the difference between the first scl and third scl
clmatrix_diff1_angle = sclmatrix_diff1*2*pi./n_theta;
clmatrix_diff2_angle = sclmatrix_diff2*2*pi./n_theta;

% 1. cosine is invariant to 2pi.
% 2. abs is invariant to +-pi diff corresponding to J-ambiguity
clmatrix_diff1_angle_mean = mean(acos(abs(cos(clmatrix_diff1_angle))));
clmatrix_diff2_angle_mean = mean(acos(abs(cos(clmatrix_diff2_angle))));

% we need not tell the difference between the first scl and third scl so
% take the min difference
[min_mean_angle_diff, scl_idx] = min([clmatrix_diff1_angle_mean ; clmatrix_diff2_angle_mean]);
correct_idxs = find(min_mean_angle_diff < angle_tol_err);
detec_rate = numel(correct_idxs)/nImages;
% for debug purposes: just making sure that more or less half of time we
% retrieved the first self-cl and half of the time the third self-cl
scl_dist = histc(scl_idx,1:2)/numel(scl_idx);
log_message('self-common lines detection rate=%.2f%%',detec_rate*100);
log_message('scl-distribution=[%.2f %.2f]',scl_dist);


% find the polar angles of viewing directions of offending rotations
rots_gt = zeros(3,3,nImages);
for k = 1:nImages
    rots_gt(:,:,k) = q_to_rot(refq(:,k))';
end

bad_idxs  = find(min_mean_angle_diff >= angle_tol_err);
if ~isempty(bad_idxs)
    bad_polar_angs = acosd(rots_gt(3,3,bad_idxs));
    nbad = min(numel(bad_polar_angs),5);
    log_message('sample of polar angles failing self-cl [%.2f,%.2f,%.2f,%.2f,%.2f]',...
        bad_polar_angs(1:nbad));
end

end

function sclmatrix_gt = detectScls_gt(n_symm,n_theta,refq)

if n_symm ~= 3 && n_symm ~= 4
    error('n_symm may be either 3 or 4');
end

nImages = size(refq,2);
% we find the first and last self common-lines (c3 and c4s ymmetry)
sclmatrix_gt = zeros(2,nImages);
% correlations_selfcl  = zeros(1,nImages);
rots_gt = zeros(3,3,nImages);
for i = 1:nImages
    rots_gt(:,:,i) = q_to_rot(refq(:,i))';
end

g = [cosd(360/n_symm) -sind(360/n_symm) 0; ...
     sind(360/n_symm)  cosd(360/n_symm) 0; ...
     0                 0  1]; % rotation matrix of 120 or 90 degress around z-axis


for i=1:nImages
    Ri = rots_gt(:,:,i);
    
    % first self common-lines
    U1 = Ri.'*g*Ri;
    c1=[-U1(2,3)  U1(1,3)]';
    idx1 = clAngles2Ind(c1,n_theta);
    sclmatrix_gt(1,i) = idx1;
    % third self common-lines
    U2 = Ri.'*g^(n_symm-1)*Ri;
    c2=[-U2(2,3)  U2(1,3)]';
    idx2 = clAngles2Ind(c2,n_theta);
    sclmatrix_gt(2,i) = idx2;
    
    %npf_k = nonTop_npf(:,:,i);
    %correlations_selfcl(i) = npf_k(:,selfCL_matrix(1,i)).'*npf_k(:,selfCL_matrix(3,i));
end

% if strcmp(params.SCL,'GT') && params.confuse_scl
%     heads = find(round(rand(1,nImages)));
%     sclmatrix_gt(:,heads) = flipud(sclmatrix_gt(:,heads));
% end

end


function Riis = estimate_all_Riis_c3_c4(n_symm,self_cls,n_theta,refq)
%
% Finds the estimates for the self relative rotation matrices.
% Each estimate Rii is a rotation matrix that corresponds to either 
% either R_i g R_i or to R_i g^3 R_i (for c4, i.e. n_symm=4)
% either R_i g R_i or to R_i g^2 R_i (for c3, i.e. n_symm=3)
% 
% Input parameters:
%   n_symm     Either 3 (for c_3) or 4 (for c_4)
%   self_cls   A 2-by-n table where n is the number of images. For c4, i.e. n_symm=4, the i-th
%              column holds the indeces of the first and third self-common-lines in the
%              i-th image (with no particualr order). For c3, i.e. n_symm=3, the i-th
%              column holds the indeces of the first and second self-common-lines in the
%              i-th image (with no particualr order)
%   n_theta    The angular resolution of each image. E.g., n_theta=360 means 
%              that each image has 360 lines 
%   refq       (Optional) A 4-by-n table. The i-th column represent the quaternion of
%              that corresponds to the rotation matrix of the i-th image
%
%
% Output parameters:
%   Riis       A 3-by-3-by-n array. The i-th slice is the 3-by-3 
%              rotation matrix Rii which is an estimate for either 
%              R_i g R_i or R_i g^3 R_i (for c4, i.e. n_symm=4), or
%              R_i g R_i or R_i g^2 R_i (for c2, i.e. n_symm=3)


if n_symm ~= 3 && n_symm ~= 4
    error('n_symm may be either 3 or 4');
end

if exist('refq','var') && ~isempty(refq)
    is_simulation = true;
else
    is_simulation = false;
end

log_message('Computing self relative orientations');

nImages = size(self_cls,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 1: calculate the angle between Pi,Pgi (which is the same as the angle between Pi and Pg^3i)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% note : since we'll take the cosine of the angle we don't really care if
% we retrieve $-twiceAlpha$ or $360-twiceAlpha$ instead of $twiceAlpha$,
idxDiff      = self_cls(2,:)-self_cls(1,:);
% NOTE: DONT use method inds2clAngles. It assumes input to be positive (and
% even larger than zero), calculate therefore explitely the angle
twiceAlpha   = idxDiff*2*pi./n_theta;
cos_twiceAlpha = cos(twiceAlpha);

if n_symm == 4
    % we expect that cos_twiceAlpha lives in [-1,0], but sometimes due to
    % discretization issues we might get values larger than zero. Assert that
    % values are not too large and constrain them to be zero (these actually correspond to top-view images)
    assert(max(cos_twiceAlpha) <= eps,['max(cos_twiceAlpha) is ' num2str(max(cos_twiceAlpha))]);
    cos_twiceAlpha(cos_twiceAlpha>0) = 0;
    
    angles = acos((1+cos_twiceAlpha)./(1-cos_twiceAlpha));
else
    angles = acos(cos_twiceAlpha./(1-cos_twiceAlpha));
    angles = real(angles); % TODO Gabi why this comes out complex???
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% debug code for step 1 (angles between planes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if is_simulation
    
    g = [cosd(360/n_symm) -sind(360/n_symm) 0; ...
     sind(360/n_symm)  cosd(360/n_symm) 0; ...
     0                 0  1]; % rotation matrix of 120 degress around z-axis

    assert(nImages == size(refq,2));
    
    angle_tol_err = 10/180*pi;
    
    angles_gt = zeros(1,nImages);
    for i=1:nImages
        Ri_gt = q_to_rot(refq(:,i))';
        Ri_3_gt = Ri_gt(:,3);
        angles_gt(i) = acos(Ri_3_gt.'*g*Ri_3_gt);
    end
    correct_idxs = find(abs(angles_gt-angles) < angle_tol_err);
    log_message('(P_i,gP_i) success rate=%.2f%%',numel(correct_idxs)/nImages*100);
    % find offending images (the tilt angle of their beaming direction)
    bad_idxs = find(abs(angles_gt-angles) >= angle_tol_err);
    bad_tilt_angles = zeros(1,numel(bad_idxs));
    k = 0;
    for bad_idx=bad_idxs
        k=k+1;
        Ri_bad = q_to_rot(refq(:,bad_idx))';
        bad_tilt_angles(k) = acosd(Ri_bad(3,3));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 2: calculate the remaining euler angles moving from 
% between Pi,Pgi (which is the same as the angle between Pi and Pg^3i)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

aa = (self_cls(1,:)-1)*2*pi/n_theta;
bb = (self_cls(2,:)-1)*2*pi/n_theta + pi; % note the pi: C(gR_i,Ri) = C(R_i,g^3Ri)+pi.

% if ~params.real_data && strcmp(params.SCL,'GT') && params.confuse_scl_J
%     heads = find(round(rand(1,nImages)));
%     angles(heads) = -1*angles(heads);
% end

Riis = ang2orth(-bb, angles, aa);

for i=1:nImages
    [U,~,V] = svd(Riis(:,:,i)); %TODO: Gabi, is this needed? doesn't ang2orth return rotation matrices?
    Riis(:,:,i) = U*V.'; %Rii is the estimation of Ri^TgRi or Ri^Tg^3Ri up to J-conjugacy
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% debug code for step 2 (self relative orientations)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if is_simulation
    
    J = diag([1 1 -1]); % Reflection matrix
    
    errs = zeros(1,nImages);
    errs_idx = zeros(1,nImages);
    for i=1:nImages
        Ri_gt = q_to_rot(refq(:,i))';
        Rii_g_gt  = Ri_gt.'*g*Ri_gt;
        Rii_g2_gt = Ri_gt.'*g^(n_symm-1)*Ri_gt;
        
        Rii = Riis(:,:,i);
        [errs(i),errs_idx(i)] = min([norm(Rii-Rii_g_gt,'fro'),  norm(J*Rii*J-Rii_g_gt, 'fro'),...
            norm(Rii-Rii_g2_gt,'fro'), norm(J*Rii*J-Rii_g2_gt, 'fro')]);
    end
    cls_dist = histc(errs_idx,1:4)/numel(errs_idx);
    errs(bad_idxs) = [];
    log_message('MSE of Rii''s=%.2f',mean(errs.^2));
    log_message('cls_dist=[%.2f %.2f %.2f %.2f]',cls_dist);
    
%     [~,inds_err_Rii] = sort(errs,'descend');
    
%     params.inds_err_Rii = inds_err_Rii;
end

end


function Rijs = cryo_estimate_all_Rijs_c3_c4(n_symm,clmatrix,n_theta,refq)
%
% Estimate a single relative rotation Rij between images i and j. For n_symm==3 (i.e., c3) the
% estimate may correspond to either RiRj, RigRj, or Rig^{2}Rj.
% For n_symm==4 (i.e., c4) the estimate may correspond to either RiRj, RigRj, or Rig^{2}Rj, or Rig^{3}Rj
% 
% Input parameters:
%   n_symm      Either 3 (for c_3) or 4 (for c_4)
%   clmatrix   An n-by-n matrix (where n represents the number of images).
%              The (i,j)-th entry is the index of one of the common lines
%              in image i with image j
%   n_theta    The angular discretization of lines in each image. E.g.,
%              n_theta=360 means that there are 360 lines in each image
%   refq       (Optional) A 4-by-n table. The i-th column represent the quaternion of
%              that corresponds to the rotation matrix of the i-th image
%
% Output parameters:
%   Rijs       A 3x3xn_choose_2 array holding the estimates for the
%              relative orientations


Rijs = cryo_sync3n_estimate_all_Rijs(clmatrix, n_theta);

%%%%%%%%%%%%%%%%%%% debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if exist('refq','var') && ~isempty(refq)
    
g = [cosd(360/n_symm) -sind(360/n_symm) 0; ...
     sind(360/n_symm)  cosd(360/n_symm) 0; ...
     0                 0  1]; % rotation matrix of 120 or 90 degress around z-axis
    
    J = diag([1 1 -1]); % Reflection matrix
    
    nImages = size(refq,2);    
    errs = zeros(1,nchoosek(nImages,2));
    %precompile g,g^2
    gs = zeros(3,3,n_symm);
    for s=0:n_symm-1
        gs(:,:,s+1) = g^s;
    end
    
    for i=1:nImages
        for j=i+1:nImages
            
            ind = uppertri_ijtoind(i,j,nImages);
            Rij = Rijs(:,:,ind);
            
            Ri_gt = q_to_rot(refq(:,i))';
            Rj_gt = q_to_rot(refq(:,j))';
            
            errs_tmp = zeros(1,3);
            for s=1:n_symm
                Rij_gt = Ri_gt.'*gs(:,:,s)*Rj_gt;
                % we are oblivious to a possible J conjugation at this moment
                errs_tmp(s) = min([norm(Rij-Rij_gt,'fro'),norm(J*Rij*J-Rij_gt,'fro')]);
            end
            % we are oblivious to which relative rotation we actualy have
            % in hand, so take the optimal
            errs(ind) = min(errs_tmp);
        end
    end
    
    mse = mean(errs.^2);
    log_message('MSE of Rij''s=%.2f',mse);
end

%%%%%%%%%%%%%%%%%%% end of debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	

end



function [vijs,viis,im_inds_to_remove,pairwise_inds_to_remove,npf,projs,refq,ref_shifts] = local_sync_J_c3_c4(n_symm,Rijs,Riis,npf,...
                                projs,is_remove_non_rank1,remov_percent,verbose,refq,ref_shifts)

% Local J-synchronization of all relative orientations.
%
% Input parameters:
%   n_symm               Either 3 (for c_3) or 4 (for c_4)
%   Rijs                 A 3x3xn_choose_2 array holding the estimates for the
%                        relative orientations. Specifically, each slice Rij
%                        is equal to either $Ri^{T}g^{sij}Rj$, or $JRi^{T}g^{sij}RjJ$
%                        where for n_symm=3 sij is either 0,1 or 2 and for n_symm=4 sij is either 0,1,2 or 3 
%
%   Riis                 A 3x3xn array holding the estimates for the
%                        self relative orientations. Specifically, each slice Rii
%                        is equal to either $Ri^{T}g^{si}Rj$, or $J Ri^{T}g^{si}Rj J$
%                        where for n_symm=3 each si equals 1 or 2, and for
%                        n_symm=4 each si equals 1 or 3
%   npf                  (Optional, required if is_remove_non_rank1==1) 
%                        A 3D array where each image npf(:,:,i) corresponds to the Fourier
%                        transform of projection i.
%   projs                (Optional, required if is_remove_non_rank1==1) 
%                        A 3D array of projection images
%   is_remove_non_rank1  (Optional) zero or one. Whether to remove or not images that
%                        induce too many non-rank-1 matrices. Defualt=1
%   remov_percent
%   refq                 (Optional) A 4-by-n table. The i-th column represent the quaternion of
%                        that corresponds to the rotation matrix of the i-th image
%
%   ref_shifts           (Optional) An nx2 table of shifts that each image
%                        has undergone
%
% Output parameters:
%   vijs           A 3x3xn_choose_2 array where each slice holds an estimate for
%                  the corresponding outer-product vi*vj^{T} between the
%                  third rows of matrices Ri and Rj. Each such estimate
%                  might have a spurious J independently of other estimates
%   viis           A 3x3xn array where the i-th slice holds an estimate for
%                  the outer-product vi*vi^{T} between the
%                  third row of matrix Ri with itself. Each such estimate
%                  might have a spurious J independently of other estimates
%  im_inds_to_remove The image indexes that are removed since they induce
%                    too many non rank-1 matrices
%  pairwise_inds_to_remove 
%   npf            Only if provided in input. Returns the input npf where
%                  all images that correspond to image indexes to remove are removed
%   projs          Only if provided in input. Returns the input projs where
%                  all projections that correspond to image indexes to remove are removed
%   refq           Only if provided in input. Returns the input refq where
%                  all queternions that correspond to image indexes to remove are removed
%   ref_shifts     Only if provided in input. Returns the input ref_shifts where
%                  all shifts that correspond to image indexes to remove are removed

log_message('Local J synchronization');

if n_symm ~= 3 && n_symm ~= 4
    error('n_symm may be either 3 or 4');
end

if exist('is_remove_non_rank1','var') && is_remove_non_rank1 == 1
    if ~exist('npf','var')
        error('variable npf must be given if is_remove_non_rank1==true');
    end
    
    if ~exist('projs','var')
        error('variable projs must be given if is_remove_non_rank1==true');
    end
end

if ~exist('is_remove_non_rank1','var')
    is_remove_non_rank1 = true;
end

if ~exist('non_rank1_remov_percent','var')
    remov_percent = 0.25;
end

nImages = size(Riis,3);
assert(nchoosek(nImages,2) == size(Rijs,3));

nrank1 = 0;
e1 = [1 0 0].';
msg = [];

viis = zeros(3,3,nImages);
if n_symm == 3
    % no matter whether Rii=RigRi or Rii=Rig^{2}Ri (and, possibly
    % also J-conjugated), the sum Rii + Rii.' + eye(3) is rank-1 and is equal to the
    % outor product vi*vj^{T} (or J*vi*vj^{T}*J) of third row of matrices Ri
    % and Rj
    for i=1:nImages
        Rii = Riis(:,:,i);
        viis(:,:,i) = (Rii + Rii.' + eye(3))/3;
    end
else % i.e., n_symm = 4
    % no matter whether Rii=RigRi or Rii=Rig^{3}Ri (and, possibly
    % also J-conjugated), the sum Rii + Rii.' is rank-1 and is equal to the
    % outor product vi*vj^{T} (or J*vi*vj^{T}*J) of third row of matrices Ri
    % and Rj
    for i=1:nImages
        Rii = Riis(:,:,i);
        viis(:,:,i) = 0.5*(Rii + Rii.');
    end
end

vijs         = zeros(3,3,nchoosek(nImages,2));
isRank1_ijs  = zeros(nImages,nImages);
stats        = zeros(1,nchoosek(nImages,2));
J = diag([1 1 -1]); % Reflection matrix
TOL = 1.0E-1; % tollerance value
for i=1:nImages
    
    for j=i+1:nImages
        
        t1 = clock;
        ind = uppertri_ijtoind(i,j,nImages);
        
        Rij = Rijs(:,:,ind);
        
        Rii = Riis(:,:,i);
        Rjj = Riis(:,:,j);
        
        % A rank-1 matrix is attained if and only if the
        % following two conditions are satisfied:
        % 1. matrices Rij,Rii,Rjj either are all J-conjugated or none are at all
        % 2. either (Rii = RigRi and Rjj = RjgRj)
        %        or (Rii = Rig^{3}Ri and Rjj = Rjg^{3}Rj).
        
        % There are 2*2*2 possibilities to check, only one of which is of rank one
        % 1. J-conjugate Rii or not
        % 2. J-conjugate Rjj or not
        % 3. transpose Rjj (note that (RjgRj)^T = Rjg^{3}Rj).
        JRiiJ = J*Rii*J;
        c_Rii = {Rii, Rii, Rii, Rii, JRiiJ, JRiiJ, JRiiJ, JRiiJ};
        
        JRjjJ       = J*Rjj*J;
        Rjj_T       = Rjj.';
        JRjjJ_T = J*Rjj_T*J;
        
        c_Rjj = {Rjj, JRjjJ, Rjj_T, JRjjJ_T, Rjj, JRjjJ, Rjj_T, JRjjJ_T};
        
        vij_cands   = zeros(3,3,8);
        svlaues     = zeros(3,8); % three singular values for each possibility
        is_rank1    = false(1,8);
        score_rank1 = zeros(1,8);
        for s = 1:8
            Rii_cand = c_Rii{s};
            Rjj_cand = c_Rjj{s};
            
            if n_symm == 3
                vij_cand = (Rij + Rii_cand*Rij*Rjj_cand + Rii_cand^2*Rij*Rjj_cand^2)/3;
            else
                vij_cand = (Rij + Rii_cand*Rij*Rjj_cand)/2;
            end
            
            vij_cands(:,:,s) = vij_cand;
            svals = svd(vij_cand);
            % meassure how close are the singular values to (1,0,0)
            is_rank1(s) = abs(svals(1)-1) < TOL && sum(abs(svals(2:3)))/2 < TOL;
            score_rank1(s) = norm(svals-e1,2);
            svlaues(:,s) = svals;
        end
        
        if any(is_rank1 == true)
            isRank1_ijs(i,j) = 1;
            nrank1 = nrank1 + 1; % just for stats puproses
        end
        % even if none is rank-1 we still need to choose the best one
        [~,ii] = min(score_rank1);
        vijs(:,:,ind) = vij_cands(:,:,ii);
        stats(ind) = ii;
        
        %%%%%%%%%%%%%%%%%%% debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if verbose
            t2 = clock;
            t = etime(t2,t1);
            bs = char(repmat(8,1,numel(msg)));
            fprintf('%s',bs);
            msg = sprintf('k1=%3d/%3d k2=%3d/%3d t=%7.5f',i,nImages,j,nImages,t);
            fprintf('%s',msg);
        end
        %%%%%%%%%%%%%%%%%%% end of debug code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
fprintf('\n');

stats_dist = histc(stats,1:8)/numel(stats);
log_message('percentage of rank-1 matrices= %.2f%%', nrank1/nchoosek(nImages,2)*100);
log_message('inner_sync_dist=[%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f]', stats_dist);


if is_remove_non_rank1
    
    isRank1_ijs = isRank1_ijs + isRank1_ijs.';
    
    % find the image indeces whose relative orientations that they induce
    % involve the largest number of matrices that are not rank 1
    [~,inds_err_rank1] = sort(sum(isRank1_ijs),'ascend');
    
    nRemove = floor(remov_percent*nImages);
    im_inds_to_remove = inds_err_rank1(1:nRemove);
    n = numel(im_inds_to_remove);
    log_message('Removing %d out of %d images that induce non rank-1 matrices',...
        numel(im_inds_to_remove),nImages);
    
    % precalculate the number of relative orientations to remove (inclusion exclusion).
    %Each image to remove is part of nImages-1 relative orientations, but we need to
    % subtract the number of relative orientations that comprise of two images
    % that need to be removed.
    sz = n*(nImages-1)-nchoosek(n,2);
    
    pairwise_inds_to_remove = zeros(1,sz);
    k=1;
    for i=im_inds_to_remove        
        for j=1:nImages
            if j<i
                ind = uppertri_ijtoind(j,i,nImages);
            elseif j==i
                continue;
            else
                ind = uppertri_ijtoind(i,j,nImages);
            end
            pairwise_inds_to_remove(k) = ind;
            k = k+1;
        end
    end
    
    npf(:,:,im_inds_to_remove) = [];
    projs(:,:,im_inds_to_remove) = [];
    viis(:,:,im_inds_to_remove) = [];
    vijs(:,:,pairwise_inds_to_remove) = [];
    
    if exist('refq','var') && ~isempty(refq)
        assert(size(refq,2) == nImages);
        refq(:,im_inds_to_remove) = [];
    end
    
    if exist('ref_shifts','var') && ~isempty(ref_shifts)
        assert(size(ref_shifts,1) == nImages);
        ref_shifts(im_inds_to_remove,:) = [];
    end
else
    % nothing to remove
    im_inds_to_remove = [];
    pairwise_inds_to_remove = [];
end

% if ~params.real_data && params.debug && isfield(params,'inds_err_Rii')
%
%     inds_err_Rii = params.inds_err_Rii;
%     top_inds_err_Rii = inds_err_Rii(1:floor(0.2*numel(inds_err_Rii)));
%
%     [~,locs] = ismember(top_inds_err_Rii,im_inds_to_remove);
%     figure; scatter(locs,zeros(1,numel(locs)));
%
%     %     [~,locs] = ismember(top_inds_err_Rii,inds_err_rank1);
%     %     figure; scatter(locs,zeros(1,numel(locs)));
%
% end
end



function [detec_rate,clmatrix_correct] = cl_detection_rate_c3_c4(n_symm,clmatrix,n_theta,refq)
%
% Checks the detection rate of common-lines between 
% images of a c4 or c3 symmetric molecule which is invariant to handedness ambiguity. 
% For each pair of images (i,j) the method checks if the
% single pair of common line found is one of the four pairs of common lines
% that are shared by the images.
% 
% Input parameters:
%   n_symm      either 3 (for c_3) or 4 (for c_4)
%   clmatrix    A n-by-n table where n represens the number of images.  
%               The (i,j) entry contains the index of a
%               common-line in image i betwen images i and j. 
%   n_theta     The angular resolution of common-lines. It is the number
%               of lines in any given image. E.g., n_theta=360 means that
%               each image has 360 lines
%   refq        A 4-by-n table. The i-th column represent the quaternion of
%               that corresponds to the rotation matrix of the i-th image
%
%
% Output parameters:
%   detec_rate        The detection rate (in [0,1]) of common-lines.
%                      For example, detec_rate=1 means that for each pair
%                      of images the common-line found is one of the four
%                      common-lines between the corresponding images
%
%   clmatrix_correct  A boolean matrix of size n-by-n. the (i,j)-th entry
%                     is equal 1 if the common line found is one of the 
%                     four pairs of common lines between images i and j,
%                     and is 0 otherwise. 

if n_symm ~= 3 && n_symm ~= 4
    error('n_symm may be either 3 or 4');
end

angle_tol_err = 10/180*pi; % how much angular deviation we allow for a common-line to have
nImages = size(clmatrix,1);
clmatrix_correct = zeros(size(clmatrix));
% clmatrix_gt is a n*n*n_symm matrix representing the four pairs of common-lines between each two images
clmatrix_gt = find_cl_gt(n_symm,n_theta,refq); 

clmatrix_diff = bsxfun(@minus,clmatrix_gt,clmatrix);
clmatrix_diff_angle = clmatrix_diff*2*pi./n_theta;
% take absolute cosine because of handedness
% there might be +180 independendt diff for each image which at this stage
% hasn't been taken care yet.
nCorrect = 0;
hand_idx = zeros(1,nchoosek(nImages,2));
for i=1:nImages
    for j=i+1:nImages
        ind = uppertri_ijtoind(i,j,nImages);
        diffs_cij = clmatrix_diff_angle(i,j,:);
        diffs_cji = clmatrix_diff_angle(j,i,:);
        min_diff1 = min(acos(cos(diffs_cij))    + acos(cos(diffs_cji)));
        min_diff2 = min(acos(cos(diffs_cij+pi)) + acos(cos(diffs_cji+pi)));
        if min_diff1 < min_diff2
            min_diff = min_diff1;
            hand_idx(ind) = 1;
        else
            min_diff = min_diff2;
            hand_idx(ind) = 2;
        end
        if min_diff < 2*angle_tol_err
            nCorrect  = nCorrect+1;
            clmatrix_correct(i,j) = 1;
            clmatrix_correct(j,i) = 1;
        end
    end
end

cl_dist = histc(hand_idx,1:2)/numel(hand_idx);
detec_rate = nCorrect/(nImages*(nImages-1)/2);
log_message('common lines detection rate=%.2f%%',detec_rate*100);
log_message('cl_J_dist=[%.2f %.2f]',cl_dist);

end

function clmatrix_gt = find_cl_gt(n_symm,n_theta,refq)

if n_symm ~= 3 && n_symm ~= 4
    error('n_symm may be either 3 or 4');
end

nImages = size(refq,2);
clmatrix_gt = zeros(nImages,nImages,n_symm);

g = [cosd(360/n_symm) -sind(360/n_symm) 0; ...
     sind(360/n_symm)  cosd(360/n_symm) 0; ...
     0                 0  1]; % rotation matrix of 120 or 90 degress around z-axis

gs = zeros(3,3,n_symm);
for s=0:n_symm-1
    gs(:,:,s+1) = g^s;
end

for i=1:nImages
    for j=i+1:nImages
        Ri = q_to_rot(refq(:,i))';
        Rj = q_to_rot(refq(:,j))';
        for s=0:n_symm-1
            U = Ri.'*gs(:,:,s+1)*Rj;
            c1 = [-U(2,3)  U(1,3)]';
            c2 = [ U(3,2) -U(3,1)]';
            
            idx1 = clAngles2Ind(c1,n_theta);
            idx2 = clAngles2Ind(c2,n_theta);
            
%             if strcmp(params_simul.CL,'GT') && params_simul.confuse_cl_J
%                 if round(rand)==1
%                     % j-conjugating amounts at choosing the antipodal
%                     % common-lines in each image
%                     idx1 = mod(idx1+n_theta/2-1,n_theta)+1;
%                     idx2 = mod(idx2+n_theta/2-1,n_theta)+1;
%                 end
%             end
            clmatrix_gt(i,j,s+1) = idx1;
            clmatrix_gt(j,i,s+1) = idx2;
        end
    end
end


end