function [clstack,corrstack,shift_equations,shift_equations_map,clstack_mask]=...
    cryo_clmatrix_cpu_fast_v2(pf,NK,verbose,max_shift,shift_step,map_filter_radius,...
    ref_clmatrix,ref_shifts_2d)
%
%
%   Generate common-lines matrix for the Fourier stack pf.
%
% Input parameters:
%   pf       3D array where each image pf(:,:,k) corresponds to the Fourier
%            transform of projection k.
%   NK       For each projection find its common-lines with NK other
%            projections. If NK is less than the total number a projection,
%            a random subset of NK projections is used. Default: n_proj. 
%   verbose  Bitmask of debugging level.Bits:
%           0   silent
%           1   One line progress message (not written to log) (Default)
%           2   Print detailed debug messages
%           4   Draw common-line debugging plots
%           8   Draw shift estimation debugging plots
%   max_shift       Maximal 1D shift (in pixels)  to search between
%       common-lines. Default: 15.
%   shift_step      Resolution of shift estimation in pixels. Note that
%        shift_step can be any positive real number. Default: 1.
%   map_filter_radius      If nonzero, the common line between a pair
%       images is detected not by the pair of lines with the highest 
%       correlation, but rather the pair of lines that both them and their
%       sorroundings given the best match. The radius for comparison is
%       determined by the value of map_filter_radius (Default 0).
%   ref_clmatrix    True common-lines matrix (for debugging).
%   ref_shifts_2d   True 2D shifts between projections (for debugging).
%
% Returned variables:
%
%   clstack     Common lines matrix. (k1,k2) and (k2,k1) contain the index
%       of the common line of projections k1 and k2. (k1,k2)  contains the
%       index of the common line in the projection k1. (k2,k1) contains the
%       index of the common line in k2. 
%   corrstack   The correlation of the common line between projections k1
%       and k2. Since corrstack is symmetric, it contain entries only above
%       the diagonal. corrstack(k1,k2) measures how ''common'' is the between
%       projections k1 and k2. Small value means high-similariry.
%   shift_equations  System of equations for the 2D shifts of the
%       projections. This is a sparse system with 2*n_proj+1 columns. The
%       first 2*n_proj columns correspond to the unknown shifts dx,dy of
%       each projection. The last column is the right-hand-side of the
%       system, which is the relative shift of each pair of common-lines.
%   shift_equations_map   2D array of size n_proj by n_proj. Entry (k1,k2)
%       is the index of the equation (row number) in the array
%       "shift_equations" that corresponds to the common line between
%       projections k1 and k2. shift_map is non-zero only for k1<k2. 
%   clstack_mask  If ref_clmatrix is given, this array discribes which
%       common-lines were identified correcly. It is of size
%       n_projXn_projs, where entry (k1,k2) is 1 if the common-line between
%       projections k1 and k2 was correctly identified, and 0 otherwise.
%       This matrix will be non-zero only if bit 2 of verbose it set.
%
% Future version comment:
% The function assumes that the common-line is the pair of lines
% with maximum correlation. When the noise level is high, the
% maximal correlation usually does not correspond to the true
% common-line. In such cases, we would like to take several
% candidates for the common-line. See cryo_clmatrix_v4 for how
% to handle multiple candidates for common-line.
%
% Revisions:
%   02/03/09  Filename changed from cryo_clmatrix_v6.m to cryo_clmatrix.m.
%   02/04/09  Cleaning the code.
%   02/19/09  Maximal allowed deviation between common lines to be
%             considered as match was changed from 5 to 10 degrees.
%   02/24/09  Normalize the array pf only once (improves speed).
%   03/05/09  cryo_clmatrix_v3 created from cryo_clmatrix_v2
%   03/05/09  Exploit the conjugate symmetry of the Fourier transform to
%             compute correlations of length rmax instead of 2*rmax-1. This
%             gives a factor of 2 in performance.
%   15/06/13  Replace repmat with bsxfnu when comparing shifted lines. This
%             should be faster and makes the code easier to port to GPU.
%   28/2/17   Rename cryo_clmatrix to cryo_clmatrix_cpu.

msg=[];

T=size(pf,2);

if mod(T,2)~=0
    error('n_theta must be even');
end

% pf is of size n_rxn_theta. Convert pf into an array of size
% (2xn_r-1)xn_theta, that is, take then entire ray through the origin, but
% thake the angles only up PI.
% This seems redundant: The original projections are real, and thus
% each ray is conjugate symmetric. We therefore gain nothing by taking
% longer correlations (of length 2*n_r-1 instead of n_r), as the two halfs
% are exactly the same. Taking shorter correlation would speed the
% computation by a factor of two.
pf=[flipdim(pf(2:end,T/2+1:end,:),1) ; pf(:,1:T/2,:) ];

% XXX The PCA should not be done here. Move outside the common lines
% XXX matrix.
% Project all common lines on the first 15 principal components.
% % % nPCs=30;
% % % spf=reshape(pf,size(pf,1),size(pf,2)*size(pf,3));
% % % 
% % % m=mean(spf,2);
% % % spf1=zeros(size(spf));
% % % for k=1:size(spf,2);
% % %     spf1(:,k)=spf(:,k)-m;
% % % end
% % % spf=spf1;
% % % 
% % % [U,S]=svd(spf);
% % % U=U(:,1:nPCs);
% % % pf_sav=pf;
% % % spf=U*U'*spf;
% % % pf=reshape(spf,size(pf,1),size(pf,2),size(pf,3));
% % % 
% % % log_message('Number of PCs = %d',nPCs);

n_theta=size(pf,2);
n_proj=size(pf,3);

%% Check input parameters and set debug flags.
if (nargin<2) || (NK==-1)
    NK=n_proj; % Number of common-line pairs to compute for each projection
end

if ~exist('verbose','var')
    verbose=1;
end

if ~exist('max_shift','var')
    max_shift=15; % Maximal shift between common-lines in pixels. The 
                  % shift  is from -max_shift to max_shift. 
end

if ~exist('shift_step','var')
    shift_step=1.0; % Resolution of shift estimation in pixels.
end
n_shifts=ceil(2*max_shift/shift_step+1); % Number of shifts to try.

if ~exist('map_filter_radius','var')
    map_filter_radius=0;
end

if ~exist('ref_clmatrix','var') || isempty(ref_clmatrix)
    ref_clmatrix=0;
end

if ~exist('ref_shifts_2d','var') || isempty(ref_shifts_2d)
    ref_shifts_2d=0;
end


% Set flag for progress and debug messages
verbose_progress=0;
verbose_detailed_debugging=0;
verbose_plot_cl=0;
verbose_plot_shifts=0;

if bitand(verbose,1)
    verbose_progress=1;
end;

found_ref_clmatrix=0;
if ~isscalar(ref_clmatrix) 
    found_ref_clmatrix=1;
else
    if verbose>0 
        log_message('Reference clmatrix not found');
    end
end

found_ref_shifts=0;
if ~isscalar(ref_shifts_2d)
    found_ref_shifts=1;
else
    if verbose>0
        log_message('Reference shifts not found');
    end
end

if bitand(verbose,2)
        verbose_detailed_debugging=1;
        verbose_progress=0;
end;

if bitand(verbose,4) 
    if isscalar(ref_clmatrix) 
        log_message('Common-lines plots not available. Reference clmatrix is missing\n');
    end
    verbose_plot_cl=1;
end;

if bitand(verbose,8) 
    if isscalar(ref_clmatrix) || isscalar(ref_shifts_2d)
        log_message('Only partial information will be plotted. Reference clmatrix or shifts are missing\n');
    end
    verbose_plot_shifts=1;
end;

if verbose~=0
    log_message('Verbose mode=%d',verbose);
end

%%

clstack=zeros(n_proj,n_proj);      % Common lines-matrix.
corrstack=zeros(n_proj,n_proj);    % Correlation coefficient for each common-line.
clstack_mask=zeros(n_proj,n_proj); % Which common-lines were correctly identified.

refcorr=zeros(n_proj,n_proj); % Correlation between true common-lines.
thetadiff=zeros(n_proj,n_proj); % Angle between true and estimated common lines.

%% Allocate variables used for shift estimation

shifts_1d=zeros(n_proj,n_proj);     % Estimated 1D shift between common-lines. 

ref_shifts_1d=zeros(n_proj,n_proj); % True shift along the common-line 
    % between each pair of projections. Computed from the reference 2D
    % shifts. 
     
shift_estimation_error=zeros(n_proj,n_proj); % The difference between the 
    % estimated shift along each common line and the true shift.

% Based on the estimated common-lines, construct the equations for
% determining the 2D shift of each projection. The shift equations are
% represented using a sparse matrix, since each row in the system contains
% four non-zeros (as it involves exactly four unknowns).
% The variables below are used to construct this sparse system. The k'th
% non-zero element of the equations matrix is stored at index 
% (shift_I(k),shift_J(k)).
shift_I=zeros(4*n_proj*NK,1);  % Row index for sparse equations system.

shift_J=zeros(4*n_proj*NK,1);  % Column index for sparse equations system.

shift_eq=zeros(4*n_proj*NK,1); % The coefficients of the center estimation
    % system ordered as a single vector.
     
shift_equations_map=zeros(n_proj); % Entry (k1,k2) is the index of the 
    % euqation for the common line of projections k1 and k2. 
                               
shift_equation_idx=1;  % The equation number we are currently processing.
shift_b=zeros(n_proj*(n_proj-1)/2,1);   % Right hand side of the system.
dtheta=pi/n_theta; % Not 2*pi/n_theta, since we divided n_theta by 2 to 
    % take rays of length 2*n_r-1.
                                      
if verbose>0
    log_message('Shift estimation parameters: max_shift=%d   shift_step=%d',max_shift,shift_step);
end

if verbose>0
    log_message('map_filter_radius = %d',map_filter_radius);
end


%% Debugging handles and variables

matched_cl=0;  % How many times the estimated common-line is close (to 
    % within a prescribed tolerance) to the true common-line.
               % Used for debugging.

if verbose_plot_shifts
    if found_ref_clmatrix
        h1=figure;
    end
    h2=figure;
end

if verbose_plot_cl && verbose_detailed_debugging
    h3=figure;
end
           
%% Search for common lines between pairs of projections

% Construct filter to apply to each Fourier ray.                   
rmax=(size(pf,1)-1)/2;    
rk=-rmax:rmax;
rk=rk(:);
H=sqrt(abs(rk)).*exp(-rk.^2/(2*(rmax/4).^2)); 
H=repmat(H(:),1,n_theta);  % Filter for common-line detection.

% Bandpass filter and normalize each ray of each projection.
% XXX We do not override pf since it is used to debugging plots below. Once
% XXX these debugging plots are removed, replace pf3 by pf. This will save
% XXX a lot of memory. 
pf3=pf;
for k=1:n_proj
    proj=pf(:,:,k);
    proj=proj.*H;
    proj(rmax:rmax+2,:)=0;
    proj=cryo_raynormalize(proj);
    pf3(:,:,k)=proj;
end

rk2=rk(1:rmax);
for k1=1:n_proj
    
    n2=min(n_proj-k1,NK);
    subsetK2=sort(randperm(n_proj-k1)+k1);
    subsetK2=subsetK2(1:n2); % Select a subset of at most NK projections 
        % with which to search for common-lines with projection k1. 
   
    proj1=pf3(:,:,k1);
    P1=proj1(1:rmax,:);  % Take half ray plus the DC
    P1_real = real(P1);
    P1_imag = imag(P1);
    
    % Make sure the DC component is zero. This is assumed  below in
    % computing correlations.
    if norm(proj1(rmax+1,:))>1.0e-13
        error('DC component of projection is not zero');
    end
    
    for k2=subsetK2
        
        t1=clock;                       
        proj2=pf3(:,:,k2); % proj1 and proj2 are both normalized to unit norm.
        P2_flipped=conj(proj2(1:rmax,:));
        
        if norm(proj2(rmax+1,:))>1.0e-13
            error('DC component of projection is not zero');
        end


        %%%%%%%%%%%% Beginning of debug code %%%%%%%%%%%%
        if verbose_plot_shifts && found_ref_clmatrix
            % Plot the signals that correspond to the true common-line.
            % This allows to appreciate visually that this is indeed the
            % common line, as well as the shift between the signal. Note
            % that the plotted signal are not filtered.
            xx=-rmax:1:rmax;
            if ref_clmatrix(k1,k2)<=n_theta
                pf1=pf(:,ref_clmatrix(k1,k2),k1);
            else
                pf1=pf(:,ref_clmatrix(k1,k2)-n_theta,k1);
                pf1=flipud(pf1);
            end
            
            p1=enforce_real(cfft(cryo_raynormalize(pf1)));
            v1=triginterp(p1,1);
            
            if ref_clmatrix(k2,k1)<=n_theta
                pf2=pf(:,ref_clmatrix(k2,k1),k2);
            else
                pf2=pf(:,ref_clmatrix(k2,k1)-n_theta,k2);
                pf2=flipud(pf2);
            end

            p2=enforce_real(cfft(cryo_raynormalize(pf2)));
            v2=triginterp(p2,1);
            
            figure(h1);
            plot(xx,real(v1),'-b');
            hold on;
            plot(xx,real(v2),'-r');
            hold off;
            % We always measure by how much we need to shift the blue
            % signal (v1). If we need to shift it to the right then shift is
            % negative and dx is positive.
            legend('Shifted signal','Fixed Signal')
                        
        end
        %%%%%%%%%%%% End of debug code %%%%%%%%%%%%

        % Find the shift that gives best correlation.
        for shiftidx=1:n_shifts
            shift=-max_shift+(shiftidx-1)*shift_step;
            shift_phases=exp(-2*pi*sqrt(-1).*rk2.*shift./(2*rmax+1)); 
                                    
            P2_shifted_flipped=bsxfun(@times,P2_flipped,shift_phases);
            
            part1=P1_real.'*real(P2_shifted_flipped);
            part2=P1_imag.'*imag(P2_shifted_flipped);
            % Compute correlations in the positive r direction           
            C1=2*(part1-part2);
            C2=2*(part1+part2);
            % Compute correlations in the negative r direction
            
            C = [C1,C2];
   
            if map_filter_radius > 0
                C = cryo_average_clmap(C, map_filter_radius);
            end
            
            [sval,sidx]=max(C(:));            
           
            improved_correlation=0; % Indicates that we found a better 
                % correlation than previously known.
            
            if sval>corrstack(k1,k2)
                [cl1,cl2]=ind2sub([n_theta 2*n_theta],sidx);
                clstack(k1,k2)=cl1;
                clstack(k2,k1)=cl2;
                corrstack(k1,k2)=sval;
                shifts_1d(k1,k2)=shift;
                improved_correlation=1;
            end
            
            
% The above code is a compact version of the following one.            
%             % The best match is the maximum among C1 and C2.
%             [sval1,sidx1]=max(C1(:));
%             [sval2,sidx2]=max(C2(:));
%                                       
%             improved_correlation=0; % Indicates that we found a better 
%                 % correlation than previously known.
%             
%             if sval1>sval2
%                 if sval1>corrstack(k1,k2)
%                     [cl1,cl2]=ind2sub([n_theta n_theta],sidx1);
%                     clstack(k1,k2)=cl1;
%                     clstack(k2,k1)=cl2;
%                     corrstack(k1,k2)=sval1;
%                     shifts_1d(k1,k2)=shift;
%                     improved_correlation=1;
%                 end
%             else
%                 if sval2>corrstack(k1,k2)
%                     [cl1,cl2]=ind2sub([n_theta n_theta],sidx2);
%                     clstack(k1,k2)=cl1;
%                     clstack(k2,k1)=cl2+n_theta;
%                     corrstack(k1,k2)=sval2;
%                     shifts_1d(k1,k2)=shift;
%                     improved_correlation=1;
%                 end
%             end
            
            if verbose_detailed_debugging && found_ref_clmatrix && found_ref_shifts
                % Compute and store the correlation between the true
                % common-lines.
                l1=ref_clmatrix(k1,k2);
                l2=ref_clmatrix(k2,k1);

                if l1<=n_theta
                    r1=proj1(:,l1);
                else
                    r1=proj1(:,l1-n_theta);
                    r1=flipud(r1);
                end

                if l2<=n_theta
                    r2=proj2(:,l2);
                else
                    r2=proj2(:,l2-n_theta);
                    r2=flipud(r2);
                end

                alpha=(l1-1)*dtheta;
                beta =(l2-1)*dtheta;
                dx1=ref_shifts_2d(k1,1); dy1=ref_shifts_2d(k1,2);
                dx2=ref_shifts_2d(k2,1); dy2=ref_shifts_2d(k2,2);
                % Shift by the exact amount:
                ds=sin(alpha)*dx1+cos(alpha)*dy1-sin(beta)*dx2-cos(beta)*dy2;
                phi=exp(-2*pi*sqrt(-1).*rk.*ds./(2*rmax+1));
                r1=r1.*phi;
                refcorr(k1,k2)=enforce_real(r1'*r2);
            end

            %%%%%%%%%%%% Beginning of debug code %%%%%%%%%%%%            
            if verbose_plot_shifts && improved_correlation
            % If the current shift produces a better correlation, then plot
            % the two appropriately shifted estimated common-lines. The
            % figure also displays the true and estimated common lines, and
            % the estimated correlation value.
            
                fac=max(1/shift_step,1);
                xx=-rmax:1/fac:rmax;
                
                if clstack(k2,k1)<=n_theta                               
                    proj1_shifted=[P1; zeros(1,n_theta) ; conj(flipud(P1))];
                    p1=enforce_real(icfft(proj1_shifted(:,clstack(k1,k2))));                                                            
                    p2=enforce_real(icfft(proj2(:,clstack(k2,k1))));
                else
                    proj1_shifted_flipped=[P1_shifted_flipped; zeros(1,n_theta) ; conj(flipud(P1_shifted_flipped))];
                    p1=enforce_real(icfft(proj1_shifted_flipped(:,clstack(k1,k2))));                   
                    p2=enforce_real(icfft(proj2(:,clstack(k2,k1)-n_theta)));
                end
                
                v1=triginterp(p1,fac);
                v2=triginterp(p2,fac);
                
                figure(h2);
                plot(xx,real(v1));
                hold on;                
                plot(xx,real(v2),'r');
                px=get(gca,'Xlim');
                py=get(gca,'Ylim');
                                
                if found_ref_clmatrix
                    l1str=sprintf('%3d',ref_clmatrix(k1,k2));
                    l2str=sprintf('%3d',ref_clmatrix(k2,k1));
                else
                    l1str='N/A';
                    l2str='N/A';
                end
                
                str=sprintf(strcat(' k1=%d  k2=%d \n shift = %4.2f \n',...
                    ' est cl   = [ %3d  %3d ] \n',...
                    ' ref cl   = [ %s  %s ] \n',...
                    ' est corr = %7.5f'),...
                    k1,k2,shift,clstack(k1,k2),clstack(k2,k1),...
                    l1str,l2str,...
                    corrstack(k1,k2));
                
                if found_ref_clmatrix && found_ref_shifts
                    str=strcat(str,...
                        sprintf('\n ref corr = %7.5f',refcorr(k1,k2)));
                end

                text(px(2)*0.4,py(2)*0.7,str,'EdgeColor','k')
                hold off;                                
            end;
            %%%%%%%%%%%% End of debug code %%%%%%%%%%%%
        end

        t2=clock;
        t=etime(t2,t1);

%         fname=sprintf('cl_4_%02d%02d',k1,k2);
%         print('-depsc',fname);


        %%%%%%%%%%%% Beginning of debug code %%%%%%%%%%%%
        % Count how many times the estimated common-line is close to within
        % max_angle of the true common-line.
        if verbose_detailed_debugging
            % True line for reference
            if verbose_plot_cl
                figure(h3);
                plot(1,squeeze(corrstack(k1,k2)),'.');
                axis([0 2 0 1.1]);
            end

            % The example below assumes that the polar Fourier
            % transform was computed with n_theta=360, that is, 360
            % Fourier rays per projection. This means that C1 and C2
            % are of size 180x180. In the table below, c1 and c2 are
            % the computed common-line between proj1 and proj 2, and
            % tcl1 and tcl2 are the true common-line:
            % clstack(k1,k2,j) is always less than n_theta, so pairs of
            % common-lines that may match are (for example)
            %  cl1    cl2     tcl1  tcl2
            %   1      1        1     1
            %   1      1      181   181
            %   1    181        1   181
            %   1    181      181     1
            % These satisfy the IF statement below.
            % All the other cases
            %   1      1        1   181
            %   1      1      181     1
            %   1    181        1     1
            %   1    181      181   181
            % do not match because of orientation (one pair matches in
            % the same orientation and the other in opposite
            % orientation), and so do not statisfy the IF statement
            % below. The tables above are referred to as Table 1.

            found_matched_cl=0; % Among the NL high correlation pairs,
            % this counts the number of pairs that are close (up to
            % angle_tol) to the true common-line.

            % l1 and l2 are considered close to the true common-lines
            % tcl1 and tcl2, if the discrepancy between each line and
            % its corresponding true line is less than max_angle.
            max_angle=5/180*pi;  % 5 degrees max.
            angle_tol=2*sin(max_angle/2)+1.0e-10;

            alpha=2*pi*sqrt(-1)/(2*n_theta);
            PI=4*atan(1.0);

            % The estimated common-lines (l1,l2) and the true
            % common-lines (tcl1,tcl2) should be both in the same
            % orientation. For example, if the common-lines are
            % (l1,l2)=(1,1), that is the common-line between the
            % projections is in positive orientation, then the true
            % common-line is either (tcl1,tcl2)=(1,1) or
            % (tcl1,tcl2)=(180,180), that is, tcl1 and tcl2 are in the
            % same orientation. It cannot be, for example
            % (tcl1,tcl2)=(1,180). By inspecting Table 1 above we see
            % that (l1,l2) is close to (tcl1,tcl2) if
            %   a) tcll1 is close to l1 and tcll2 is close to l2. That
            %   covers the cases (l1,l2)=(1,1) (tcl1,tcl2)=(1,1), and
            %   (l1,l2)=(1,180) and (tcl1,tcl2)=(1,180). In that case,
            %   the pair (l1,l2) and the pair (tcl1,tcl2) are in the
            %   same orientation.
            %   Or
            %   b) tcl1 is close to flipped l1 and tcll2 is close to
            %   flipped l2. That covers the cases  (l1,l2)=(1,1)
            %   (tcl1,tcl2)=(180,180), and (l1,l2)=(1,180) and
            %   (tcl1,tcl2)=(180,1). In this case tcl1 is close to a
            %   flipped l1, That is, we need to flip l1 to get a
            %   match with tcl1. However, to maintain the orientation
            %   between l1 and l2 (so we stay we the same
            %   common-line), we should flip also l2, and so tcl2
            %   should match a flipped l2.
            % To conclude, for (l1,l2) to be close to (tcl1,tcl2) we
            % need that either l1 is close to tcl1 and l2 is close to
            % tcl2 (small d1s and d2s below), or, a flipped l1 is
            % close to tcl1 and a flipped l2 is close to tcl2 (small
            % d1f and d2f).
            
            if found_ref_clmatrix
                tcl1=ref_clmatrix(k1,k2);
                tcl2=ref_clmatrix(k2,k1);
                l1=clstack(k1,k2);
                l2=clstack(k2,k1);
                d1s=abs(exp(alpha*(l1-1))-exp(alpha*(tcl1-1)));
                d2s=abs(exp(alpha*(l2-1))-exp(alpha*(tcl2-1)));
                d1f=abs(exp(alpha*(l1-1)+sqrt(-1)*PI)-exp(alpha*(tcl1-1)));
                d2f=abs(exp(alpha*(l2-1)+sqrt(-1)*PI)-exp(alpha*(tcl2-1)));

                if (d1s<=angle_tol) && (d2s<=angle_tol) || ...
                        (d1f<=angle_tol) && (d2f<=angle_tol)
                    found_matched_cl=1;

                    % Estimated common line is close to true common-line.
                    if verbose_plot_cl
                        hold on;
                        plot(corrstack(k1,k2),'o','MarkerSize',10,'MarkerEdgeColor','g');
                        hold off;
                    end
                else
                    % Estimated common-line is far from true common-line.
                    if verbose_plot_cl
                        hold on;
                        plot(corrstack(k1,k2),'o','MarkerSize',10,'MarkerEdgeColor','r');
                        hold off;
                    end
                end

                % Estimation error in angles
                if (tcl1<=n_theta && l1<=n_theta) ||...
                        (tcl1>n_theta && l1>n_theta)  % Same orientation for l1
                    thetadiff(k1,k2)=d1s/pi*180;
                else
                    thetadiff(k1,k2)=d1f/pi*180;
                end
                if (tcl2<=n_theta && l2<=n_theta) ||...
                        (tcl2>n_theta && l2>n_theta)  % Same orientation for l1
                    thetadiff(k2,k1)=d2s/pi*180;
                else
                    thetadiff(k2,k1)=d2f/pi*180;
                end

                if found_matched_cl
                    matched_cl=matched_cl+1;
                end
            end
        end

        %%%%%%%%%%%% End of debug code %%%%%%%%%%%%
       
         if verbose_detailed_debugging                        

%              figure(3);
%              plot(triginterp(enforce_real(icfft(cryo_raynormalize(r1))),1),'b');
%              hold on;
%              plot(triginterp(enforce_real(icfft(cryo_raynormalize(r2))),1),'r');
%              hold off;
             

            if found_ref_shifts
                % Compute the true 1D shift between the common lines and
                % compare it to the  estimated shift.
                alpha=(clstack(k1,k2)-1)*dtheta;
                beta =(clstack(k2,k1)-1)*dtheta;
                dx1=ref_shifts_2d(k1,1); dy1=ref_shifts_2d(k1,2);
                dx2=ref_shifts_2d(k2,1); dy2=ref_shifts_2d(k2,2);

                % If clstack(k2,k1)==151 then beta is supposed to be exactly
                % pi. However, because of roundoff error, it might come
                % slightly less than pi, in which case the first IF will be
                % true, although it should be false. We fix this by comparing
                % beta against pi-1.0e-13 (under the reasonable assumption that
                % dtheta is larger than 1.0e-13).
                if beta<pi-1.0e-13
                    ref_shifts_1d(k1,k2)=sin(alpha)*dx1+cos(alpha)*dy1-sin(beta)*dx2-cos(beta)*dy2;
                else
                    beta=beta-pi;
                    ref_shifts_1d(k1,k2)=-sin(alpha)*dx1-cos(alpha)*dy1-sin(beta)*dx2-cos(beta)*dy2;
                end

                shift_estimation_error(k1,k2)=shifts_1d(k1,k2)-ref_shifts_1d(k1,k2);

            end
            
            log_message('Finding common-line between projections [k1 k2]=[ %d %d ]',k1,k2);
            log_message('\t Common lines:');
            log_message('\t \t clstack= [ %3d %3d ]',clstack(k1,k2),clstack(k2,k1)); 
            
            if found_ref_clmatrix
                log_message('\t \t ref    = [ %3d %3d ]',ref_clmatrix(k1,k2),ref_clmatrix(k2,k1));
                log_message('\t \t dtheta1= %5.2f (degrees)',thetadiff(k1,k2));
                log_message('\t \t dtheta2= %5.2f (degrees)',thetadiff(k2,k1));
                if found_matched_cl
                    clstack_mask(k1,k2)=1;
                    log_message('\t \t status = MATCHED (less than %5.2f degrees)',max_angle/pi*180);
                else
                    log_message('\t \t status = NOT MATCHED (more than %5.2f degrees)',max_angle/pi*180');
                end
            end
            
            log_message('\t Correlation:');
            log_message('\t \t est    =  %9.6f',corrstack(k1,k2));
            
            if found_ref_clmatrix && found_ref_shifts
                log_message('\t \t true   =  %9.6f',refcorr(k1,k2));
            end
            
            log_message('\t Shifts:');
            log_message('\t \t est  shift= %6.4f',shifts_1d(k1,k2));
            
            if found_ref_shifts
                log_message('\t \t True shift= %6.4f',ref_shifts_1d(k1,k2));
                log_message('\t \t shift err = %3.2e',shift_estimation_error(k1,k2));
            end
            log_message(' ');
        end

        % Create a shift equation for the projections pair (k1,k2).
        idx=4*(shift_equation_idx-1)+1:4*shift_equation_idx;
        shift_alpha=(clstack(k1,k2)-1)*dtheta;  % Angle of common ray in projection 1.
        shift_beta= (clstack(k2,k1)-1)*dtheta;  % Angle of common ray in projection 2.
        shift_I(idx)=shift_equation_idx; % Row index to construct the sparse equations.
        shift_J(idx)=[2*k1-1 2*k1 2*k2-1 2*k2]; % Columns of the shift variables that correspond to the current pair (k1,k2).
        shift_b(shift_equation_idx)=shifts_1d(k1,k2); % Right hand side of the current equation.

        % Compute the coefficients of the current equation.
        if shift_beta<pi       
            shift_eq(idx)=[sin(shift_alpha) cos(shift_alpha) -sin(shift_beta) -cos(shift_beta)];
        else
            shift_beta=shift_beta-pi; % In the derivation we assume that all angles are less 
                                      % than PI where angles larger than PI are assigned 
                                      % nigative orientation.
            shift_eq(idx)=[-sin(shift_alpha) -cos(shift_alpha) -sin(shift_beta) -cos(shift_beta)];
        end
    
        shift_equations_map(k1,k2)=shift_equation_idx;  % For each pair (k1,k2), store the index of its equation.
        shift_equation_idx=shift_equation_idx+1;
        
        if verbose_progress
            bs=char(repmat(8,1,numel(msg)));
            fprintf('%s',bs);
            msg=sprintf('k1=%3d/%3d  k2=%3d/%3d  t=%7.5f',k1,n_proj,k2,n_proj,t);
            fprintf('%s',msg);
        end
    end;
end;


if verbose_progress
    fprintf('\n');
end

if verbose_detailed_debugging && found_ref_clmatrix
    log_message('Matched common-lines=%d/%d  (%3.1f%%)',...
        matched_cl,n_proj*(n_proj-1)/2,...
        100*matched_cl/((n_proj*(n_proj-1))/2));
end

corrstack(corrstack~=0)=1-corrstack(corrstack~=0);
             
% Construct least-squares for the two-dimensioal shifts.
shift_equation_idx=shift_equation_idx-1;
shift_equations=sparse(shift_I(1:4*shift_equation_idx),...
    shift_J(1:4*shift_equation_idx),shift_eq(1:4*shift_equation_idx),...
    shift_equation_idx,2*n_proj);

shift_equations=[shift_equations shift_b(1:shift_equation_idx)];


if verbose_detailed_debugging
    % XXX Check that the shift estimation improves with n_theta.
    % XXX Does it improve with n_proj?
    if n_proj<=100
        [~,S,V]=svd(full(shift_equations(:,1:end-1)));
        s=diag(S);
        log_message('Singular values of the shift system of equations:');
        log_message('%d  ',fliplr(s.'));

        % Check that the difference between the true shifts and the estimated ones
        % is in the null space of the equations.
        est_shifts=shift_equations(:,1:end-1)\shift_equations(:,end);
        est_shifts=transpose(reshape(est_shifts,2,n_proj));
        
        if found_ref_shifts
            s1=reshape(ref_shifts_2d.',2*n_proj,1);
            s2=reshape(est_shifts.',2*n_proj,1);
            V=V(:,1:end-3); % Null space of shift_equations.
            % Compute the difference between the true and estimated shifts in
            % the subspace that is orthogonal to the null space of
            % shift_equations.

            if norm(V.'*s1)>1.0e-12
                log_message('Difference between true and estimated shifts: %8.5e',...
                    (norm(V.'*(s1-s2))/norm(V.'*s1)));
            else
                log_message('norm(V.''*s1) = %7.5e',norm(V.'*s1));
            end
        end
    else
        log_message('Not computing SVD of shifts matrix -- matrix is too big');
    end;
end

function y=enforce_real(x)
% The polar Fourier transform of each projection is computed to single
% precision. The correlations should therefore be real to single precision.
err=norm(imag(x(:)))/norm(x(:));
if err>1.0e-7
    warning('GCAR:imaginaryComponents','Imaginary components err=%d',err);
end
y=real(x);
