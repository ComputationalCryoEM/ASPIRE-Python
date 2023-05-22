




import numpy as np
import pymanopt
from numpy.linalg import norm
import wemd
import mrcfile
import warnings
from aspire.utils.rotation import Rotation
from aspire.volume import Volume
from numpy.random import normal
from scipy.optimize import minimize
from numpy import pi
from scipy.ndimage import shift






def RR():
    return Rotation.generate_random_rotations(1)._matrices[0]

def get_angle(R1,R2):
    cos_theta=(np.trace(R1@R2.T)-1)/2;
    if cos_theta>1:
        cos_theta=1
    if cos_theta<-1:
        cos_theta=-1
    return np.arccos(cos_theta)*(180)/pi


def u_to_rot(u):
    v=np.sqrt(u[0]**2+u[1]**2+u[2]**2); 
    if v==0:
        q=np.array([1,0,0,0]);
    else:
        q=np.array([np.cos(v/2), np.sin(v/2)/v*u[0], np.sin(v/2)/v*u[1], np.sin(v/2)/v*u[2]]); 
    R=np.zeros((3,3),dtype=np.float32); 
    R[0,0]=q[0]**2+q[1]**2-q[2]**2-q[3]**2;
    R[0,1]=2*q[1]*q[2]-2*q[0]*q[3]; 
    R[0,2]=2*q[0]*q[2]+2*q[1]*q[3]; 
    R[1,0]=2*q[1]*q[2]+2*q[0]*q[3];
    R[1,1]=q[0]**2-q[1]**2+q[2]**2-q[3]**2; 
    R[1,2]=-2*q[0]*q[1]+2*q[2]*q[3]; 
    R[2,0]=-2*q[0]*q[2]+2*q[1]*q[3]; 
    R[2,1]=2*q[0]*q[1]+2*q[2]*q[3]; 
    R[2,2]=q[0]**2-q[1]**2-q[2]**2+q[3]**2;
    return R


def rot_to_u(R):
    q=np.zeros(4);
    if R[1,1]>-R[2,2] and R[0,0]>-R[1,1] and R[0,0]>-R[2,2]:
        q[0]=np.sqrt(1+R[0,0]+R[1,1]+R[2,2]);
        q[1]=(R[2,1]-R[1,2])/q[0];
        q[2]=(R[0,2]-R[2,0])/q[0];
        q[3]=(R[1,0]-R[0,1])/q[0]; 
    
    elif R[1,1]<-R[2,2] and R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        q[1]=np.sqrt(1+R[0,0]-R[1,1]-R[2,2]);
        q[0]=(R[2,1]-R[1,2])/q[1];     
        q[2]=(R[1,0]+R[0,1])/q[1];
        q[3]=(R[2,0]+R[0,2])/q[1];
    
    elif R[1,1]>R[2,2] and R[0,0]<R[1,1] and R[0,0]<-R[2,2]:
        q[2]=np.sqrt(1-R[0,0]+R[1,1]-R[2,2]);
        q[0]=(R[0,2]-R[2,0])/q[2]; 
        q[1]=(R[1,0]+R[0,1])/q[2]; 
        q[3]=(R[2,1]+R[1,2])/q[2]; 
    
    elif R[1,1]<R[2,2] and R[0,0]<-R[1,1] and R[0,0]<R[2,2]:
        q[3]=np.sqrt(1-R[0,0]-R[1,1]+R[2,2]);
        q[0]=(R[1,0]-R[0,1])/q[3];
        q[1]=(R[2,0]+R[0,2])/q[3];
        q[2]=(R[2,1]+R[1,2])/q[3];    
    
    q=q/2; v=2*np.arccos(q[0]); 
    if v==0:
        return np.zeros(3); 
    else:
        return v/np.sin(v/2)*q[1:4]; 
    
    

def q_to_rot(q):
    R=np.zeros((3,3));
    R[0,0]=q[0]**2+q[1]**2-q[2]**2-q[3]**2; R[0,1]=2*(q[1]*q[2]-q[0]*q[3]); R[0,2]=2*(q[1]*q[3]+q[0]*q[2]);
    R[1,0]=2*(q[1]*q[2]+q[0]*q[3]); R[1,1]=q[0]**2-q[1]**2+q[2]**2-q[3]**2; R[1,2]=2*(q[2]*q[3]-q[0]*q[1]);
    R[2,0]=2*(q[1]*q[3]-q[0]*q[2]); R[2,1]=2*(q[2]*q[3]+q[0]*q[1]); R[2,2]=q[0]**2-q[1]**2-q[2]**2+q[3]**2;
    
    return R
       


def generate_data(data_name,inv_SNR):
    with mrcfile.open('data/'+data_name) as mrc:
        template=Volume(mrc.data)
    L=template.shape[1]; shape=(L,L,L); ns_std=np.sqrt(inv_SNR*norm(template)**2/L**3);    
    vol0=template+np.float32(normal(0,ns_std,shape));   
    r=Rotation.generate_random_rotations(1); R_true=r._matrices[0]; 
    vol_given=template.rotate(r)+np.float32(normal(0,ns_std,shape)); 
    
    return vol0, vol_given, L, R_true    





         
def center(vol,order_shift,threshold=-np.inf):
    v=np.copy(vol); v.setflags(write=1); v[v<threshold]=0; 
    v=v/v.sum(); L=vol.shape[1]; X=np.zeros(L); mid=int(L/2);
    for i in range(L):
        X[i]=i-mid;
    
    vx=np.sum(v,axis=(1,2)); vy=np.sum(v,axis=(0,2)); vz=np.sum(v,axis=(0,1));
    m=np.array([X@vx,X@vy,X@vz]); vol_b=shift(vol,-m,order=order_shift,mode='constant')
    return vol_b



def align_BO(vol0,vol_given,para,reflect=False):       
    loss_type=para[0]; ds=para[1]; Niter=para[2]; refine=para[3];          

    vol0_ds=vol0.downsample(ds); vol_given_ds=vol_given.downsample(ds); 
    if loss_type=='wemd':
        lengthscale=0.75;
        warnings.filterwarnings("ignore"); wavelet='sym3'; level=6; 
        embed_0=wemd.embed(vol0_ds._data[0],wavelet,level);
    if loss_type=='eu':
        lengthscale=1;
    
    def loss(R):
        v_rot=vol_given_ds.rotate(Rotation(R))._data[0];            
        if loss_type=='eu':
            return norm(vol0_ds-v_rot)
        if loss_type=='wemd':
            warnings.filterwarnings("ignore")
            embed_rot=wemd.embed(v_rot, wavelet, level); 
            return norm(embed_rot-embed_0,ord=1)  
    
    def cf(x1,x2):
        d=norm(x1-x2)/lengthscale
        return np.exp(-d**2/2)
    def cf_grad(x1,x2):
        return cf(x1,x2)*(x2-x1)/lengthscale**2

    R=np.zeros((Niter,3,3),dtype='float32');  
    ninit=1; R[0]=np.float32(np.eye(3)); 
    if reflect:
        manifold=pymanopt.manifolds.Stiefel(3,3);
    else:
        manifold=pymanopt.manifolds.SpecialOrthogonalGroup(3) 
    
    C=np.zeros((Niter,Niter)); 
    for i in range(ninit):       
        for j in range(ninit):
            C[i,j]=cf(R[i],R[j])
          
    y=np.zeros(Niter);
    for i in range(ninit):       
        y[i]=loss(R[i]);
        
    tau=1e-3; max_iter=500; min_grad=0.1; min_sz=0.1; verb=0;
    
    for t in range(ninit,Niter):   
        q=np.linalg.solve(C[:t,:t]+tau*np.eye(t),y[:t]); 
        
        @pymanopt.function.numpy(manifold)
        def cost(new):
            kx=np.array([cf(new,R[j]) for j in range(t)]); mu=kx@q; 
            return mu
        @pymanopt.function.numpy(manifold)
        def eu_grad(new):
            kx_grad=np.array([cf_grad(new,R[j]) for j in range(t)]); grad=np.einsum('ijk,i',kx_grad,q); 
            return grad                       
                    
        problem=pymanopt.Problem(manifold, cost, euclidean_gradient=eu_grad)                  
        optimizer=pymanopt.optimizers.SteepestDescent(max_iterations=max_iter, min_gradient_norm=min_grad, min_step_size=min_sz, verbosity=verb)        
        result=optimizer.run(problem); R_new=np.float32(result.point);           
     
        y[t]=loss(R_new); R[t]=R_new;      
        k=np.array([cf(R_new,R[i]) for i in range(t+1)]);
        C[t,0:t]=k[0:t]; C[0:t+1,t]=k
                               
    idx=y.argmin(0); R_init=R[idx];

    if ds>32:
        vol0_ds=vol0_ds.downsample(32); vol_given_ds=vol_given_ds.downsample(32); 
    if refine:
        sign=np.sign(np.linalg.det(R_init)); 
        def loss_u(u):
            R=sign*u_to_rot(u); v_rot=vol_given_ds.rotate(Rotation(R))._data[0];
            return norm(vol0_ds-v_rot)
        
        x0=rot_to_u(sign*R_init);   
        result=minimize(loss_u,x0,method='nelder-mead',options={'disp': False}); 
        u_est=result.x; R_est=sign*u_to_rot(u_est); 
        
    else:
        R_est=R_init
    
    return R_init, R_est
        
        