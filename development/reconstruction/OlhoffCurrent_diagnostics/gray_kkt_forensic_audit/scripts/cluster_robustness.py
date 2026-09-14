"""Exploratory dual robustness after observing resolved but close 800 eigenpair.
No design optimization: three-parameter linear least-squares diagnostic only.
Enlarges dual feasible set; thus its residual is an optimistic lower bound.
"""
from identity import *
import scipy.io as sio

def main():
 out={}
 for nx in [400,480,800]:
  d=sio.loadmat(OUT/'evaluations'/f'spectral_{nx}.mat',squeeze_me=True);r=d['rho'];gray=(r>.1)&(r<.9);free=(r>.0010001)&(r<.9999999);scale=np.sqrt(np.mean(d['gRaw'][free]**2));lam=d['lam'];D=np.diag(lam[:2]-lam[0]);res={}
  for key in ['Fraw','Ffiltered']:
   F=d[key];A=np.c_[F[gray,0,0]-F[gray,1,1],2*F[gray,0,1],-np.ones(gray.sum())]
   a,b,t=np.linalg.lstsq(A,-F[gray,1,1],rcond=None)[0];Q=np.array([[a,b],[b,1-a]])
   g=a*F[:,0,0]+2*b*F[:,0,1]+(1-a)*F[:,1,1]
   res[key]={'Q':Q.tolist(),'threshold_t':float(t),'mu_volume':float(.5*len(r)*t/lam[0]),'Q_eigenvalues':np.linalg.eigvalsh(Q).tolist(),'Q_PSD':bool(np.linalg.eigvalsh(Q).min()>=-1e-12),'volume_dual_nonnegative':bool(t>=0),'complementarity_trace_QD_over_lambda1':float(np.trace(Q@D)/lam[0]),'complementarity_QD_Fro_over_lambda1':float(np.linalg.norm(Q@D)/lam[0]),'gray_RMS_lower_bound_over_raw_interior_RMS':float(np.sqrt(np.mean((g[gray]-t)**2))/scale),'note':'Unconstrained trace-one symmetric dual fit: lower bound, NOT an admissible exact KKT multiplier for separated eigenvalues.'}
   if nx==800:
    GK=d['GK'];GM=d['GM'];kg=a*GK[:,0,0]+2*b*GK[:,0,1]+(1-a)*GK[:,1,1];mg=a*GM[:,0,0]+2*b*GM[:,0,1]+(1-a)*GM[:,1,1]
    res[key]['raw_dual_weighted_gray_cancellation_median']=float(np.median(abs(kg[gray]+mg[gray])/(abs(kg[gray])+abs(mg[gray])+1e-30)))
  out[str(nx)]=res
 (OUT/'evaluations/cluster_robustness.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
