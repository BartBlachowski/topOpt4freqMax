from identity import *
import scipy.io as sio

def main():
 out={}
 for nx in [480,800]:
  d=sio.loadmat(OUT/'evaluations'/f'spectral_{nx}.mat',squeeze_me=True);r=d['rho'];g=d['gRaw'];gf=d['gFiltered'];gray=(r>.1)&(r<.9);geom=np.load(OUT/'evaluations'/f'geometry_{nx}.npz');dep=geom['depth'].T.ravel();b=dep>.06;scale=np.sqrt(np.mean(g*g));t=max(0,g[b].mean());tf=max(0,gf[gray].mean());lam=d['lam'][0]
  out[str(nx)]={'broad_fit_threshold':float(t),'broad_fit_mu':float(.5*len(r)*t/lam),'filtered_gray_threshold':float(tf),'raw_broad_RMS_best_core_fit':float(np.sqrt(np.mean((g[b]-t)**2))/scale),'raw_broad_RMS_filtered_gray_dual':float(np.sqrt(np.mean((g[b]-tf)**2))/scale),'raw_allgray_RMS_core_fit':float(np.sqrt(np.mean((g[gray]-t)**2))/scale),'interpretation':'Local core-only compatibility is not a global KKT certificate; no common dual removes residuals throughout gray.'}
 (OUT/'evaluations/local_core.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
