"""Render released reconstruction parameters without retraining or dataset access."""
import argparse,json,os,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',required=True);p.add_argument('--backbone',required=True,choices=['splatam','endogslam']);p.add_argument('--frames',default='0',help='Comma-separated processed-frame indices, or all');p.add_argument('--output',default='rendered');p.add_argument('--gpu',default='0');p.add_argument('--stride',type=int,default=1)
 a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 import numpy as np,torch
 from PIL import Image
 sys.path.insert(0,str(Path(__file__).resolve().parent/'final'/a.backbone))
 from utils.recon_helpers import setup_camera
 from utils.slam_helpers import transform_to_frame,transformed_params2rendervar,transformed_params2depthplussilhouette
 from diff_gaussian_rasterization import GaussianRasterizer
 z=np.load(a.checkpoint,allow_pickle=False)
 params={k:torch.as_tensor(z[k],device='cuda').float() for k in z.files if z[k].dtype.kind in 'fiu'}
 n=z['cam_trans'].shape[-1];ids=list(range(0,n,a.stride)) if a.frames=='all' else [int(x) for x in a.frames.split(',')]
 if any(i<0 or i>=n for i in ids):raise ValueError(f'Frame index must lie in [0,{n-1}]')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 cam=setup_camera(int(z['org_width']),int(z['org_height']),z['intrinsics'],z['w2c'])
 with torch.no_grad():
  for idx in ids:
   pts=transform_to_frame(params,idx,gaussians_grad=False,camera_grad=False)
   rgb,*_=GaussianRasterizer(raster_settings=cam)(**transformed_params2rendervar(params,pts))
   depth,*_=GaussianRasterizer(raster_settings=cam)(**transformed_params2depthplussilhouette(params,params['w2c'],pts))
   Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(out/f'rgb_{idx:05d}.png')
   np.save(out/f'depth_{idx:05d}.npy',depth[0].cpu().numpy())
   print(f'Rendered processed frame {idx}',flush=True)
 (out/'render_manifest.json').write_text(json.dumps({'backbone':a.backbone,'processed_frame_indices':ids,'processed_frames':n,'width':int(z['org_width']),'height':int(z['org_height']),'deformation_coefficients_present':'deform_coeffs' in z.files,'depth':'raw renderer depth; unnormalized; scene units; not a grayscale visualization','frame_convention':'EndoGSLAM indices refer to the retained training split, not original RGB filenames'},indent=2))
if __name__=='__main__':main()
