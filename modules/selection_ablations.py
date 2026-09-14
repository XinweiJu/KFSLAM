"""Isolated selection hooks; native SplaTAM optimization remains unchanged."""
from pathlib import Path
import os,sys,json,importlib.util
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
def attach(s,c):
 mode=c['selection_ablation']['mode']
 s._redundancy_gate=lambda proposed,debug,color,keyframes,config:(proposed,'disabled_for_selection_ablation')
 if mode=='motion_proposal_only':return
 if mode=='monogs_fixed10':
  def fixed(t,T,params,kfs,config,state):
   return (not kfs or t-kfs[-1]['id']>=10),{'reason':'monogs_fixed10','gap':t-kfs[-1]['id'] if kfs else 0}
  s._accum_motion_keyframe_decision=fixed;return
 path=ROOT/'third_party/monogs_rasterizer/diff_gaussian_rasterization/__init__.py';spec=importlib.util.spec_from_file_location('monogs_selection_raster',path,submodule_search_locations=[str(path.parent)]);native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
 import yaml
 k=yaml.safe_load(Path(c['data']['gradslam_data_cfg']).read_text())['camera_params'];w=c['data']['desired_image_width'];h=c['data']['desired_image_height'];K=np.array([[k['fx']*w/k['image_width'],0,k['cx']*w/k['image_width']],[0,k['fy']*h/k['image_height'],k['cy']*h/k['image_height']],[0,0,1]])
 cam=s.setup_camera(w,h,K,np.eye(4));kw={name:getattr(cam,name) for name in native.GaussianRasterizationSettings._fields if hasattr(cam,name)};kw.update(projmatrix_raw=cam.projmatrix.squeeze(0),viewmatrix=cam.viewmatrix.squeeze(0),projmatrix=cam.projmatrix.squeeze(0),debug=False);renderer=native.GaussianRasterizer(native.GaussianRasterizationSettings(**kw))
 def visibility(params,t):
  transformed=s.transform_to_frame(params,t,gaussians_grad=False,camera_grad=False);rv=s.transformed_params2rendervar(params,transformed);_,_,depth,opacity,touched=renderer(**rv);return touched>0,depth,opacity
 def decide(t,T,params,kfs,config,state):
  if not kfs:return True,{'reason':'monogs_init','gap':0}
  last=kfs[-1]['id'];gap=t-last
  if gap<10:return False,{'reason':'monogs_min_interval','gap':gap}
  with torch.no_grad():
   v,depth,opacity=visibility(params,t);prev,_,_=visibility(params,last);union=(v|prev).count_nonzero();overlap=float((v&prev).count_nonzero()/union) if union else 1.0
   valid=(depth>0)&(opacity>.95);median=float(depth[valid].median()) if valid.any() else float('nan')
   a,ra=s._kfs_pose(params,t);b,rb=s._kfs_pose(params,last);dist=float(np.linalg.norm(a-ra@rb.T@b))
   if len(kfs)<10:add=overlap<.95;reason='monogs_bootstrap_overlap'
   else:add=bool(np.isfinite(median) and ((overlap<.95 and dist>.02*median) or dist>.04*median));reason='monogs_overlap_translation'
  debug=dict(reason=reason,gap=gap,overlap=overlap,median_depth=median,translation=dist,proposal=add)
  print('[MonoGSSelector] frame='+str(t)+' '+json.dumps(debug),flush=True);return add,debug
 s._accum_motion_keyframe_decision=decide

