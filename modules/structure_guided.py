import torch
from scipy.ndimage import distance_transform_edt
def objective(render,target,mode,dt=None,dt_weight=.1):
 if mode=='soft_dice':
  return 1-(2*(render*target).sum()+1e-6)/(render.sum()+target.sum()+1e-6)
 l1=(render-target).abs().mean()
 if dt is None:return render.sum()*0 if mode=='target_dt' else l1
 distance=(render*dt).sum()/(render.sum()+1e-6)
 if mode=='target_dt':return distance
 if mode=='l1_dt':return l1+dt_weight*distance
 raise ValueError(mode)

def install(s,config):
 settings=config["fold_loss_variant"]
 cache={}
 stable_targets={}
 from modules import fold_boundary_dt as hist
 def loss(params,data,idx):
  transformed=s.transform_to_frame(params,idx,gaussians_grad=True,camera_grad=False)
  rgb,_,_=s.Renderer(raster_settings=data['cam'])(**s._fold_rendervar(params,transformed));pred=rgb[:1];target=data['fold_seg'].to(pred).clamp(0,1)
  if target.shape[-2:]!=pred.shape[-2:]:target=s.F.interpolate(target[None],size=pred.shape[-2:],mode='nearest')[0]
  key=(idx,tuple(target.shape),str(target.device))
  if settings['mode'].startswith('historical_'):
   # Hold target tensors alive: legacy helper caches key by storage address.
   if key not in stable_targets:stable_targets[key]=target.detach().clone()
   fixed=stable_targets[key]
   if settings['mode']=='historical_target_dt':value=hist._fold_target_dt_loss(pred,fixed,'mean')
   else:value=(hist._fold_boundary_map(pred)*hist._fold_boundary_distance_target(fixed)).mean()
   return float(data.get('fold_seg_weight',1))*value
  if key not in cache:
   support=target[0].detach().cpu().numpy()>settings['support_threshold']
   dt=None if not support.any() else torch.as_tensor(np.minimum(distance_transform_edt(~support),settings['distance_cap_px'])/settings['distance_cap_px'],device=pred.device,dtype=pred.dtype)[None]
   cache[key]=dt
  return float(data.get('fold_seg_weight',1))*objective(pred,target,settings['mode'],cache[key],settings['dt_weight'])

 s._fold_mapping_loss=loss
