"""Paper pose metrics and true depth RMSE for a completed local run."""
import argparse,json,os
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--backbone',choices=['splatam','endogslam'],required=True);p.add_argument('--dataset',choices=['c3vd','c3vd_v2'],required=True);p.add_argument('--gpu',default='0');a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 from evaluation import metrics as m
 import numpy as np
 run=Path(a.run).resolve();c=json.loads((run/'effective_config.json').read_text());scene=c['data']['sequence']
 if a.backbone=='splatam':
  from evaluation import splatam_metrics as s
  result=m.pose_metrics(*m.w2c_from_splatam_params(run/'params.npz'))
  for k in ['psnr','ssim','lpips']:result[k],result[k+'_frames']=s.sample_metric(run/'eval'/f'{k}.txt')
  ids=s.selected_frame_ids(run)
  if not ids:raise ValueError('No saved evaluation frames found')
  result['depth_rmse'],result['depth_frames']=s.true_depth_rmse(run,c,a.dataset,scene,ids,'cuda:0')
 else:
  import lpips
  m.C3VD_ROOT=m.C3VD_V2_ROOT=Path(c['data']['basedir']);result=m.pose_metrics(m.read_w2c_txt(run/'eval/est_w2c.txt'),m.read_w2c_txt(run/'eval/gt_train_w2c.txt'));result.update(m.endo_render_metrics(run,a.dataset,scene,lpips.LPIPS(net='alex',verbose=False).to('cuda:0').eval(),'cuda:0',None))
 with np.load(run/'params.npz',allow_pickle=True) as z:
  if 'keyframe_time_indices' in z.files:result['keyframes']=len(z['keyframe_time_indices'])
 result['translation_units']='mm';result['rotation_units']='degrees';result['protocol']='all saved native evaluation frames; backbone-specific sampling retained'
 (run/'paper_metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
