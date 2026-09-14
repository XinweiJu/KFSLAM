"""Compose a moving estimated camera/trajectory alongside final-map RGB renders."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--renders',required=True);p.add_argument('--output',required=True);p.add_argument('--method',required=True);p.add_argument('--sequence',required=True);p.add_argument('--fps',type=int,default=15);a=p.parse_args()
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 import imageio_ffmpeg
 z=np.load(a.checkpoint,allow_pickle=False);q=z['cam_unnorm_rots'][0].T;q=q/np.linalg.norm(q,axis=1,keepdims=True);w,x,y,zz=q.T
 R=np.stack([1-2*(y*y+zz*zz),2*(x*y-w*zz),2*(x*zz+w*y),2*(x*y+w*zz),1-2*(x*x+zz*zz),2*(y*zz-w*x),2*(x*zz-w*y),2*(y*zz+w*x),1-2*(x*x+y*y)],axis=1).reshape(-1,3,3)
 centers=-np.einsum('nji,nj->ni',R,z['cam_trans'][0].T);n=len(centers)
 files=sorted(Path(a.renders).glob('rgb_*.png'));ids=[int(f.stem.split('_')[-1]) for f in files]
 lo=centers.min(0);hi=centers.max(0);mid=(lo+hi)/2;span=max(float(np.max(hi-lo)),1.);radius=span*.64;scale=span*.085
 fig=plt.figure(figsize=(14,6),dpi=120,facecolor='white');ax=fig.add_axes([.01,.11,.46,.75],projection='3d');right=fig.add_axes([.51,.1,.48,.76]);right.set_axis_off()
 ax.view_init(elev=24,azim=-62);ax.set_box_aspect((1,1,1))
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_pane_color((1,1,1,1));axis._axinfo['grid']['color']=(.84,.87,.87,1)
 ax.set(xlim=(mid[0]-radius,mid[0]+radius),ylim=(mid[1]-radius,mid[1]+radius),zlim=(mid[2]-radius,mid[2]+radius),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)');ax.tick_params(labelsize=8)
 ax.plot(*centers.T,color='#d3dbda',lw=1.5,label='Full estimated trajectory');path,=ax.plot([],[],[],color='#c54843',lw=2.6,label='Traversed trajectory');camera,=ax.plot([],[],[],color='#1565c0',lw=2,label='Current camera');dot,=ax.plot([],[],[],'o',color='#1565c0',ms=5)
 ax.legend(loc='upper left',fontsize=8,frameon=False)
 k=z['intrinsics'];width=int(z['org_width']);height=int(z['org_height']);pix=np.array([[0,0,1],[width,0,1],[width,height,1],[0,height,1]])
 corners=(np.linalg.inv(k)@pix.T).T*scale
 first=np.array(Image.open(files[0]).convert('RGB'));img=right.imshow(first)
 fig.text(.25,.92,'Estimated camera & trajectory',ha='center',fontsize=15,weight='bold');fig.text(.75,.92,'Synchronized reconstruction',ha='center',fontsize=15,weight='bold');fig.text(.5,.985,a.method+'  |  '+a.sequence,ha='center',va='top',fontsize=16,weight='bold')
 status=fig.text(.5,.04,'',ha='center',fontsize=11);fig.text(.5,.01,'Final-map replay at saved poses; not online map growth. Gray shows the complete estimated trajectory.',ha='center',fontsize=9,color='#536363')
 dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
 writer=imageio_ffmpeg.write_frames(str(dest),(1680,720),fps=a.fps,codec='libx264',pix_fmt_out='yuv420p',quality=7,macro_block_size=1,output_params=['-map_metadata','-1','-movflags','+faststart']);writer.send(None)
 for f,idx in zip(files,ids):
  c=centers[idx];world=corners@R[idx]+c
  vertices=[]
  for j in range(4):vertices.extend([c,world[j],world[(j+1)%4],np.full(3,np.nan)])
  camera.set_data_3d(np.array(vertices).T);dot.set_data_3d([c[0]],[c[1]],[c[2]]);path.set_data_3d(centers[:idx+1].T);img.set_data(np.array(Image.open(f).convert('RGB')))
  status.set_text(f'Processed frame {idx} / {n-1}   ·   Playback {a.fps} fps   ·   '+('Time-conditioned map' if 'deform_coeffs' in z.files else 'Static map'))
  fig.canvas.draw();buf=np.asarray(fig.canvas.buffer_rgba())[:,:,:3];writer.send(np.ascontiguousarray(buf))
 writer.close();plt.close(fig)
 dest.with_suffix('.json').write_text(json.dumps({'method':a.method,'sequence':a.sequence,'frames':ids,'fps':a.fps,'camera_convention':'world-to-camera quaternion wxyz; camera center = -R.T @ t','view':'final reconstruction replay, not online mapping','deformation_coefficients_present':'deform_coeffs' in z.files},indent=2));print(dest,flush=True)
if __name__=='__main__':main()
