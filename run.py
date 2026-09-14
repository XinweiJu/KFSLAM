"""Portable entry point for the paper methods and named ablations."""
from pathlib import Path
import argparse,contextlib,copy,importlib.util,json,os,sys
ROOT=Path(__file__).resolve().parent

def resolve_config(args):
    manifest=json.loads((ROOT/'configs/experiments.json').read_text())
    matches=[x for x in manifest if (x['backbone'],x['variant'],x['dataset'],x['sequence'])==(args.backbone,args.variant,args.dataset,args.sequence)]
    if not matches:raise ValueError('No exported configuration for the requested combination; use --list.')
    c=json.loads((ROOT/matches[0]['config']).read_text());output=Path(args.output).resolve()
    replace={'${DATASET_ROOT}':str(Path(args.dataset_root).resolve()),'${FOLD_ROOT}':str(Path(args.fold_root).resolve()) if args.fold_root else '', '${CAMERA_CONFIG}':str(ROOT/'final'/args.backbone/'configs/data'/f'{args.dataset}.yaml'),'${OUTPUT_PARENT}':str(output.parent),'${RUN_NAME}':output.name,'${MODULE_ROOT}':str(ROOT/'modules')}
    def visit(v):
        if isinstance(v,dict):return {k:visit(x) for k,x in v.items()}
        if isinstance(v,list):return [visit(x) for x in v]
        return replace.get(v,v) if isinstance(v,str) else v
    c=visit(c)
    if c.get('fold_seg_loss',{}).get('enable') and not args.fold_root:raise ValueError('--fold-root is required for fold-supervised methods')
    if args.frames is not None:c['data']['end']=c['data'].get('start',0)+args.frames;c['data']['num_frames']=-1
    if args.mapping_iters is not None:c['mapping']['num_iters']=args.mapping_iters;c['dynamic_mapping_iters']['enable']=False
    if args.tracking_iters is not None:c['tracking']['num_iters']=args.tracking_iters
    c['primary_device']='cuda:0';c['use_wandb']=False
    return c

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',choices=['splatam','endogslam'],default='splatam');p.add_argument('--variant',default='full');p.add_argument('--dataset',choices=['c3vd','c3vd_v2'],default='c3vd_v2');p.add_argument('--sequence',default='c1_rectum_t2_v1');p.add_argument('--dataset-root',default='data');p.add_argument('--fold-root');p.add_argument('--output',default='outputs/run');p.add_argument('--gpu',default='0');p.add_argument('--list',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--frames',type=int);p.add_argument('--mapping-iters',type=int);p.add_argument('--tracking-iters',type=int);p.add_argument('--cpu-offload',action='store_true')
    args=p.parse_args()
    if args.list:
        jobs=json.loads((ROOT/'configs/experiments.json').read_text());print('\n'.join(f'{b}: {v}' for b,v in sorted({(j['backbone'],j['variant']) for j in jobs})));return
    c=resolve_config(args)
    if args.dry_run:print(json.dumps(c,indent=2));return
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True');os.environ.setdefault('WANDB_MODE','disabled')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'effective_config.json').write_text(json.dumps(c,indent=2))
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'modules'));sys.path.insert(0,str(ROOT/'final'/args.backbone))
    spec=importlib.util.spec_from_file_location('kfs4lam_backbone',ROOT/'final'/args.backbone/'scripts/kfs4lam.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if args.backbone=='splatam':
        if c.get('fold_seg_loss',{}).get('enable'):
            from modules.structure_guided import install
            install(module,c)
        if c.get('selection_ablation'):
            from modules.selection_ablations import attach
            attach(module,c)
    elif c.get('fold_seg_loss',{}).get('enable'):
        from modules.structure_guided_endogslam import install_stable_cache
        install_stable_cache(module)
    import torch
    module.seed_everything(c['seed'])
    with (torch.autograd.graph.save_on_cpu(pin_memory=True) if args.cpu_offload else contextlib.nullcontext()):module.rgbd_slam(c)
if __name__=='__main__':main()
