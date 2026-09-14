"""Native EndoGSLAM Boundary-DT with bounded, lifetime-safe target cache."""
from collections import OrderedDict

def install_stable_cache(module):
    original=module._fold_boundary_distance_target
    retained=OrderedDict()
    module._FOLD_BOUNDARY_DT_CACHE.clear()
    capacity=module._FOLD_BOUNDARY_DT_CACHE_MAX
    def target_dt(target,dmax=10.0):
        key=(int(target.data_ptr()),tuple(target.shape),str(target.device),str(target.dtype),float(dmax))
        if key not in retained:
            # Keeping the source alive prevents allocator address reuse from aliasing another frame.
            fixed=target[:1].detach().float().contiguous()
            retained[key]=(target,fixed)
        retained.move_to_end(key)
        result=original(retained[key][1],dmax)
        while len(retained)>capacity:retained.popitem(last=False)
        return result.to(device=target.device,dtype=target.dtype)
    module._fold_boundary_distance_target=target_dt
    return retained

