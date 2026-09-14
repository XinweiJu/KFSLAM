# Paper module names and implementation locations

| Paper module | Implementation |
|---|---|
| Motion-Guided Keyframe Observation Policy | `final/splatam/scripts/kfs4lam.py`: accumulated-motion proposal and redundancy veto; EndoGSLAM counterpart retains the evaluated native motion/reference/pair gates |
| Structure-Guided Mapping | `structure_guided.py`, `structure_guided_endogslam.py`, `fold_boundary_dt.py` |
| Restricted Temporal Deformation | `final/*/utils/slam_helpers.py`: `temporal_deformation_delta`, `transform_to_frame`; regularizers in the corresponding `scripts/kfs4lam.py` |
| DynMap | `dynamic_mapping_num_iters` and the corresponding EndoGSLAM budget function in the backbone scripts |
| Motion Proposal Only / MonoGS Frame Selection | `selection_ablations.py` |
| Appearance Pair Statistics | `networks/multiwrapper.py`: `compute_pair_proxies_pose_free` |

The backbone source retains inactive optional upstream paths for compatibility. The exported per-sequence configurations and `run.py` determine which components are enabled. The paper configurations do not require optional learned pose or ScoreNet checkpoints. Do not enable unexported legacy switches expecting the anonymous release to include those external models.
