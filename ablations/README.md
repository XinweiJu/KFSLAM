# Named component ablations

The directory names follow the paper terminology. The matrix below describes SplaTAM. The EndoGSLAM `motion_and_structure_guided` ablation disables deformation in its evaluated Full configuration and retains its own selection implementation. The evaluated EndoGSLAM Full implementation is in `final/endogslam/`.

| Configuration | Keyframe policy | DynMap | Fold-DT | Restricted deformation |
|---|---|---|---|---|
| backbone | Fixed interval 8 | No | No | No |
| dynmap | Fixed interval 8 | Yes | No | No |
| motion_guided | Accumulated-motion proposal + appearance veto | Yes | No | No |
| structure_guided | Fixed interval 8 | Yes | Yes | No |
| motion_and_structure_guided | Accumulated-motion proposal + appearance veto | Yes | Yes | No |
| motion_proposal_only | Accumulated-motion proposal, no veto | Yes | No | No |
| structure_guided_deformation | Fixed interval 8 | Yes | Yes | Yes |
| monogs_frame_selection | Adapted overlap + relative translation | No (25 iterations) | No | No |
| full (in final/) | Accumulated-motion proposal + appearance veto | Yes | Yes | Yes |

`structure_guided` means the DynMap-enabled Fold-DT configuration used in the paper table. It does not refer to a different frame selector. All deformation variants use the same restricted two-basis mean displacement. The backbone and motion-guided configurations are reconstructed from the evaluated corresponding configurations by switching off DynMap or removing the no-veto hook, respectively; they are not new measured runs.

The primary control is `structure_guided_deformation` vs `full`. This changes the keyframe policy while preserving the Fold-DT loss, deformation model and DynMap rule. Different retained history can still produce different actual mapping iteration counts. The learned map, deformation and trajectory are allowed to change; this is a system-level comparison, not frozen-deformation or equal-compute evaluation.

The MonoGS adaptation uses a minimum interval of 10, overlap threshold 0.95, relative translation thresholds 0.04/0.02 of rendered median depth, and overlap-based bootstrap for the first 10 retained keyframes. It rerenders both visibility sets on the current SplaTAM map and retains SplaTAM's mapping/replay machinery. It is neither the full MonoGS backend nor forced fixed-10 selection.
