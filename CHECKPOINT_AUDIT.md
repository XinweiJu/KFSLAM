# Released reconstruction audit

The result archives contain the completed reconstructions for the 25 evaluated sequences with each backbone. The checkpoint is authoritative for rendering; configuration flags alone do not prove that a component was instantiated.

| Backbone | Checkpoints | Temporal coefficients in checkpoint | Reconstruction replay |
|---|---:|---|---|
| SplaTAM | 25 | Present | Time-conditioned map |
| EndoGSLAM | 25 | Absent | Static Gaussian map at saved estimated poses |

The historical EndoGSLAM configurations contain `temporal_deformation.enable=true`, but its parameter initializer checks `MPKFS_TEMPORAL_DEFORMATION_ENABLE`. The evaluated queue clears that environment variable and the entry point calls `rgbd_slam` without setting it. All 25 released EndoGSLAM checkpoints lack `deform_coeffs`; the renderer's deformation helper returns zero when that parameter is absent. Therefore these particular completed results must **not** be described as evidence of active restricted temporal deformation. Their adaptive observation policy, dynamic mapping budget and fold-boundary supervision remain represented in the evaluated configuration.

The portable entry point preserves this historical behavior. Setting the environment variable to enable EndoGSLAM deformation would define a new experiment, not reproduce these checkpoints. The SplaTAM deformation results are distinct and do contain the six coefficients per Gaussian.

The historical EndoGSLAM trajectory illustration in the project page is a keyframe-policy example, not a trajectory generated from these newly exported checkpoints. Reconstruction videos, in contrast, are rendered directly from the exported final parameters.

The per-archive manifest reports actual parameter presence. Parameters are numeric, loaded with `allow_pickle=False`, and retain their original values. Reconstruction archives contain no original machine paths or experiment logs.
