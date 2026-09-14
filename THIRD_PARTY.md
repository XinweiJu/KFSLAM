# Third-party implementations

The backbone implementations derive from SplaTAM and EndoGSLAM. Attribution and applicable upstream copyright notices are retained; they identify third-party projects, not the authors of this anonymous submission.

- SplaTAM: https://github.com/spla-tam/SplaTAM — license in LICENSES/SplaTAM-LICENSE.
- EndoGSLAM: https://github.com/Loping151/EndoGSLAM — its original code notices are retained in the backbone files. This release does not relicense upstream material.
- Depth-enabled Gaussian rasterizer: https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth — install separately at the commit stated in README.
- MonoGS visibility rasterizer: third_party/monogs_rasterizer contains the original source and its LICENSE.md. It is used only for the adapted MonoGS frame-selection ablation.
- C3VD / C3VD-v2 data and JMEES-derived fold responses are external inputs, not included in this code release. Dataset use remains subject to the original terms.

No dataset, inference checkpoint, personal credentials, experiment directory, or Git history is bundled. Qualitative figures are included as paper illustrations.
