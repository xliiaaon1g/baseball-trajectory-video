# License and asset provenance

No new open-source license has been assigned to the project code in this release.
Third-party model weights are downloaded from their official repositories and are
not redistributed here. Consult each upstream model and ProPainter license before reuse.
Broadcast-derived example images/video and Statcast records remain third-party assets;
this repository does not grant a new license to them. The full raw-data archive is
kept separately from the Git source tree. No cloud credential, account screenshot,
or private endpoint is required to reproduce the numerical demo.

## Scope of this upload

The repository landing page and Phase I Release do not assign a blanket license to the project. The owner's original project code and writing remain without a newly granted open-source license. Third-party rights are not replaced by ownership of this repository. Keeping the repository private is an access setting, not copyright clearance.

The supplied `Informal-report.pdf` is uploaded unchanged. Its real broadcast frames, extracted ball images and experimental visuals retain their source rights. Generated/composited examples may retain identifiable source imagery; no exclusive ownership or unrestricted reuse is asserted. Statcast-derived records are identified as reference data from Baseball Savant; public accessibility is not a new redistribution license.

## Third-party sources and model revisions

| Component | Source / applicable notice |
|---|---|
| ProPainter | [Official repository](https://github.com/sczhou/ProPainter), recorded commit `e870e79321c31b733e2031af5aa2fb1fe3ac7eec`. [S-Lab License 1.0 at that commit](https://github.com/sczhou/ProPainter/blob/e870e79321c31b733e2031af5aa2fb1fe3ac7eec/LICENSE) permits use/redistribution for non-commercial purposes subject to its conditions; commercial use requires contacting the contributors. No upstream ProPainter source or weights are included in the project package. |
| AnimateDiff motion adapter / motion LoRA | [Official implementation](https://github.com/guoyww/AnimateDiff); model repositories and exact revisions are recorded in `configs/generative_models.json` inside the project ZIP. Consult each model's own terms. |
| SparseCtrl RGB | [Official model repository](https://huggingface.co/guoyww/animatediff-sparsectrl-rgb); recorded revision `b8003d681d813c095e459b9141122894daff2d13`. Consult its model and base-model licenses. |
| Realistic Vision base | [Official model repository](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE); recorded revision `1e9f017a7b1eaefb63a1900ea6c5953d2739fd21`. Consult its model license and inherited terms. |
| SD VAE ft-MSE | [Official model repository](https://huggingface.co/stabilityai/sd-vae-ft-mse); recorded revision `31f26fdeee1355a5c34592e401dd41e45d25a493`. Consult its model license. |
| Diffusers, PyTorch, Transformers and other dependencies | Installed separately using recorded requirements; retain their own licenses. Listing a dependency does not relicense it. |
| Baseball Savant / Statcast | [Official parameter documentation](https://baseballsavant.mlb.com/csv-docs). Reference parameters are not independent frame-level ground truth; underlying data rights remain with the relevant rights holders. |
| Broadcast-derived frames, clips and ball sprites | Examples from the original baseball broadcast. Relevant broadcast/event/media rights remain with their rights holders; no public-distribution or commercial clearance is asserted. |

The project's small trained ODE checkpoint is included as a recorded project artifact; no third-party generative checkpoint is bundled. Historical adapters and planning files are project workflow records and do not grant rights to the external software they reference.
