"""Weight declarations of the patchcore plugin.

Side-effect free on purpose: this module only declares. ``cuvis_ai_patchcore/__init__``
registers the tuple with cuvis-ai-core's ``ModelWeights`` at import, and cuvis-ai's
``emit_metadata`` projects it into the plugin manifest's ``weights:`` block, so CuvisNEXT and
the installer know what to provision without importing the plugin.

The rows are the six Cubert-trained walnut foreign-object pipelines published at
``cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts``: one fitted state (``.pt``)
per pipeline with its yaml and the two shell-segmentation weight files as aux files, so
``download-model download <name>`` fetches a runnable folder. The detector of every pipeline is
``PatchCoreDetector``; the SteerViT features and the RF-DETR shell segmentation come from the
steervit and rfdetr plugins named in the yaml.
"""

from __future__ import annotations

from cuvis_ai_schemas.plugin import AuxFile, PluginWeightEntry

PLUGIN_NAME = "patchcore"
"""The manifest name of this plugin (what pipelines list under ``plugins:``)."""

_WALNUT_REPO = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Walnuts"
_WALNUT_REVISION = "0000000000000000000000000000000000000000"
"""The published commit of the walnut model repository (40 hex digits)."""

WEIGHTS: tuple[PluginWeightEntry, ...] = (
    PluginWeightEntry(
        name="walnut_best_refit",
        display_name="Walnut foreign objects, refit (stand pick)",
        summary="Walnut foreign objects, refit banks, robust mask",
        used_for=["Anomaly detection", "Trained pipeline"],
        kind="trained_pipeline",
        repo_id=_WALNUT_REPO,
        filename="walnut_best_refit/walnut_best_refit.pt",
        revision=_WALNUT_REVISION,
        sha256="8ef10835d72f58cdd2e7d4e09a53e2eed5955d45cbf0891e71c8243b09c17e3f",
        size_bytes=916_906_227,
        aux_files=[
            AuxFile(
                path="walnut_best_refit/walnut_best_refit.yaml",
                size_bytes=8_753,
                sha256="e41150ce0747ee913fa5ba8f1473d889dc6b45fd77d50c638bfff8aad0ba6ec9",
            ),
            AuxFile(
                path="weights/rgb_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="9e6dcf4d1ac6d86e8520935e217145b904a702d66f6b4c161890234544fdebf8",
            ),
            AuxFile(
                path="weights/cir_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="018d7714ef61219dc88bdf5507135cd8fc8034d55a0a84d38cece151005e0929",
            ),
        ],
        license="Apache-2.0",
        license_file="LICENSE",
        description=(
            "Walnut foreign-object pipeline with the 1 October refit banks and the robust mask, "
            "the first choice at the production stand."
        ),
    ),
    PluginWeightEntry(
        name="walnut_best_original",
        display_name="Walnut foreign objects, original",
        summary="Walnut foreign objects, original banks, robust mask",
        used_for=["Anomaly detection", "Trained pipeline"],
        kind="trained_pipeline",
        repo_id=_WALNUT_REPO,
        filename="walnut_best_original/walnut_best_original.pt",
        revision=_WALNUT_REVISION,
        sha256="0b8812933b7585017afea2cf5558d3aa2aa6dbb1ce5e7044aa91e4c9b24062f5",
        size_bytes=916_907_738,
        aux_files=[
            AuxFile(
                path="walnut_best_original/walnut_best_original.yaml",
                size_bytes=8_757,
                sha256="ba08d05e285eeca0cca8b651e6e8f786015279a52ce8d75c50bd050a7420d2a3",
            ),
            AuxFile(
                path="weights/rgb_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="9e6dcf4d1ac6d86e8520935e217145b904a702d66f6b4c161890234544fdebf8",
            ),
            AuxFile(
                path="weights/cir_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="018d7714ef61219dc88bdf5507135cd8fc8034d55a0a84d38cece151005e0929",
            ),
        ],
        license="Apache-2.0",
        license_file="LICENSE",
        description=(
            "Walnut foreign-object pipeline with the original banks and the robust mask, for "
            "lighting other than the production stand."
        ),
    ),
    PluginWeightEntry(
        name="walnut_best_refit_cut",
        display_name="Walnut foreign objects, refit, cut mask",
        summary="Refit banks, robust mask cut to the objects",
        used_for=["Anomaly detection", "Trained pipeline"],
        kind="trained_pipeline",
        repo_id=_WALNUT_REPO,
        filename="walnut_best_refit_cut/walnut_best_refit_cut.pt",
        revision=_WALNUT_REVISION,
        sha256="a1d3ef939263e9ee558cf810f86bb806053ac924dbe6662ca2bdc985f37dbc95",
        size_bytes=916_908_007,
        aux_files=[
            AuxFile(
                path="walnut_best_refit_cut/walnut_best_refit_cut.yaml",
                size_bytes=10_642,
                sha256="ca858dc0b6bb1d2cd993c6ab2ac583a33f2ad260b8e53386c52ada1b1b18f469",
            ),
            AuxFile(
                path="weights/rgb_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="9e6dcf4d1ac6d86e8520935e217145b904a702d66f6b4c161890234544fdebf8",
            ),
            AuxFile(
                path="weights/cir_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="018d7714ef61219dc88bdf5507135cd8fc8034d55a0a84d38cece151005e0929",
            ),
        ],
        license="Apache-2.0",
        license_file="LICENSE",
        description=(
            "The refit walnut pipeline with the mask cut to the objects of the frame, for a "
            "tighter display at a small cost per frame."
        ),
    ),
    PluginWeightEntry(
        name="walnut_best_original_cut",
        display_name="Walnut foreign objects, original, cut mask",
        summary="Original banks, robust mask cut to the objects",
        used_for=["Anomaly detection", "Trained pipeline"],
        kind="trained_pipeline",
        repo_id=_WALNUT_REPO,
        filename="walnut_best_original_cut/walnut_best_original_cut.pt",
        revision=_WALNUT_REVISION,
        sha256="1966c03d2d0831bca1b195592bc4de86d6aed37f32a274bfa9b3e92a67fdb699",
        size_bytes=916_909_582,
        aux_files=[
            AuxFile(
                path="walnut_best_original_cut/walnut_best_original_cut.yaml",
                size_bytes=10_644,
                sha256="0e124a0e812da2ea21f98a54c6eec3043fb41dd47d449235a94fbe442e4cddbe",
            ),
            AuxFile(
                path="weights/rgb_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="9e6dcf4d1ac6d86e8520935e217145b904a702d66f6b4c161890234544fdebf8",
            ),
            AuxFile(
                path="weights/cir_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="018d7714ef61219dc88bdf5507135cd8fc8034d55a0a84d38cece151005e0929",
            ),
        ],
        license="Apache-2.0",
        license_file="LICENSE",
        description=(
            "The original walnut pipeline with the mask cut to the objects of the frame, for a "
            "tighter display at a small cost per frame."
        ),
    ),
    PluginWeightEntry(
        name="walnut_best_refit_shellaware",
        display_name="Walnut foreign objects, refit, shell-aware",
        summary="Refit banks, FO map damped on segmented shells",
        used_for=["Anomaly detection", "Trained pipeline"],
        kind="trained_pipeline",
        repo_id=_WALNUT_REPO,
        filename="walnut_best_refit_shellaware/walnut_best_refit_shellaware.pt",
        revision=_WALNUT_REVISION,
        sha256="9283acf93e16de371e30a3ed9cf6e28c4c02410101669f3e92f6d1217d141ef0",
        size_bytes=916_911_298,
        aux_files=[
            AuxFile(
                path="walnut_best_refit_shellaware/walnut_best_refit_shellaware.yaml",
                size_bytes=8_968,
                sha256="67b9de9503f38c82ebdcd7658bf17451e892e9739dea43cc2d7670831b414ce4",
            ),
            AuxFile(
                path="weights/rgb_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="9e6dcf4d1ac6d86e8520935e217145b904a702d66f6b4c161890234544fdebf8",
            ),
            AuxFile(
                path="weights/cir_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="018d7714ef61219dc88bdf5507135cd8fc8034d55a0a84d38cece151005e0929",
            ),
        ],
        license="Apache-2.0",
        license_file="LICENSE",
        description=(
            "The refit walnut pipeline with the foreign-object map damped on the segmented "
            "shells, for stands where real shells next to a foreign object light up."
        ),
    ),
    PluginWeightEntry(
        name="walnut_best_original_shellaware",
        display_name="Walnut foreign objects, original, shell-aware",
        summary="Original banks, FO map damped on segmented shells",
        used_for=["Anomaly detection", "Trained pipeline"],
        kind="trained_pipeline",
        repo_id=_WALNUT_REPO,
        filename="walnut_best_original_shellaware/walnut_best_original_shellaware.pt",
        revision=_WALNUT_REVISION,
        sha256="6d32cadefa4d5f99cfaa00fca6ad8d9d3e866f9a65175cc1ab28446fa83775fa",
        size_bytes=916_912_809,
        aux_files=[
            AuxFile(
                path="walnut_best_original_shellaware/walnut_best_original_shellaware.yaml",
                size_bytes=8_944,
                sha256="429179147ecf93a93be410a40a0d398932966791f8f8ce7e63ddd540e78a1524",
            ),
            AuxFile(
                path="weights/rgb_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="9e6dcf4d1ac6d86e8520935e217145b904a702d66f6b4c161890234544fdebf8",
            ),
            AuxFile(
                path="weights/cir_v2_ema.pth",
                size_bytes=143_694_121,
                sha256="018d7714ef61219dc88bdf5507135cd8fc8034d55a0a84d38cece151005e0929",
            ),
        ],
        license="Apache-2.0",
        license_file="LICENSE",
        description=(
            "The original walnut pipeline with the foreign-object map damped on the segmented "
            "shells, for stands where real shells next to a foreign object light up."
        ),
    ),
)
"""Every trained pipeline the patchcore plugin publishes."""
