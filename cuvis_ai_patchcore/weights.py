"""Weight declarations of the patchcore plugin.

Side-effect free on purpose: this module only declares. ``cuvis_ai_patchcore/__init__``
registers the tuple with cuvis-ai-core's ``ModelWeights`` at import, and cuvis-ai's
``emit_metadata`` projects it into the plugin manifest's ``weights:`` block, so CuvisNEXT and
the installer know what to provision without importing the plugin.

The rows are the six Cubert-trained walnut foreign-object pipelines published at
``cubert-gmbh/Industrial_Foreign_Object_Detection_Walnuts``: one fitted state (``.pt``)
per pipeline with its yaml and the two shell-segmentation weight files as aux files, so
``download-model download <name>`` fetches a runnable folder. The detector of every pipeline is
``PatchCoreDetector``; the SteerViT features and the RF-DETR shell segmentation come from the
steervit and rfdetr plugins named in the yaml.
"""

from __future__ import annotations

from cuvis_ai_schemas.plugin import AuxFile, PluginWeightEntry

PLUGIN_NAME = "patchcore"
"""The manifest name of this plugin (what pipelines list under ``plugins:``)."""

_WALNUT_REPO = "cubert-gmbh/Industrial_Foreign_Object_Detection_Walnuts"
_WALNUT_REVISION = "f4b660aa1e2598ec4bc9354474c942bdbc485fe8"
"""The published commit of the walnut model repository (40 hex digits)."""

_SHELL_AUX_FILES = (
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
)
"""The two shell-segmentation weight files every walnut pipeline ships alongside its checkpoint."""


def _walnut_pipeline(
    name: str,
    display_name: str,
    summary: str,
    *,
    sha256: str,
    size_bytes: int,
    yaml_size_bytes: int,
    yaml_sha256: str,
    description: str,
) -> PluginWeightEntry:
    return PluginWeightEntry(
        name=name,
        display_name=display_name,
        summary=summary,
        used_for=["Anomaly detection", "Trained pipeline"],
        kind="trained_pipeline",
        repo_id=_WALNUT_REPO,
        filename=f"{name}/{name}.pt",
        revision=_WALNUT_REVISION,
        sha256=sha256,
        size_bytes=size_bytes,
        aux_files=[
            AuxFile(path=f"{name}/{name}.yaml", size_bytes=yaml_size_bytes, sha256=yaml_sha256),
            *_SHELL_AUX_FILES,
        ],
        license="Apache-2.0",
        license_file="LICENSE",
        description=description,
    )


WEIGHTS: tuple[PluginWeightEntry, ...] = (
    _walnut_pipeline(
        "walnut_best_refit",
        "Walnut foreign objects, refit (stand pick)",
        "Walnut foreign objects, refit banks, robust mask",
        sha256="8f1771038e729849fe9241bd76b23e32f4c84da341b592fd6a4027d02b60005c",
        size_bytes=916_907_547,
        yaml_size_bytes=8_830,
        yaml_sha256="9df82c67c688770ccb21258866b53ce38f780a7007e261ceb28b4e785788372f",
        description=(
            "Walnut foreign-object pipeline with the 1 October refit banks and the robust mask, "
            "the first choice at the production stand."
        ),
    ),
    _walnut_pipeline(
        "walnut_best_original",
        "Walnut foreign objects, original",
        "Walnut foreign objects, original banks, robust mask",
        sha256="81a311b1aa1c598c6dff78f280dc93dc6596a0b0d2c898106ab8346734aea350",
        size_bytes=916_909_009,
        yaml_size_bytes=8_834,
        yaml_sha256="9c1c2c7de1738327d86f8893e70f579f3ae8e4f1d4f2c88e2187280c40fe2a54",
        description=(
            "Walnut foreign-object pipeline with the original banks and the robust mask, for "
            "lighting other than the production stand."
        ),
    ),
    _walnut_pipeline(
        "walnut_best_refit_cut",
        "Walnut foreign objects, refit, cut mask",
        "Refit banks, robust mask cut to the objects",
        sha256="7110eb0f13011f2c6dca0a0197f75220ef27fe1a80dfababee95d040890055da",
        size_bytes=916_909_347,
        yaml_size_bytes=10_654,
        yaml_sha256="a3239920f97b8caedcca4d2840ecfa70e2ee1372f7eefe41b3c8d6730417c94c",
        description=(
            "The refit walnut pipeline with the mask cut to the objects of the frame, for a "
            "tighter display at a small cost per frame."
        ),
    ),
    _walnut_pipeline(
        "walnut_best_original_cut",
        "Walnut foreign objects, original, cut mask",
        "Original banks, robust mask cut to the objects",
        sha256="6460527319d97f3161e343cc70b19a880a8ec352cff7e9530dcc546c1880edf6",
        size_bytes=916_910_873,
        yaml_size_bytes=10_656,
        yaml_sha256="ccb98a65ee406c65835ec419dd14e20ba42e97552f9cacce811f65545a8b74da",
        description=(
            "The original walnut pipeline with the mask cut to the objects of the frame, for a "
            "tighter display at a small cost per frame."
        ),
    ),
    _walnut_pipeline(
        "walnut_best_refit_shellaware",
        "Walnut foreign objects, refit, shell-aware",
        "Refit banks, FO map damped on segmented shells",
        sha256="9cb66ec53adb1d2b0eb6b1606f611bf76c151086ca11b8a225a2b4111fee0d47",
        size_bytes=916_912_673,
        yaml_size_bytes=9_037,
        yaml_sha256="9fe7c9cb39d3afa152204f9978ff05c0cb94cf0efc2dee7db7330be2f4b8fb15",
        description=(
            "The refit walnut pipeline with the foreign-object map damped on the segmented "
            "shells, for stands where real shells next to a foreign object light up."
        ),
    ),
    _walnut_pipeline(
        "walnut_best_original_shellaware",
        "Walnut foreign objects, original, shell-aware",
        "Original banks, FO map damped on segmented shells",
        sha256="dd454509794914049275fc28da66b6a602f6a5a8e92a45dd471791154a0910f5",
        size_bytes=916_914_135,
        yaml_size_bytes=9_013,
        yaml_sha256="d2224fe10f5988e12a7d4f696309a9925c7c96daf83db912b7b04010480e97fd",
        description=(
            "The original walnut pipeline with the foreign-object map damped on the segmented "
            "shells, for stands where real shells next to a foreign object light up."
        ),
    ),
)
"""Every trained pipeline the patchcore plugin publishes."""
