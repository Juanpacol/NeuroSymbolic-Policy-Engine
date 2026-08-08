"""Hateful Memes Challenge dataset wrapper.

Loads the ``neuralcatcher/hateful_memes`` public mirror on the Hugging
Face Hub (no login/gating required, unlike the original DrivenData
challenge). Images are resolved lazily, one file at a time via
``hf_hub_download``'s own cache, so touching a handful of examples for
a smoke test does not require pulling the full ~3.4GB ``img/``
directory up front.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

_REPO_ID = "neuralcatcher/hateful_memes"


def _available_image_files() -> set[str]:
    """Lists image files actually present in the dataset repo.

    This mirror is missing image files for some rows (9664 images for
    12540 metadata rows, as of writing) -- a known gap in this
    unofficial repackaging, not a bug in this loader. Fetching the file
    listing once lets the dataset filter itself down to rows it can
    actually serve, instead of failing lazily on a missing file deep
    inside training.
    """
    files = HfApi().list_repo_files(_REPO_ID, repo_type="dataset")
    return {f for f in files if f.startswith("img/")}


class HatefulMemesDataset(Dataset[dict[str, Any]]):
    """Torch ``Dataset`` over the Hateful Memes Challenge (public mirror).

    Args:
        split: one of ``"train"``, ``"validation"``, or ``"test"``.
        transform: optional callable applied to each decoded RGB
            ``PIL.Image`` (e.g. a CLIP preprocessing transform). If
            ``None``, the raw PIL image is returned.
    """

    def __init__(
        self,
        split: str = "train",
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        # The specific error code depends on whether `datasets` (a
        # `train`-extra dependency) is installed in the checking
        # environment: import-not-found if absent, import-untyped if
        # present but unstubbed. A bare ignore covers both without
        # `warn_unused_ignores` (part of strict mode) flagging whichever
        # one didn't fire.
        from datasets import load_dataset  # type: ignore

        hf_split = load_dataset(_REPO_ID)[split]
        available = _available_image_files()
        self._hf = hf_split.filter(lambda row: row["img"] in available)
        self.transform = transform

    def __len__(self) -> int:
        """Number of examples with a resolvable image file."""
        return len(self._hf)

    def labels(self) -> list[int]:
        """Returns every label, without downloading any image.

        Reads the metadata column directly rather than going through
        :meth:`__getitem__`, which resolves (and therefore downloads)
        an image per call. Cheap enough to check a split's class
        balance before committing to a run -- worth doing, because this
        mirror's rows are ordered by label, so any head-of-split subset
        is single-class.
        """
        return list(self._hf["label"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Downloads (if needed) and returns one example.

        Args:
            idx: row index into the filtered split.

        Returns:
            A dict with ``id``, ``image`` (RGB, transformed if
            ``self.transform`` is set), ``text``, and ``label``.
        """
        row = self._hf[idx]
        image_path = hf_hub_download(
            repo_id=_REPO_ID, repo_type="dataset", filename=row["img"]
        )
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "id": row["id"],
            "image": image,
            "text": row["text"],
            "label": row["label"],
        }
