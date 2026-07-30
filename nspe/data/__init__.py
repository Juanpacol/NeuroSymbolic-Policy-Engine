"""Synthetic and real dataset helpers.

`HatefulMemesDataset` is intentionally not imported here: it depends on
the optional `data` extra (`datasets`, `huggingface_hub`), while this
package's synthetic helpers only need `torch`. Import it directly from
`nspe.data.hateful_memes` when that extra is installed.
"""

from nspe.data.synthetic import make_layered_policy

__all__ = ["make_layered_policy"]
