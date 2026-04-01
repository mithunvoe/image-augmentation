from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image


class Augmentation(Protocol):
    def __call__(
        self,
        image: Image.Image,
        rng: np.random.Generator,
        **kwargs: object,
    ) -> Image.Image: ...
