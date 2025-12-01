# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import enum


class ModelType(enum.Enum):
    """Model type."""

    encoder_or_decoder = 1
    retro_encoder = 2
    retro_decoder = 3

    @property
    def encoder_and_decoder(self):
        """Deprecated property - use encoder_or_decoder instead."""
        raise ValueError(
            "ModelType.encoder_and_decoder is deprecated. Please use ModelType.encoder_or_decoder "
            "instead."
        )


class Fp8Recipe(str, enum.Enum):
    """FP8 recipe names: delayed, tensorwise, mxfp8, blockwise."""

    delayed = "delayed"
    tensorwise = "tensorwise"
    mxfp8 = "mxfp8"
    blockwise = "blockwise"


class Fp4Recipe(str, enum.Enum):
    """FP4 recipe names: nvfp4."""

    nvfp4 = "nvfp4"
    metis_te_fp4 = "metis_te_fp4"

class MetisRecipe(str, enum.Enum):
    """Metis recipe names: metis_te, metis_persudo."""

    metis_te = "metis_te"
    metis_persudo = "metis_persudo"