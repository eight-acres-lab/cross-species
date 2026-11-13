# Third-party notices

The MIT License applies to the original software and software documentation in this repository. Third-party material retains the terms identified below and in its source records. Manuscript text, publication figures, datasets and model parameters are outside the scope of this software license.

## DiffusionNFT adaptation

`src/cross_species/scdfm_nft.py` adapts the negative-aware flow fine-tuning loss from [NVIDIA DiffusionNFT](https://github.com/NVlabs/DiffusionNFT/blob/cbb14f84b8312b620390dfcbe2fab69c1383b104/scripts/train_nft_sd3.py), commit `cbb14f84b8312b620390dfcbe2fab69c1383b104`, lines 910–970. It uses scDFM's noise-to-data convention and exposes the loss as standalone functions.

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This adaptation is licensed under [Apache License 2.0](LICENSES/Apache-2.0.txt). Its source file retains the copyright and license notice.

## Illustration assets

Servier Medical Art illustrations use [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). DBCLS macaque artwork uses [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution, modifications and source links are recorded in the [figure credits](figures/README.md#artwork-credits) and [asset manifest](figures/assets/sources.json).

## Data and models

Source datasets and pretrained model files are governed by their providers' licenses and terms. The [data acquisition records](reproduction/data_acquisition.json), experiment configurations and input manifests identify their sources and applicable terms.
