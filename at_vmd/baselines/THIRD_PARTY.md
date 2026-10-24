# Third-party code

## VMDNet and forecasting baselines

`at_vmd/baselines/third_party/tsl/` contains the minimal files used by PatchTST and
iTransformer, obtained from the MIT-licensed VMDNet repository:

- Source: https://github.com/weibin-feng/VMDNet
- Commit: `d72ba5c0d8b6d61a7e03300d396b33b42b22d914`
- License: [upstream MIT notice](third_party/tsl/LICENSE)

VMDNet builds on [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
Vendored imports were changed to package-relative imports. These baseline models
are not new implementations proposed by the AT-VMD authors. TimesNet and its
dedicated convolution blocks are not included in this release.

The mode-wise TCN in `forecasters.py` and the VMDNet baseline in
`at_vmd/baselines/vmdnet.py` follow VMDNet's implementation; the same upstream license
notice applies to adapted code. Our experiment settings are in `configs/`.

## Variational decomposition

`vmd.py` implements the VMD update convention used by VMDNet, whose README
acknowledges [vmdpy](https://github.com/vrcarva/vmdpy). AT-VMD adds window-conditional
frequency initialization and differentiation through a finite sweep.
`at_vmd/baselines/du_vmd.py` is a matched experimental baseline implemented here,
not the original MAGN/UVMD code release.

## Data

No datasets are redistributed. Dataset access and use remain subject to the
original providers' terms; this repository's software license does not license
their data.
