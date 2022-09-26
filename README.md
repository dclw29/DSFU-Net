# ML-DiffuseReader
Method to read diffuse scattering images and decompose them

# Todo still
1) Prepare dataset (conditional images and output)
2) Get architecture to train both networks sep.
3) See if we can at least get reasonable image output from initial trainings
4) Run ablation tests on level of training with cross product of output contributing to MSE loss
5) Different losses? E.g. pix2pixHD, WGAN...?
6) Diffuse network if this fails?

Run conditions using dFF and SRO data into diffusion denoising model as extra channels?
Include extra loss from multiplier of the conditions (no weighting)
