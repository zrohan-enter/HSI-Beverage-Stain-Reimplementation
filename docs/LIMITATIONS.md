# Limitations and Interpretation Rules

1. The original beverage-stain dataset and official code are not publicly available in the paper repository used for this project.
2. Accuracy obtained on Salinas measures performance on agricultural hyperspectral classes, not beverage stains.
3. Accuracy obtained on synthetic beverage spectra measures software behavior, not real forensic generalization.
4. Published paper values and local values must not be described as a direct comparison unless the same original data, split protocol, and implementation details are available.
5. The initial implementation uses a pixel-wise stratified split to resemble the paper's reported protocol. Spatially or instance-grouped splitting is a stronger future validation because nearby pixels may be correlated.
6. Several architecture details were not fully specified in the article; documented PyTorch choices are independent reconstruction decisions.
7. A successful faculty demonstration proves that the computational pipeline is operational. It does not by itself prove deployment readiness in forensic practice.
