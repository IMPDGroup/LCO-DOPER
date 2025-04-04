# Unraveling Doping Effects in LaCoO3 via Machine Learning-Accelerated First-Principles Simulations


<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567) -->

## Overview

An application for analyzing dopant effects on LaCoO3 (LCO), supporting composition optimization and material performance enhancement.

## Get Started

* 🌐 [Try it Online](https://lco-doper.streamlit.app/)

* For any questions or suggestions, please contact us (gliu4@wpi.edu; yzhong@wpi.edu). Please STAR  ⭐️ this repository if you find it helpful :)

### Usage Description

1. **Select Dopants and Concentration**: In the `Dopants Selection` section, choose the desired dopants and their concentrations (at.%) for both A and B sites in the LCO structure. Supported dopants include:
    - A-site: Mg, Ca, Sr, Ba, Ce, Pr, Nd, Sm, Gd
    - B-site: Sc, Ti, V, Cr, Mn, Fe, Ni, Cu, Zn, Al, Ga

    <div align=left><img src='./res/dopants_table.jpg' alt='' width=''/></div>

2. **Select System Conditions**: In the `System Conditions` section, specify the oxygen vacancy concentration (at.%) and the temperature (K) for the simulation. The temperature range is from 1000 K to 2500 K. The oxygen vacancy concentration ranges from 0 at.% to 5 at.%.

3. **Predict**: Click the `Predict` button to initiate the prediction process. The system will analyze the selected dopants and conditions, and provide predictions for the following properties:
    - Formation Energy (eV/atom)
    - Diffusion Coefficient (cm²/s)
    - Lattice Distortion (%)
    - Atomic Distance (Å)

4. **Download Results**: After the prediction is complete, you can download the results in CSV format by clicking the `Download predicted data` button. The results will include the predicted properties for the selected dopants and conditions.

5. **Visualize Results**: The predicted results will be displayed in the `Predicted Data Visualization:` section, where you can visualize the predicted properties with interactive plots. The plots will show the relationship between the selected dopants, their concentrations, and the predicted properties.

## Cite Us
If you use this code in your research, please cite our paper:

```
@article{
  title={Unraveling Doping Effects in LaCoO3 via Machine Learning-Accelerated First-Principles Simulations},
  author={G. Liu and S. Yang and Y. Zhong},
  journal={},
  volume={},
  number={},
  pages={},
  year={2025}
}
```