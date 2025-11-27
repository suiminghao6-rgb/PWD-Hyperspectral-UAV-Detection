# PWD-Hyperspectral-UAV-Detection
Comparative analysis of spectral methods for pine wilt disease detection using unattended UAV-based hyperspectral imagery.
Overview
This repository contains the codes and sample data used for the comparative analysis of spectral methods for pine wilt disease (PWD) detection based on unattended UAV hyperspectral imagery.
The study evaluates four representative methods that use discrete, partially continuous, and full-spectrum information.
Methods
PWDAI：A vegetation index calculated from three discrete wavelengths (550, 680, and 927 nm).
SAM：Similar Bands Selection–New algorithm based on spectral angle similarity, optimized by a class-separability factor K to select key spectral regions.
PLS–DA：Partial Least Squares Discriminant Analysis using full-spectrum data for both binary and four-class classification.
1D–CNN ：Deep learning model applied to the full hyperspectral spectrum for automated feature extraction and classification.
Data
The included hyperspectral datasets were collected using unattended UAV systems over pine forests affected by pine wilt disease.
The dataset includes labeled samples of healthy, diseased, shadowed, and mixed pixels.
Only representative subsets are provided for demonstration.  
Full datasets and detailed metadata are available upon request or through the public data repository linked in the paper.
Usage
Clone this repository
git clone https://github.com/suiminghao6-rgb/PWD-Hyperspectral-UAV-Detection
cd PWD-UAV-Hyperspectral-Analysis
