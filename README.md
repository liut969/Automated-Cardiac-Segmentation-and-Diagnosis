# Residual Convolutional Neural Network and Bidirectional-Convolutional LSTM for Automated Cardiac Segmentation and Diagnosis
## Step
- 1, Register and download ACDC-2017 dataset from [https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html)
- 2, Create a folder outside the project with name ```data``` and copy the dataset
- 3, Run the scipt ```main.py```
- 4, The segmentation results are saved in ```./predict_nii_gz_result```
- 5, The diagnosis result are saved in ```./diagnose_result.txt```
## Evaluation score
- 1, Submit the segmentation result ```./predict_nii_gz_result``` to the [online evaluation website](https://acdc.creatis.insa-lyon.fr/#challenges) and get the segmentation score
- 2, Submit the diagnosis result ```./diagnose_result.txt``` to the [online evaluation website](https://acdc.creatis.insa-lyon.fr/#challenges) and get the diagnose score
