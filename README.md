# PredictiveQualityMonitoring
Master Thesis - Autonomous Systems

# How to run the code
In the folder /Data/ there is a lot of vairous dataset all created from a central data. 
The Aritifical neural network makes use of the /Data/ANN_train.csv.
If you wish to test this out simply download the repo, run pip --install requriments.txt and then run main.py.

#### Note on the DEA
OBS: This is only relevant if you want to update the data, or test out other data!
The DEA efficenciy scores are calculated in matlab. In order to use the correct values for traning of the ANN, make sure that you put in the Mixed_transform_noClass_unNorm.csv in the deatoolbox folder. Produced by the function buildMixedData() in main.py. Then you can run the dea_onData.m script to get the eff_mixed.csv file out. Place this file into the /Data/ folder and then run the BuildDataForANN() function in main.py before traning the ANN. 

The DEA toolbox has been developed by Álvarez Inmaculada C., Barbero Javier, and Zofío José L. 
The toolbox can be accessed here: https://se.mathworks.com/matlabcentral/fileexchange/56025-data-envelopment-analysis-toolbox-for-matlab. Go check them out!

# ANN
The ANN is a simple fully connected layer algortihm developed in pytorch. It makes use of ReLU activations and Adam optimzation with a learning rate of 10e-5. It aims to predict the DEA effcicency score produces by the script dea_onData.m.


