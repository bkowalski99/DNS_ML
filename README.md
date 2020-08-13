# DNS Simulation Assistance by Ben Kowalski

Step by Step Guide:
1. Use datainput.py to load in data from the Ensight-3D simulation of a mixed, turbulent engine.
   
   a. Enter in the path to the files you are reading in. 
   
   b. Enter in the name of the values.
2. Use normalizeInputData.py to normalize the values read in by the previous step.
3. Run either alternateCNNShape.py or cnnFromTurbulentEngine.py to train a network to predict the variance fields given the smoothed fields.

    a. alternateCNNShape.py uses the direct convolutional shape used in the Seltz Domingo paper.
    
    b. cnnFromTurbulentEngine.py uses the U-Net model proposed by Ronneberger's team.
