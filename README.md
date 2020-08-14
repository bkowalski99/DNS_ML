# DNS Simulation Assistance by Ben Kowalski

Step by Step Guide:
If you want to train using the 8 x 8 matrices jump to step 4.

1. Use dataInput.py to load in data from the Ensight-3D simulation of a mixed, turbulent engine.
   
    a. Enter in the path to the files you are reading in. 
   
    b. Enter in the name of the values.
2. Use normalizeInputData.py to normalize the values read in by the previous step.
3. Run either alternateCNNShape.py or cnnFromTurbulentEngine.py to train a network to predict the variance fields given the smoothed fields.

    a. alternateCNNShape.py uses the direct convolutional shape used in the Seltz Domingo paper.
    
    b. cnnFromTurbulentEngine.py uses the U-Net model proposed by Ronneberger's team.

4. Use dataInput8By8.py to load in data from the Ensight-3D simulation of a mixed, turbulent engine.
   
    a. Enter in the path to the files you are reading in. 
   
    b. Enter in the name of the values.
   
2. Use normalizeInputData8By8.py to normalize the values read in by the previous step.
3. Run either alternateCNNShape8x8.py to train the network.
    
