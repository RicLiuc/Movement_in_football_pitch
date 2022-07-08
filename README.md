# Movement in a football pitch

The project aims to create a tool that can predict the future occupancy of a football pitch, starting from the previous pitch occupancies. 

The data used develop the model have been provided by SkillCorner, a set of these data are provided open source in:
https://github.com/SkillCorner/opendata

The data used in the training and in the testing phase regards the 2019/2020 Serie A.

Moreover, we used also WyScout event dataset to enrich the model's prediction.


The model used for the prediction has been the OST-ResNet, a model which slightly modify ST-ResNet (https://arxiv.org/pdf/1610.00081.pdf).
ST-ResNet is model commonly used in the prediction of the human mobility. 
OST-ResNet changes the paradigm of ST-ResNet (from inflow/outflow to occupancy) and it reduces the number of temporal decomposition done on the data (from 3 to 2) to adapt the problem to the football environment.
A visualization of the architecture of OST-ResNet is showed in the picture below.

<img src="/images/OST.png" alt="OST-ResNet" width="400" height="400">



To start the project:
```
python main.py
```

Then just type the team to analyze. It will start the training if the model is not presented and it will save the dataset and the prediction for the analysis.


The nootebooks Start_Explaination_Period and Start_Explaination_Closeness allow to understand how a change of a position in a sequence will modify the positions of the teams. Opening the notebooks it is possible to see all the instruction to use it.

The notebook StartEEE analyzes which different starting situation will bring to the same result. Opening the notebook it is possible to see all the instruction to use it.

# Description of the files

- main.py: module to start the project, call all the other modules and links them.
- Create_dataset.py: module to create the whole dataset to fit the OST-ResNet.
- ST_ResNet.py: module with the model and all its function to use it.
- MinMaxNorm.py: class to normalize the data in the range [-1, 1].
- params.py: file with all the parameters used in the model. To modify one parameter just change its value in it.
- path.py: file with all the path for save or access the data.
- utils.py: some additional function.
