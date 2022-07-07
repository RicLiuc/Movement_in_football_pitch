# Movement in a football pitch

The project aim to create a tool that can predict the future occupancy of a football pitch, starting from the previous pitch occupancies. 

The data used develop the model have been provided by SkillCorner, a set of these data are provided open source in:
https://github.com/SkillCorner/opendata
The data are about the 2019/2020 Serie A.

The model used for the prediction has been the OST-ResNet, a model which slightly modify ST-ResNet (https://arxiv.org/pdf/1610.00081.pdf).
ST-ResNet is model commonly used in the prediction of the human mobility. 
OST-ResNet changes the paradigm of ST-ResNet (from inflow/outflow to occupancy) and it reduces the number of temporal decomposition done on the data (from 3 to 2) to adapt the problem to the football environment.



To start the project:
```
python main.py
```

Then just type the team to analyze. It will start the training if the model is not presented and it will save the dataset and the prediction for the analysis.


The nootebooks Start_Explaination_Period and Start_Explaination_Closeness allow to understand how a change of a position in a sequence will modify the positions of the teams.
The notebook StartEEE analyzes which different starting situation will bring to the same result.
