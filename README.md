# Implementation of simple Multiple Linear Regression Model <br/><br/>
## Dataset: Fuel Consumption (2014) Canada<br/>
### Model: Predicting Carbon Dioxide emissions from vehicle using its Engine Size, No of Cylinders, Fuel Consumption in City, Fuel Consumption on Highway, Fuel Consumption in Combination<br/>
### Libraries used:<br/>
**Numpy** - For numpy arrays<br/>
**Matplotlib** - For plotting<br/>
**Pandas** - To read data<br/>
**Scikit-Learn** - For calculating R2_score<br/>
*scikit-learn is not used for predicting modelling, parameters have been learnt by writing code from scratch*<br/>

### Description
- Train and Test set has been splited in ratio 80:20. Training has been done on train set.
- Model selection has been using R2_score by trying various Learning Rates.
- Number of Iterations has been kept equal to 1000.
- Test set has been used to calculate cost/loss on unknown set.


