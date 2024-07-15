# Parkinson Disease Prediction 

![brain](brain.avif)

This study deals with exploring various ML-methods to detect parkinson disease for early treatment and cure.

## Requirements
1. Python
2. Numpy
3. Pandas
4. ScikitLearn

To run this project, install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Dataset 
Download the enitre dataset here - [Kaggle dataset for parkinson](https://github.com/aryanc381/Parkinson-Disease-Prediction-using-ML-Models/blob/main/Parkinsson%20disease.csv)

## Project Workflow
1. Data extraction using pandas
2. Data Preprocessing
   - Seperating the target features
   - Splitting dataset into train and test
   - Data Stadardisation
   - Scalar transformation
3. Model Building
   - Creation
   - Training
   - Evaluation
4. Building the prediction system
   - Passing random input data from test dataset
   - Converting into numpy array
   - Reshaping and standardisation
   - Prediction
   - Logical conditioning

## Support Vector Machine
- Type : ```Classifier```
- Kernel : ```Linear```
- Input : ```Numpy Array```
- Output : ```0 (-ve)``` and ```1 (+ve)```
- Accuracy on training data : ```88.46%```
- Accuracy on test data : ```87.17%```

## Usage
Clone the repository:
```bash
git clone https://github.com/aryanc381/Parkinson-Disease-Prediction-using-ML-Models.git
```
Navigate to the project directory:
```bash
cd Parkinson-Disease-Prediction-using-ML-Models
```
Open the Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```

## Reach out
For any queries, doubts or improvements, reach me out at ```venomc381@gmail.com```
