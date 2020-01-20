File : data_extraction.py

This file uses a mongoDB query to receive data from the api. The data is stored as event_10_2018.json.

File: BigData_proj.ipynb

Input file for this notebook:

event_10_2018.json

This notebook loads the input json file and converts it to a JSONL format and outputs it as out.json. This fileis later read in the same notebook for further processing and analysis.

The notebook contains step by step instructions as to what is done. Run each cell to get the output.

General overview of what is done in the notebook:

-Reading JSON input and converting it to JSONL and further reading.
-Preprocessing and removal of unwanted columns and rows with NA values.
-Encoding string inputs and ine hot encoding for categorical data
-Combining features
-Training and prediction using:
	-Decision trees
	-Random Forest
	-Logistic Regression and 
	-Support Vector Machines

File : project-notebook1-LSTM.ipynb

-Reading JSON input
-Initial analysis of data
-Add time stamp to date column for LSTM
-Use LSTM model on data for prediction of rootcode
