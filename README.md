# Portuguese-Art-Auction-Prices

A project to collect art sales in portuguese auction houses to explore and build a predictive model.

This is an unfishined ongoing project. Main objectives are:

1 - Collect data from the main modern and contemporary art auction houses in Portugal (online on 1st stage, online and offline later); 2 - Clean data and create a unified dataset (or several datasets with same structure, ready for data exploration, further feature engineering); 3 - Create a ML model capable of reasonably guess the final sale price of an artwork.

All 3 point were first executed for the Cabral Moncada Leitão auction house. Model prediction on XGBoost was able to beat a simple baseline as well a the auction house's estimated sale price when considering RMSE but on the second case, only by a slim majority.

The art world is fickle and somewhat speculative which may difficult this prediction job.

More data should be able to increase the model's generalization ability as it seems to be tending to overfit (not surprising considering that CML only has < 2K rows of sold artwork).

Finally, a Plotly Dask dashboard should also be created.

Further exploration is necessary to try and extract more features from the artwork image itself as it may be an excelent source of information. Another option is to pass the image to a Neural Network as well as the "metadata" about the artwork itself (data currently stored in a csv that has been used to train the model) and see if it produces a better result.

Clustering should also be explored to try and identify interesting patterns in the data.

File structure is as follows:

cml_eda_predictions.ipynb - Contains the graphical exploration, further feature engineering and the model preparation and evaluation itself.
cml_pre_process_data.py - A script that receives the scrapped data for CML, cleans and processes it.
cml_functions.py - Contains functions used on both files above.
cml_data_cleaning_with_explanation.ipynn - Has the notebook version of ml_pre_process_data.py with a step by step exploration of the process and data itself.
cml/ - Folder that contains the scrapping notebook (still not converted to a script as of now) as well as the data files in .csv format used on that process.

Lemmatization was performed using Stanford's Stanza.
