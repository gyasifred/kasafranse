from kasafranse.preprocessing import Preprocessing
import argparse
import warnings
warnings.simplefilter('ignore')


# Read in the raw datasets
# import preprocessing class
# Create an instance of tft preprocessing class
preprocessor = Preprocessing()

# Read raw parallel dataset
raw_data_twi, raw_data_en = preprocessor.read_parallel_dataset(
    filepath_1="data/twi_test.txt",
    filepath_2="data/english_test.txt"
)

# Normalize the raw data
raw_data_en = [preprocessor.normalize_FrEn(data) for data in raw_data_en]
raw_data_twi = [preprocessor.normalize_twi(data) for data in raw_data_twi]

# write the preprocess traning and test dataset to a file
preprocessor.writeTotxt('data/testing_en', raw_data_en)
preprocessor.writeTotxt('data/testing_tw', raw_data_twi)
