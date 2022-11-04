from sklearn.model_selection import train_test_split
from kasafranse.preprocessing import Preprocessing
import argparse
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split dataset into Training, validation and Testing sets')
    parser.add_argument(
        'filepath_1', metavar='filepath_1', help="Enter the file path of Twi document", type=str)
    parser.add_argument(
        'filepath_2', metavar='filepath_2', help="Enter the file path of English document", type=str)
    parser.add_argument(
        'filepath_3', metavar='filepath_3', help="Enter the file path of French document", type=str)
    parser.add_argument('--val_set', default=False, type=bool,
                        help='Pass True if validation set is required. Default = False')
    parser.add_argument('--output_dir', default= ".", type=str,
                        help='Pass output directory')
    
    args = parser.parse_args()

    # Read in the raw datasets
    # import preprocessing class
    # Create an instance of tft preprocessing class
    preprocessor = Preprocessing()

    # Read raw parallel dataset
    raw_data_twi, raw_data_en, raw_data_fr = preprocessor.read_parallel_dataset(
        filepath_1=args.filepath_1,
        filepath_2=args.filepath_2,
        filepath_3=args.filepath_3
    )

    # Normalize the raw data
    raw_data_en = [preprocessor.normalize_FrEn(data) for data in raw_data_en]
    raw_data_twi = [preprocessor.normalize_twi(data) for data in raw_data_twi]
    raw_data_fr = [preprocessor.normalize_FrEn(data) for data in raw_data_fr]

    # split the dataset into training and test sets
    # Keep 20% of the data as test set and validation
    if args.val_set:

        train_twi, test_twi, train_en, test_en, train_fr, test_fr = train_test_split(
            raw_data_twi, raw_data_en, raw_data_fr, test_size=0.2, random_state=42)

        # split test into testans validation
        test_twi, val_twi, test_en, val_en, test_fr, val_fr = train_test_split(
            test_twi, test_en, test_fr, test_size=0.5, random_state=42)

        # write the preprocess traning and test dataset to a file
        preprocessor.writeTotxt(f'{args.output_dir}/train_twi', train_twi)
        preprocessor.writeTotxt(f'{args.output_dir}/train_en', train_en)
        preprocessor.writeTotxt(f'{args.output_dir}/test_twi', test_twi)
        preprocessor.writeTotxt(f'{args.output_dir}/test_en', test_en)
        preprocessor.writeTotxt(f'{args.output_dir}/train_fr', train_fr)
        preprocessor.writeTotxt(f'{args.output_dir}/test_fr', test_fr)
        preprocessor.writeTotxt(f'{args.output_dir}/val_fr', val_fr)
        preprocessor.writeTotxt(f'{args.output_dir}/val_tw', val_twi)
        preprocessor.writeTotxt(f'{args.output_dir}/val_en', val_en)
    else:
        train_twi, test_twi, train_en, test_en, train_fr, test_fr = train_test_split(
            raw_data_twi, raw_data_en, raw_data_fr, test_size=0.1, random_state=42)

        # write the preprocess traning and test dataset to a file
        preprocessor.writeTotxt(f'{args.output_dir}/train_twi', train_twi)
        preprocessor.writeTotxt(f'{args.output_dir}/train_en', train_en)
        preprocessor.writeTotxt(f'{args.output_dir}/test_twi', test_twi)
        preprocessor.writeTotxt(f'{args.output_dir}/test_en', test_en)
        preprocessor.writeTotxt(f'{args.output_dir}/train_fr', train_fr)
        preprocessor.writeTotxt(f'{args.output_dir}/test_fr', test_fr)
