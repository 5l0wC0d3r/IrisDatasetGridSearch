# IrisDatasetGridSearch


### Prerequisites

Make sure you have the following libraries installed:

- pandas
- numpy
- scikit-learn
- xgboost

You can install them using pip:
``` !pip install pandas numpy scikit-learn xgboost ```

### Usage

1. Prepare your dataset: Place the CSV file containing the dataset in the same directory as the Python script.

2. Specify the file name in the `dataset_path` variable at the beginning of the script.

3. The hyperparameters and algorithm selection are controlled using a JSON configuration file. Specify the json file in the `json_file_path` variable at the beginning of the script

4. Run the Python script: Execute the Python script to perform the following steps:

   - Load the dataset from the CSV file.
   - Handle missing values and convert string features to floats using hashing.
   - Generate new features like linear interactions, polynomial interactions, and explicit pairwise interactions (if specified in the JSON).
   - Reduce features using one of the methods: No Reduction, Correlation with Target, Tree-based, or PCA (based on the JSON).
   - Perform model selection and hyperparameter tuning using GridSearchCV for the selected algorithms and hyperparameters specified in the JSON.
   - Split the data into training and testing sets.
   - Fit the models on the training data and evaluate them on the test data.
   - Print the best algorithm, its parameters, and its performance score.

### JSON Configuration

The JSON should follow the structure provided in the example `algoparams_from_ui.json.rtf`.

Ensure that the JSON file is in the correct format and contains valid hyperparameters and algorithm names.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to modify the code and adapt it to your specific needs.

### Authors

- [Chetanya Bhan](https://github.com/5l0wC0d3r)


