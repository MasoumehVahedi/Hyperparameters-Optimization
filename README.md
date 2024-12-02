# Hyperparameter Optimization for Clustering

## Introduntion

We introduce an approach to optimize the hyperparameters of the BIRCH for efficient search. Our goal is to minimize query execution time with an index over clusters, rather than the data objects themselves, which depends critically on the properties of the clustering.

## Dependencies
- Python3
- Numpy
- scikit-learn
- Tensorflow
- Keras

## Code Structure
To run the code, you only need to execute the provided bash script `run_scripts.sh`, which handles everything from data preparation to executing queries and saving the results. Follow these simple steps to get started:

### Step 1: Install Necessary Packages
Make sure you have all required dependencies installed. You can install them using the following command:
```bash
pip install -r requirements.txt
```

### Step 2: Clone the Repository
Clone the SPLindex repository to your local machine:
```bash
git clone https://github.com/MasoumehVahedi/Hyperparameters-Optimization
```

### Step 3: Download and Prepare the Datasets
Download the necessary datasets and place them in the directory specified in the `ConfigParam.py` file.

For example:
```python
polygons_path = "path/to/data"
```

### Step 4: Run the Bash Script
Navigate to the directory containing the `run_scripts.sh` script and execute it to run the entire process:
```bash
./run_scripts.sh
```

This script will automatically:
1. **Prepare the data**: Generate the required CSV files and datasets.
2. **Execute queries**: Run all the necessary queries on the datasets.
3. **Save the results**: Store the output for further analysis.

## Example Commands Summary
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Clone the repository:
  ```bash
  git clone https://github.com/MasoumehVahedi/Hyperparameters-Optimization
  ```
- Execute the `run_scripts.sh`:
  ```bash
  ./run_scripts.sh
  ```


