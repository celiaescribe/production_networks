
# Sufficiency production network

This code was developed to run a two-region open economy production network model, and to estimate the mitigation potential of sufficiency consumption changes. The calibration relies on the GLORIA input-output database. 

## Installation

### Step 1: Git clone the folder in your computer.

```bash
git clone https://github.com/celiaescribe/production_networks.git
```

### Step 2: Create a conda environment from the environment.yml file:
- The environment.yml file is in the production_networks folder. The only packages we rely on to run the code are `numpy, pandas, matplotlib, seaborn`.
- Use the terminal and go to the production_networks folder stored on your computer.
- Type the following command:
```bash
conda env create -f environment.yml
```

### Step 3: Activate the conda environment:

```bash
conda activate networks
```
## Getting started

### Step 1: Calibrating the model 

This calibration can be done in three steps. First, preprocessing is done by running the script `gloria_preprocessing.py`. This requires specifying the two-region aggregation chosen, and the year for the input-output tables. Currently, available implementation includes a two-region economy based on a single country (e.g., the United States) and a Rest of the World region, or the European Union and a Rest of the World region. GLORIA database can be accessed at the following url: https://ielab.info/resources/gloria/supportingdocs
```bash
python gloria_preprocessing.py --country eu --year 2018
```

Some postprocessing is then required, by running the script 
```bash
python gloria_postprocessing.py --country eu --year 2018
```

Finally, calibration can be done by running the script 
```bash
python calibrate.py --country eu --year 2018
```
This saves a configuration file in the folder `production_networks/outputs`.

### Step 2: Running the model
The standard way to run the model is to launch the script main.py. This requires providing the configuration which you want to run. The configuration file specifies different options, such as the calibration file to use, the reference elasticity parameters, and the specification of the shocks, among others. Examples of configuration can be found in `production_networks/inputs/configs`.

The script is then run as follows:

```bash
python main.py --config config_eu.json
```

### Step 3: Explore outputs
Output files are stored in `production_networks/outputs`. 
