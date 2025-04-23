# Collisional Thermometry

This project computes the Quantum Fisher Information for a thermometer made by k-chains of ancillas.

## Installation
To use the code in this repository, ensure you have Python 3.8 or later installed on your system. 
You can download Python from [python.org](https://www.python.org/).
Than, the `requirements.txt` file will tell you what packages do you need.
The most common way to install packages is via **pip** or **conda**.
### Install Dependencies
If you want to use pip or conda to manage required packages, it is always best to create a virtual environment first.
#### Using `pip`

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```
   Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Using `conda`
1. Create a new conda environment:
   ```bash
   conda create -n collisional-thermometer python=3.8
   ```
   Activate the environment:
   ```bash
   conda activate collisional-thermometer
   ```

2. Install the required dependencies:
   ```bash
   conda install --file requirements.txt
   ```

## Configuration

The simulation parameters are defined in the config.toml file. 
Update the file to set your desired parameters.
Parameters specified as lists will be iterated over, creating different folders and configurations for each combination of parameters.
Below is an example configuration:

```toml
[meta]
folder = "results/variable_phaseonium"
description = " "

[system]
type = "Qubit"
ndims = 2
excitations = 1
frequency = 1.0
thermalization_time = 100.0

[environment]
g = 1.0
J = 1.0
gamma = 1.0

[ancilla]
type = "Qubit"
ndims = 2
excitations = 1
chains = [1, 2, 4]
layers = [1, 2, 10]
frequency = 1.0
g = 1.0
intra_interaction_time = 0.785398163
collision_time = 0.031415926

[thermometer]
T_min = 0.002
T_max = 2.0
accuracy = 100
```
In this example, the code will run once for every possible combination of parameters `layers` (with values 1, 2 or 4) and `chains` (with values 1, 2 or 10).

## Running the Simulation

1. Ensure the config.toml file is properly configured.
2. Run the main script:
   ```bash
   python main.py
   ```

### Command-Line Arguments
The script supports the following optional arguments:
- `--chains` or `-k`: Override the number of interacting ancilla chains.
- `--layers` or `-n`: Override the number of ancillas in each chain.
- `--plot` or `-p`: Show the comparison of QFI from the last ancilla of every chain.
- `--resume-file` or `-r`: Resume analysis for configurations without results given a common root directory.

Example:
```bash
python main.py --chains 2 --layers 5 --plot
```

## Results

The results are saved in the directory specified in the config.toml file under the `[meta]` section. If multiple parameter combinations are defined, subdirectories are created for each configuration, identified by unique IDs.

### Output Files
- `system_evolution`: Evolution of the system states.
- `ancillas_evolution`: Evolution of the ancilla states.
- `qfi_systems.pickle`: Quantum Fisher Information for the system.
- `qfi_ancillas.pickle`: Quantum Fisher Information for the ancillas.
- `last_ancilla_qfi.pickle`: QFI of the last ancilla.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

1. Mendonça, T. M., Soares-Pinto, D. O., & Paternostro, M. (2024). Information flow-enhanced precision in collisional quantum thermometry. *arXiv preprint arXiv:2407.21618*.
2. Seah, S., Nimmrichter, S., Grimmer, D., Santos, J. P., Scarani, V., & Landi, G. T. (2019). Collisional quantum thermometry. *Physical Review Letters, 123(18), 180602*.