import papermill as pm
import os
import importlib.util
import sys
import argparse

from docs.sim_parameters.sims_base_config import *

# --- Configuration ---
OUTPUT_DIR = "./sphinx/notebooks"
INPUT_NOTEBOOK = "./papermill_templates/sim_template.ipynb"
PARAMETERS_DIR = "sim_parameters"
KERNEL_NAME = "spe" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add the parent directory of 'parameters' to sys.path
# This allows dynamic import of 'parameters' as a package
# Assuming run_experiments.py is in the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_parameters_from_file(file_path: str) -> dict:
    """
    Loads parameters from a given Python file.
    Only variables with ALL_CAPS names are considered parameters.
    """
    module_name = os.path.basename(file_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module # Add to sys.modules
    spec.loader.exec_module(module)

    parameters = {}
    for key, value in module.__dict__.items():
        if key.isupper() and not key.startswith('__'):
            parameters[key.lower()] = value
    return parameters


def main():
    parser = argparse.ArgumentParser(description="A script to process data files.")
    parser.add_argument(
        "--param_files", 
        help="Parameter files to run simulations on.", 
        nargs="*", 
        type=str
    )

    args = parser.parse_args()

    if args.param_files is not None and len(args.param_files) > 0:
        param_files = [f for f in args.param_files if os.path.isfile(os.path.join(PARAMETERS_DIR, f))]
    else:
        param_files = [f for f in os.listdir(PARAMETERS_DIR) if f.endswith(".py") and f != "__init__.py"]

    if not param_files:
        print(f"No parameter files found in '{PARAMETERS_DIR}'.")
        sys.exit(1)

    print(f"Found {len(param_files)} parameter files: {', '.join(param_files)}")

    for param_file_name in sorted(param_files): # Sort for consistent order
        file_path = os.path.join(PARAMETERS_DIR, param_file_name)
        experiment_name = param_file_name.replace("_config.py", "")

        print(f"\n--- Running simulation: {experiment_name} ---")

        # try:
        # Load parameters from the current file
        params = load_parameters_from_file(file_path)
        print(f"Parameters loaded for {experiment_name}: {params}")

        # Define output notebook path
        output_notebook_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_test.ipynb")

        # Execute the notebook
        pm.execute_notebook(
            INPUT_NOTEBOOK,
            output_notebook_path,
            parameters=params,
            kernel_name=KERNEL_NAME
        )
        print(f"Successfully executed and saved to: {output_notebook_path}")

        # except Exception as e:
        #     print(f"Error running experiment {experiment_name}: {e}")

if __name__ == "__main__":
    main()