import papermill as pm

from docs.sim_parameters.sims_base_config import *

pm.execute_notebook(
    './sphinx/notebooks/sim_template.ipynb',
    './sphinx/notebooks/sim_template_test_output.ipynb',
    kernel_name='spe',
    parameters=dict(
        niter=NITER,
        gsize=GSIZE,
        sqrt_n=SQRT_N,
        p=P,
        s=S,
        delta=DELTA,
        snr=SNR,
        tr_frac=TR_FRAC,
        use_spatial_split=USE_SPATIAL_SPLIT,
        noise_kernel=NOISE_KERNEL,
        noise_length_scale=NOISE_LENGTH_SCALE,
        noise_nu=NOISE_NU,
        X_kernel=X_KERNEL,
        X_length_scale=X_LENGTH_SCALE,
        X_nu=X_NU,
        Chol_ystar=CHOL_YSTAR,
        Cov_y_ystar=COV_Y_YSTAR,
        k=K,
        model_kwargs=MODEL_KWARGS,
        model_names=MODEL_NAMES,
        est_name=EST_NAMES,
        est_strs=EST_STRS,
        est_kwargs=EST_KWARGS,
        est_names=EST_NAMES,
    )
)



# ## GEMINI outline for next version. Needs edits for taking args, etc.
# import papermill as pm
# import os
# import importlib.util
# import sys

# # --- Configuration ---
# INPUT_NOTEBOOK = "sim_template.ipynb"
# OUTPUT_DIR = "notebooks"
# PARAMETERS_DIR = "sim_parameters"
# # Name of the kernel to use (e.g., from 'jupyter kernelspec list')
# KERNEL_NAME = "spe" # IMPORTANT: Change this to your desired kernel name

# # Ensure the output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Add the parent directory of 'parameters' to sys.path
# # This allows dynamic import of 'parameters' as a package
# # Assuming run_experiments.py is in the project root
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# def load_parameters_from_file(file_path: str) -> dict:
#     """
#     Loads parameters from a given Python file.
#     Only variables with ALL_CAPS names are considered parameters.
#     """
#     module_name = os.path.basename(file_path).replace(".py", "")
#     spec = importlib.util.spec_from_file_location(module_name, file_path)
#     if spec is None:
#         raise ImportError(f"Could not load spec for module {module_name} from {file_path}")
    
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = module # Add to sys.modules
#     spec.loader.exec_module(module)

#     parameters = {}
#     for key, value in module.__dict__.items():
#         # Convention: only ALL_CAPS variables are parameters
#         if key.isupper() and not key.startswith('__'):
#             # Convert to lowercase if your notebook expects lowercase parameters
#             # Or keep as is if your notebook matches ALL_CAPS
#             parameters[key.lower()] = value
#     return parameters


# def main():
#     if args.param_files is None:
#         param_files = [f for f in os.listdir(PARAMETERS_DIR) if f.endswith(".py") and f != "__init__.py"]
#     else:
#         param_files = args.param_files

#     if not param_files:
#         print(f"No parameter files found in '{PARAMETERS_DIR}'.")
#         sys.exit(1)

#     print(f"Found {len(param_files)} parameter files: {', '.join(param_files)}")

#     for param_file_name in sorted(param_files): # Sort for consistent order
#         file_path = os.path.join(PARAMETERS_DIR, param_file_name)
#         experiment_name = param_file_name.replace(".py", "")

#         print(f"\n--- Running simulation: {experiment_name} ---")

#         try:
#             # Load parameters from the current file
#             params = load_parameters_from_file(file_path)
#             print(f"Parameters loaded for {experiment_name}: {params}")

#             # Define output notebook path
#             output_notebook_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_output.ipynb")

#             # Execute the notebook
#             pm.execute_notebook(
#                 INPUT_NOTEBOOK,
#                 output_notebook_path,
#                 parameters=params,
#                 kernel_name=KERNEL_NAME
#             )
#             print(f"Successfully executed and saved to: {output_notebook_path}")

#         except Exception as e:
#             print(f"Error running experiment {experiment_name}: {e}")
#             # Optionally, you can log the error or handle it differently
#             # For example, save an error notebook
#             # pm.execute_notebook(
#             #     INPUT_NOTEBOOK,
#             #     os.path.join(OUTPUT_DIR, f"{experiment_name}_ERROR.ipynb"),
#             #     parameters=params,
#             #     kernel_name=KERNEL_NAME
#             # )
#             # continue # Continue to next experiment even if one fails

# if __name__ == "__main__":
#     main()