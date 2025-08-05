from spe.estimators import new_y_est, kfoldcv, kmeanscv

## Estimator parameters
def get_est_array(est_str_list, incl_cv=True):
    # if isinstance(est_str_list, str):
    #     est_str_list = [est_str_list]
    est_array = [
        new_y_est.__name__,
        new_y_est.__name__,
    ] 
    est_array.extend(est_str_list)
    if incl_cv:
        est_array.extend([
            kfoldcv.__name__, 
            kmeanscv.__name__
        ]) 

    return est_array

## Markdown parameters
def get_model_md_str(spe_est_str):
    return spe_est_str.replace("cp_","").replace("_", " ").title()

def get_est_md_str(spe_est_str):
    return f"spe.estimators.{spe_est_str}"