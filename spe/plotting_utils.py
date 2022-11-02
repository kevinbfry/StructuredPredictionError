import pandas as pd
import plotly.express as px


def gen_boxplot(est):
    mse_df = pd.DataFrame(
        {
            "kfcv_mse": est.kfcv_mse / est.true_mse,
            "spcv_mse": est.spcv_mse / est.true_mse,
            "gmcp_mse": est.gmcp_mse / est.true_mse,
            "frft_mse": est.frft_mse / est.true_mse,
            "nhnst_mse": est.nhnst_mse / est.true_mse,
            "hnst_mse": est.hnst_mse / est.true_mse,
        }
    )
    mse_df["idx"] = mse_df.index.values
    mse_df.set_index("idx")
    mse_df.reset_index()
    long_df = pd.melt(
        mse_df,
        id_vars="idx",
        value_vars=[
            "spcv_mse",
            "kfcv_mse",
            "gmcp_mse",
            "frft_mse",
            "nhnst_mse",
            "hnst_mse",
        ],
    )
    long_df.drop(columns="idx", inplace=True)

    fig = px.box(
        long_df,
        x="variable",
        y="value",
        color="variable",
        points="all",
        title=f"{est.n*est.reps}x{est.p}, {est.reps} repls, {est.niter} its, {est.block_corr} blk corr, {est.inter_corr} intr corr",
        labels={"variable": "Validation Method", "value": "Relative MSE"},
    )
    fig.update_traces(boxmean=True)
    fig.add_hline(y=1.0)
    return fig
