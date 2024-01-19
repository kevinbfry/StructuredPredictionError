import numpy as np
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def gen_model_barplots(
        model_errs, 
        model_names, 
        est_names, 
        title, 
        yaxis_title="Relative MSE",
        has_test_risk=True, 
        has_elev_err=False, 
        err_bars=False
    ):

    fig = make_subplots(
        rows=1, cols=len(model_names),
        subplot_titles=model_names)
    
    if has_elev_err and not has_test_risk:
        raise ValueError("Can't have has_elev_err=True and has_test_risk=False")

    for i, errs in enumerate(model_errs):
        risks = [err.mean() for err in errs]
        if has_test_risk:
            test_risk = risks[0]
            if has_elev_err:
                elev_test_risk = risks[1]
                offset = 2
            else:
                offset = 1
        else:
            test_risk = 1.
            offset = 0

        df = pd.DataFrame({
            est_names[i]: errs[i+offset] for i in np.arange(len(est_names))
        })

        if err_bars:
            fig.add_trace(go.Bar(
                x = df.columns,
                y=(df).mean()/test_risk,
                marker_color=px.colors.qualitative.Plotly,
                text=np.around((df).mean()/test_risk,3),
                textposition='outside',
                error_y=dict(
                    type='data',
                    color='black',
                    array=(df).std() / test_risk,
                )
            ), row=1, col=i+1)
        else:
            fig.add_trace(go.Bar(
                x = df.columns,
                y=(df).mean()/test_risk,
                marker_color=px.colors.qualitative.Plotly,
                text=np.around((df).mean()/test_risk,3),
                textposition='outside',
            ), row=1, col=i+1)

        fig.update_xaxes(title_text="Method", row=1, col=i+1)
        if has_test_risk:
            fig.add_hline(
                y=1., 
                line_color='red', 
                row=1, 
                col=i+1
            )
        if has_elev_err:
            fig.add_hline(
                y=elev_test_risk / test_risk, 
                line_color='grey', 
                line_dash='dash', 
                row=1, 
                col=i+1
            )
        
    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
    fig.update_layout(title_text=title, showlegend=False)
    return fig


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
