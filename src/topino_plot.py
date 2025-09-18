import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    import plotly.express as px
    from scipy.ndimage.filters import gaussian_filter1d

    return gaussian_filter1d, mo, pd, px


@app.cell
def _(pd):
    mov_index_df = pd.read_parquet("mouse_51_motion_index.parquet")
    return (mov_index_df,)


@app.cell
def _(mo):
    smoothing_factor = mo.ui.slider(
        start=1, stop=100, label="Smoothing Factor", value=3
    )
    return (smoothing_factor,)


@app.cell
def _(gaussian_filter1d, mo, mov_index_df, pd, px, smoothing_factor):
    mov_index = mov_index_df["value"]
    smoothed = gaussian_filter1d(mov_index, sigma=smoothing_factor.value)

    df = pd.DataFrame({"time": mov_index_df["time"], "value": smoothed})
    plot = mo.ui.plotly(px.line(df, x="time", y="value", title="Smoothed Motion Index"))

    mo.vstack([smoothing_factor, plot])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
