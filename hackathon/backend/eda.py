# eda.py - Final Simple EDA Dashboard

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image


# ---------- Utility ----------
def fig_to_img(fig):
    """Convert matplotlib figure to PIL image for Gradio."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---------- Overview ----------
def dataset_overview(df):
    return f"""
### 📊 Dataset Overview
- Rows: *{df.shape[0]}*
- Columns: *{df.shape[1]}*
- Missing Values: *{df.isna().sum().sum()}*
- Duplicates: *{df.duplicated().sum()}*
"""


# ---------- Numeric ----------
def numeric_summary(df, col):
    if not col:
        return "Select a numeric column", None, None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return "No numeric data", None, None

    desc = s.describe().to_frame().T
    text = f"### 🔢 {col}\n\n\n{desc}\n"

    fig, ax = plt.subplots()
    sns.histplot(s, kde=True, ax=ax, color="skyblue")
    ax.set_title(f"Histogram of {col}")
    hist_img = fig_to_img(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x=s, ax=ax, color="lightgreen")
    ax.set_title(f"Boxplot of {col}")
    box_img = fig_to_img(fig)

    return text, hist_img, box_img


# ---------- Categorical ----------
def categorical_summary(df, col):
    if not col:
        return "Select a categorical column", None, None
    vc = df[col].astype(str).value_counts().head(10)

    text = f"### 🏷 {col}\n\nTop categories:\n\n" + str(vc)

    fig, ax = plt.subplots()
    sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="viridis")
    ax.set_title(f"Top Categories in {col}")
    bar_img = fig_to_img(fig)

    fig, ax = plt.subplots()
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%")
    ax.set_title(f"Pie Chart of {col}")
    pie_img = fig_to_img(fig)

    return text, bar_img, pie_img


# ---------- Correlation ----------
def correlation_heatmap(df):
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig_to_img(fig)


# ---------- Custom Scatter ----------
def scatter_plot(df, x, y):
    if not x or not y:
        return None
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x], y=df[y], ax=ax, color="orange")
    ax.set_title(f"{x} vs {y}")
    return fig_to_img(fig)


# ---------- Gradio App ----------
with gr.Blocks() as demo:
    gr.Markdown("## 📊 Simple EDA Dashboard")

    file = gr.File(file_types=[".csv"], label="Upload CSV")
    state_df = gr.State()

    with gr.Tab("Overview"):
        overview = gr.Markdown()

    with gr.Tab("Numeric"):
        num_col = gr.Dropdown([], label="Numeric Column")
        num_text = gr.Markdown()
        num_hist = gr.Image(type="pil")
        num_box = gr.Image(type="pil")

    with gr.Tab("Categorical"):
        cat_col = gr.Dropdown([], label="Categorical Column")
        cat_text = gr.Markdown()
        cat_bar = gr.Image(type="pil")
        cat_pie = gr.Image(type="pil")

    with gr.Tab("Correlation"):
        heatmap = gr.Image(type="pil")

    with gr.Tab("Custom Plot"):
        x_col = gr.Dropdown([], label="X Column")
        y_col = gr.Dropdown([], label="Y Column")
        scatter = gr.Image(type="pil")

    # ---------- Load File ----------
    def load_file(file):
        df = pd.read_csv(file.name)

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        all_cols = df.columns.tolist()

        default_num = num_cols[0] if num_cols else None
        default_cat = cat_cols[0] if cat_cols else None
        default_x = all_cols[0] if all_cols else None
        default_y = all_cols[1] if len(all_cols) > 1 else None

        # Pre-generate summaries
        num_summary_out = numeric_summary(df, default_num) if default_num else ("", None, None)
        cat_summary_out = categorical_summary(df, default_cat) if default_cat else ("", None, None)
        scatter_out = scatter_plot(df, default_x, default_y) if default_x and default_y else None

        return (
            df,
            dataset_overview(df),
            gr.update(choices=num_cols, value=default_num),
            gr.update(choices=cat_cols, value=default_cat),
            gr.update(choices=all_cols, value=default_x),
            gr.update(choices=all_cols, value=default_y),
            correlation_heatmap(df),
            *num_summary_out,
            *cat_summary_out,
            scatter_out,
        )

    file.upload(
        load_file,
        inputs=file,
        outputs=[
            state_df,
            overview,
            num_col,
            cat_col,
            x_col,
            y_col,
            heatmap,
            num_text,
            num_hist,
            num_box,
            cat_text,
            cat_bar,
            cat_pie,
            scatter,
        ],
    )

    # Callbacks
    num_col.change(numeric_summary, inputs=[state_df, num_col], outputs=[num_text, num_hist, num_box])
    cat_col.change(categorical_summary, inputs=[state_df, cat_col], outputs=[cat_text, cat_bar, cat_pie])
    x_col.change(scatter_plot, inputs=[state_df, x_col, y_col], outputs=scatter)
    y_col.change(scatter_plot, inputs=[state_df, x_col, y_col], outputs=scatter)


if _name_ == "_main_":
    demo.launch()
