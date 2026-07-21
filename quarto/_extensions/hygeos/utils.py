import io
import base64
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from IPython.display import display, Markdown


def display_markdown(markdown: str, append_newline=True):
    """Display markdown text in the output.

    Args:
        markdown: The markdown string to display.
    """
    display(Markdown(markdown + ("\n" if append_newline else "")))


def display_figure(
    fig: Figure | None = None,
    legend: str | None = None,
    label: str | None = None,
    width: str = "80%",
):
    """Save and display a matplotlib figure as an embedded base64 image.

    Encodes the figure as a base64 PNG and displays it as markdown with optional
    caption and label for Quarto documents.

    Args:
        fig: The matplotlib Figure to save. If None, uses the current figure.
        legend: Optional caption/alt text for the image.
        label: Optional label for cross-referencing in Quarto (e.g., '#fig-myplot').
    """
    # TODO: check if we can.should use svg ?
    # Use current figure if none provided
    if fig is None:
        fig = plt.gcf()

    # Encode figure as base64 and embed in a Quarto figure div with caption
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    if legend is None:
        legend = ''

    md = f"\n\n![{legend}]"
    md += f"(data:image/png;base64,{img_b64})"
    
    render = "{"
    if label is not None:
        assert label.startswith('#')
        render += f"{label} "
        
    assert width.endswith('%')
    render += f"width={width}"+ "}\n\n"
    md += render
    
    display(Markdown(md))
