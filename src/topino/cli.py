import logging
from pathlib import Path
import typer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def extract_frames(
    input_file: str = typer.Option(..., help="Path to the video file."),
    out_path: str = typer.Option(..., help="Output directory for extracted frames."),
    fps: int = typer.Option(1, help="Frames per second to extract."),
) -> None:
    """
    Command to extract frames from a video file.
    """
    from topino.video import extract_frames

    typer.echo(
        f"Extracting frames  at {fps} fps from {input_file} saving to {out_path}"
    )

    extract_frames(input_file, out_path, fps)


@app.command()
def process(
    input_path: str = typer.Option(
        ..., help="Path to the directory containing frames."
    ),
    out_path: str = typer.Option(
        "motion_index.parquet", help="Path to the output file."
    ),
) -> None:
    typer.echo(f"Processing frames in {input_path}")
    """
    Command to process frames and compute motion index.
    """
    from topino.frame_processor import process_frames_parallel

    # FIXME: fixed for now
    box = (90, 90, 490, 360)  # (left, upper, right, lower)

    df = process_frames_parallel(input_path, box=box)

    out_path = Path(out_path)
    if out_path.suffix in [".parquet", ".pq"]:
        df.to_parquet(out_path)
    elif out_path.suffix in [".csv"]:
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("Output file must have .parquet, .pq, or .csv extension.")

    typer.echo(f"Motion index saved to [green]{out_path}[/green] 🪵")


if __name__ == "__main__":
    app()
