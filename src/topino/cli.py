import logging
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

    process_frames_parallel(input_path, out_path)


if __name__ == "__main__":
    app()
