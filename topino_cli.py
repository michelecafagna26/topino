"""
CLI tool for processing video frames to compute motion index and save results.

This script loads image frames, computes a motion index between consecutive frames
using image processing techniques, and saves the results to a parquet file.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from collections import OrderedDict
from itertools import batched, chain
import logging
import os
from functools import partial
from typing import Sequence
import typer
from rich.progress import Progress
from rich.console import Console
import ffmpeg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()
console = Console()


def process_frame_batch(
    frames: Sequence,
    box: tuple[int, int, int, int],
    min_th: int = 50,
    max_th: int = 255,
) -> Sequence[float]:
    """
    Process a batch of image frames to compute a motion index between consecutive frames.

    Args:
        frames: Sequence of image file paths to process.
        box: Bounding box (left, upper, right, lower) to crop each frame.
        min_th: Minimum threshold for binary thresholding. Defaults to 50.
        max_th: Maximum threshold for binary thresholding. Defaults to 255.

    Returns:
        Motion index values for each pair of consecutive frames.
    """

    if box is None:
        box = (
            0,
            0,
            frames[0].size[0],
            frames[0].size[1],
        )  # (left, upper, right, lower)

    mov_index = []
    frame_psx = frames[0]

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Processing frames...", total=len(frames) - 1)

        for next_frame_psx in list(frames)[1:]:
            progress.update(task, advance=1)

            frame = Image.open(frame_psx).crop(box)
            next_frame = Image.open(next_frame_psx).crop(box)

            frame_arr = cv2.GaussianBlur(np.asarray(frame), (5, 5), 0)
            next_frame_arr = cv2.GaussianBlur(np.asarray(next_frame), (5, 5), 0)

            frameDelta = cv2.absdiff(frame_arr, next_frame_arr)
            thresh = cv2.threshold(frameDelta, min_th, max_th, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            mov_index.append(
                thresh.astype(np.bool).sum() / (thresh.shape[0] * thresh.shape[1])
            )

            frame_psx = next_frame_psx

    return mov_index


@app.command()
def extract_frames(
    input: str = typer.Option(..., help="Path to the video file."),
    fps: int = typer.Option(1, help="Frames per second to extract."),
    out_path: str = typer.Option(..., help="Output directory for extracted frames."),
):
    """
    Extract frames from a video file and save them as images.

    Uses ffmpeg to extract frames from the specified input video at a given frame rate (fps).
    The extracted frames are saved in the output directory with filenames formatted as 'frame_XXXX.png'.

    Args:
        input: Path to the input video file.
        fps: Number of frames per second to extract from the video.
        out_path: Output directory where the extracted frames will be saved.

    Example:
        extract_frames(input="video.mp4", fps=1, out_path="frames/")
        # This will save frames as frames/frame_0001.png, frames/frame_0002.png, ...
    """

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True, parents=True)

    (
        ffmpeg.input(str(input))
        .filter("fps", fps=fps)
        .output(str(out_path / "frame_%04d.png"))
        .run()
    )


@app.command()
def process(
    input_path: str = typer.Option(
        ..., help="Path to the directory containing frames."
    ),
    out_path: str = typer.Option(
        "motion_index.parquet", help="Path to the output file."
    ),
):
    typer.echo(f"Processing frames in {input_path}")
    """
    Main function to process frames, compute motion index, and save results.

    Loads image frames from a directory, splits them into batches, processes them in parallel,
    computes the motion index, and saves the results to a parquet file.
    """
    frames = [psx for psx in Path(input_path).iterdir()]
    frame_index = OrderedDict(
        sorted({int(Path(psx).stem.split("_")[1]): psx for psx in frames}.items())
    )

    box = (90, 90, 490, 360)  # (left, upper, right, lower)

    num_cores = os.cpu_count()
    logger.info(f"Number of CPU cores: {num_cores}")

    batch_size = len(frame_index) // num_cores
    batches = list(batched(list(frame_index.values()), n=batch_size))
    process_frame_batch_partial = partial(process_frame_batch, box=box)

    mov_index = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_frame_batch_partial, batches))
        mov_index = list(chain.from_iterable(results))

    time_start = timedelta(minutes=0)
    time_index = [
        datetime(2000, 1, 1) + time_start + timedelta(seconds=int(x))
        for x in range(len(mov_index))
    ]

    df = pd.DataFrame({"time": time_index, "value": mov_index})

    path = Path(out_path)
    if path.suffix in [".parquet", ".pq"]:
        df.to_parquet(out_path)
    elif path.suffix in [".csv"]:
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("Output file must have .parquet, .pq, or .csv extension.")

    console.print(f"Motion index saved to [green]{out_path}[/green] 🪵")


if __name__ == "__main__":
    app()
