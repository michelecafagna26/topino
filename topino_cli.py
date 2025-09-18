"""
CLI tool for processing video frames to compute motion index and save results.

This script loads image frames, computes a motion index between consecutive frames
using image processing techniques, and saves the results to a parquet file.
"""

from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager
from queue import Queue
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime, timedelta

from collections import OrderedDict
from itertools import batched
import logging
import os
from typing import Sequence
import typer
from rich.progress import Progress
from rich.console import Console
import ffmpeg

from pydantic import BaseModel

from rich.live import Live
from rich.panel import Panel
from rich.progress import SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = typer.Typer()
console = Console()


class FrameBatch(BaseModel):
    id: int
    frames: Sequence[Path]


def process_frame_batch(
    batch: FrameBatch,
    box: tuple[int, int, int, int],
    min_th: int = 50,
    max_th: int = 255,
    progress_queue: Queue | None = None,
) -> Sequence[float]:
    """
    Process a batch of image frames to compute a motion index between consecutive frames.

    Args:
        frames: Sequence of image file paths to process.
        box: Bounding box (left, upper, right, lower) to crop each frame.
        min_th: Minimum threshold for binary thresholding. Defaults to 50.
        max_th: Maximum threshold for binary thresholding. Defaults to 255.
        progress_queue: Queue to report progress back to main process.

    Returns:
        Motion index values for each pair of consecutive frames.
    """

    if box is None:
        box = (
            0,
            0,
            batch.frames[0].size[0],
            batch.frames[0].size[1],
        )  # (left, upper, right, lower)

    mov_index = []
    frame_psx = batch.frames[0]

    for i, next_frame_psx in enumerate(list(batch.frames)[1:]):
        if progress_queue is not None:
            progress_queue.put((batch.id, i + 1))  # Report progress

        frame = Image.open(frame_psx).crop(box)
        next_frame = Image.open(next_frame_psx).crop(box)

        frame_arr = cv2.GaussianBlur(np.asarray(frame), (5, 5), 0)
        next_frame_arr = cv2.GaussianBlur(np.asarray(next_frame), (5, 5), 0)

        frameDelta = cv2.absdiff(frame_arr, next_frame_arr)
        thresh = cv2.threshold(frameDelta, min_th, max_th, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=2)
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
) -> None:
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

    out_path_psx = Path(out_path)
    out_path_psx.mkdir(exist_ok=True, parents=True)

    (
        ffmpeg.input(str(input))
        .filter("fps", fps=fps)
        .output(out_path_psx / "frame_%04d.png")
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
) -> None:
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

    cpu_count: int | None = os.cpu_count()
    num_cores: int = cpu_count if cpu_count is not None else 1
    LOGGER.info(f"Number of CPU cores: {num_cores}")

    batch_size = len(frame_index) // num_cores
    batches = list(batched(list(frame_index.values()), n=batch_size))
    mov_index: Sequence[float] = []

    # Setup progress tracking with Manager for cross-process communication
    manager = Manager()
    progress_queue = manager.Queue()

    # Create progress bars
    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )

    # Create progress tasks for each batch
    job_tasks = {}
    for i, batch in enumerate(batches):
        job_tasks[i] = job_progress.add_task(
            f"[cyan]Batch {i + 1}/{len(batches)}",
            total=len(batch) - 1,  # -1 because we process pairs of frames
        )

    progress_table = Table.grid()
    progress_table.add_row(
        Panel.fit(
            job_progress,
            title="[b]Processing Frames",
            border_style="green",
            padding=(1, 2),
        )
    )

    mov_index, futures = [], []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Create a future for each batch and store mapping
        for idx, batch in enumerate(batches):
            futures.append(
                executor.submit(
                    process_frame_batch,
                    FrameBatch(id=idx, frames=batch),
                    box=box,
                    progress_queue=progress_queue,
                )
            )

        # Start progress display
        with Live(progress_table, refresh_per_second=10):
            completed_futures: set[Future[Sequence[float]]] = set()

            while len(completed_futures) < len(batches):
                # Check for progress updates
                while not progress_queue.empty():
                    batch_id, frame_progress = progress_queue.get()
                    job_progress.update(job_tasks[batch_id], completed=frame_progress)

                # Check for completed futures
                for future in [
                    f for f in futures if f.done() and f not in completed_futures
                ]:
                    mov_index.extend(future.result())
                    completed_futures.add(future)

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
