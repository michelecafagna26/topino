from pathlib import Path
import ffmpeg


def extract_frames(input_file: str, out_path: str, fps: int) -> None:
    """
    Extract frames from a video file and save them as images.

    Uses ffmpeg to extract frames from the specified input video at a given frame rate (fps).
    The extracted frames are saved in the output directory with filenames formatted as 'frame_XXXX.png'.

    Args:
        input_file: Path to the input video file.
        out_path: Output directory where the extracted frames will be saved.
        fps: Number of frames per second to extract from the video.


    Example:
        extract_frames(input="video.mp4", fps=1, out_path="frames/")
        # This will save frames as frames/frame_0001.png, frames/frame_0002.png, ...
    """

    out_path_psx = Path(out_path)
    out_path_psx.mkdir(exist_ok=True, parents=True)

    (
        ffmpeg.input(input_file)
        .filter("fps", fps=fps)
        .output(str(out_path_psx / "frame_%04d.png"))
        .run()
    )
