# Topino: Fast Motion Detection in Long Rat Recordings
A lightweight tool for detecting motion in day-long laboratory rat recordings.

Tired of spending hours combing through day-long lab videos of rats? 

Topino is a lightweight tool designed to streamline your workflow. It automatically analyzes extended recordings and generates motion plots within minutes, pinpointing even subtle activity across the entire video.

By surfacing potentially interesting intervals, Topino helps researchers focus their qualitative inspection where it matters most—saving time, reducing fatigue, and accelerating your analyses.

Topino is not specific for rats, at the core it uses motion detection algorithm also used in CCTV cameras. Thus, it can detects any kind of motion in any kind of video.

# Streamlit App

You can use topino via ```steamlit``` by running:

```bash
# make sure uv is installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install the package
uv sync

# run the app
uv run streamlit run src/app

```

## CLI

Depending on your machine the computation time can vary, however ```topino``` performes all the computation in parallel, optimizing your computational resources.

You can first extract the video frames using:
```bash
uv run topino extract-frames --input-file your_clip.mp4 --out-path path/to/save/frames
```

Compute the motion index
```bash
uv run topino process --input-path path/to/save/frames --out-path motion_index.csv
```

You can use ```viz/topino_plot.html``` to visualize the plot.

