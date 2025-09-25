# Topino: Fast Motion Detection in Long Rat Recordings
A lightweight tool for detecting motion in extended laboratory videos, built to save researchers hours of manual review.

## 🚀 Overview

Tired of spending hours combing through day-long lab videos of rats? 

**Topino** automates motion detection in long-form recordings, generating motion plots in minutes. It highlights subtle movements across the entire video, helping researchers zooming in the most relevant intervals for qualitative analysis.

Although originally designed for rat behavior studies, Topino uses a simple motion detection algorithm similar to those found in CCTV systems, making it suitable for virtually any kind of video.

**🔗 Live Demo on 🤗 Spaces: [Topino](https://huggingface.co/spaces/michelecafagna26/topino-app)**

## 🧪Getting Started

### Option 1: Run Locally with Streamlit


```bash
# make sure uv is installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install the package
uv sync

# run the app
uv run streamlit run src/app

```

### Option 2: Run with Docker

```bash
docker build -t topino
```

Run the container

```bash
docker container run -p 8510:8510-t topino 
```

Access the app in your browser at: ```http://0.0.0.0:8501/```

### Option 3: Run Docker Image from Hugging Face Spaces

Run:

```bash
# Authenticate using your Hugging Face token: https://hf.co/settings/tokens
docker login registry.hf.space
# Pull and run the container
docker run -it -p 7860:7860 --platform=linux/amd64 \
	registry.hf.space/michelecafagna26-topino-app:latest
```


## 🛠️ Command Line Interface (CLI)

Topino performs parallelized motion analysis to maximize performance across hardware setups.

## Step 1: Extract Frames
```bash
uv run topino extract-frames --input-file your_clip.mp4 --out-path path/to/save/frames
```

## Step 2: Compute Motion Index
```bash
uv run topino process --input-path path/to/save/frames --out-path motion_index.csv
```

## 📊 Offline Visualizer

Download the generated motion plot as ```csv``` file and view it using the HTML visualizer: ```viz/topino_plot.html```.

