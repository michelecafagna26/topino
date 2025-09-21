from pathlib import Path
import streamlit as st
import cv2
import tempfile
from PIL import Image
import plotly.express as px
from streamlit_cropper import st_cropper

from topino.frame_processor import process_frames_parallel
from topino.video import extract_frames

def parse_time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS or MM:SS or SS into total seconds."""
    parts = time_str.strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return 0

    if len(parts) == 3:  # HH:MM:SS
        h, m, s = parts
    elif len(parts) == 2:  # MM:SS
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 1:  # SS
        h, m, s = 0, 0, parts[0]
    else:
        return 0

    return h * 3600 + m * 60 + s

st.title("Topino App 🐭")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Extract first frame
    cap = cv2.VideoCapture(tfile.name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()

    if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            crop_disabled = st.session_state.get("crop_disabled", False)

            if not crop_disabled:
                st.write("Select the relevant region to analyse:")
                cropped, box = st_cropper(
                    frame_pil,
                    realtime_update=True,
                    box_color="#FF0000",
                    aspect_ratio=None,  # free selection
                    return_type="both",
                )
                if st.button("Submit selection"):
                    st.session_state["crop_disabled"] = True
                    st.success("Selection submitted. Please sit tight while processing.")
                    box = (box['left'], box['top'], box['left'] + box['width'], box['top'] + box['height'])

                    # Debugging
                    st.image(frame_pil.crop(box))


                    with tempfile.TemporaryDirectory(prefix="frames_") as frames_dir:

                        with st.status("Please wait: Processing...", expanded=True) as status:

                            st.write("Extracting Frames...")
                            extract_frames(input_file=str(tfile.name), out_path=str(frames_dir), fps=1)

                            st.write(f"Computing motion index ...")
                            # Process frames in parallel
                            df = process_frames_parallel(input_path=str(frames_dir), box=box)

                            st.session_state["motion_index"] = df

                        fig = px.line(df, x="time", y="value", title=f"Motion Index {video_file.name}")
                        st.write("Motion index DataFrame:")
                        st.plotly_chart(fig)
                        st.download_button(
                            label="Download CSV",
                            data=df.to_csv().encode("utf-8"),
                            file_name=f"{Path(video_file.name).stem}_motion_index.csv",
                            mime="text/csv",
                            icon=":material/download:",
                        )

                        time_input = st.text_input("Enter start time (HH:MM:SS)", "00:00:00")
                        start_time = parse_time_to_seconds(time_input)

                        if st.button("Go to"):
                            try:
                                start_time = parse_time_to_seconds(time_input)
                                st.video(video_file, start_time=start_time)
                            except Exception:
                                st.warning("Invalid time format! Please enter time as HH:MM:SS.")

            if st.session_state.get("motion_index", None) is not None:
                df = st.session_state["motion_index"]
                fig = px.line(df, x="time", y="value", title=f"Motion Index {video_file}")
                st.write("Motion index DataFrame:")
                st.plotly_chart(fig)
                st.download_button(
                    label="Download CSV",
                    data=df.to_csv().encode("utf-8"),
                    file_name=f"{Path(video_file.name).stem}_motion_index.csv",
                            mime="text/csv",
                            icon=":material/download:",
                        )
                
                time_input = st.text_input("Enter start time (HH:MM:SS)", "00:00:00")
                start_time = parse_time_to_seconds(time_input)

                if st.button("Go to"):
                    try:
                        start_time = parse_time_to_seconds(time_input)
                        st.video(video_file, start_time=start_time)
                    except Exception:
                        st.warning("Invalid time format! Please enter time as HH:MM:SS.")