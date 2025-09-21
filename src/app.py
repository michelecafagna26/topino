from pathlib import Path
import streamlit as st
import cv2
import tempfile
from PIL import Image
import plotly.express as px
from streamlit_cropper import st_cropper

from topino.frame_processor import process_frames_parallel
from topino.video import extract_frames

st.title("Video Annotation App")

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
                box = st_cropper(
                    frame_pil,
                    realtime_update=True,
                    box_color="#FF0000",
                    aspect_ratio=None,  # free selection
                    return_type="box",
                )
                if st.button("Submit selection"):
                    st.session_state["crop_disabled"] = True
                    st.session_state["crop_box"] = box
                    st.success("Selection submitted. Please do not change the selection while processing.")
                    box_dict = st.session_state.get("crop_box", None)
                    box = (box_dict['left'], box_dict['top'], box_dict['width'], box_dict['height'])


                    with tempfile.TemporaryDirectory(prefix="frames_") as frames_dir:
                        
                        with st.status("Please weait: Processing...", expanded=True) as status:

                            st.write("Extracting Frames...")
                            extract_frames(input_file=str(tfile.name), out_path=str(frames_dir), fps=1)

                            st.write(f"Computing motion index ...")
                            # Process frames in parallel
                            df = process_frames_parallel(input_path=str(frames_dir), box=box)

                            fig = px.line(df, x="time", y="value", title="Motion Index Over Time")
                            st.write("Motion index DataFrame:")
                            st.plotly_chart(fig)

                            st.download_button(
                                label="Download CSV",
                                data=df.to_csv().encode("utf-8"),
                                file_name=f"{Path(video_file.name).stem}_motion_index.csv",
                                mime="text/csv",
                                icon=":material/download:",
                            )

            

