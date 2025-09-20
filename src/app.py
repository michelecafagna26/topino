import streamlit as st
import cv2
import tempfile
import base64

st.title("Video Annotation App")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Extract frame
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Video has {total_frames} frames.")

    frame_number = st.slider("Select frame number to annotate", 0, total_frames-1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode()

        st.write("Draw and edit a rectangle on the frame:")

        html_code = f"""
        <canvas id="canvas" width="{frame_rgb.shape[1]}" height="{frame_rgb.shape[0]}" style="border:1px solid #000000;"></canvas>
        <p>Rectangle coordinates (x, y, width, height): <span id="coords">None</span></p>
        <button onclick="submitCoords()">Submit</button>

        <script>
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var img = new Image();
        img.src = "data:image/png;base64,{img_str}";
        img.onload = function() {{
            drawRect();
        }};

        var rect = {{x:50, y:50, w:150, h:100}};
        var dragging = false;
        var resizing = false;
        var currentHandle = null;
        var offsetX, offsetY;
        const handleSize = 10;

        function drawRect() {{
            ctx.clearRect(0,0,canvas.width,canvas.height);
            ctx.drawImage(img, 0,0);

            // 3D-style rectangle (shadow + highlight)
            ctx.strokeStyle = "rgba(255,0,0,0.8)";
            ctx.lineWidth = 3;
            ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

            ctx.strokeStyle = "rgba(255,200,200,0.6)";
            ctx.lineWidth = 1;
            ctx.strokeRect(rect.x+2, rect.y+2, rect.w, rect.h);

            // Draw corner handles with "raised" effect
            for (const [dx, dy] of [[0,0],[rect.w,0],[0,rect.h],[rect.w,rect.h]]) {{
                ctx.fillStyle = "blue";
                ctx.fillRect(rect.x+dx-handleSize/2, rect.y+dy-handleSize/2, handleSize, handleSize);
                ctx.strokeStyle = "white";
                ctx.lineWidth = 2;
                ctx.strokeRect(rect.x+dx-handleSize/2, rect.y+dy-handleSize/2, handleSize, handleSize);
            }}

            document.getElementById('coords').innerText = JSON.stringify(rect);
        }}

        function getHandle(mx, my) {{
            if (Math.abs(mx-rect.x)<handleSize && Math.abs(my-rect.y)<handleSize) return "tl";
            if (Math.abs(mx-(rect.x+rect.w))<handleSize && Math.abs(my-rect.y)<handleSize) return "tr";
            if (Math.abs(mx-rect.x)<handleSize && Math.abs(my-(rect.y+rect.h))<handleSize) return "bl";
            if (Math.abs(mx-(rect.x+rect.w))<handleSize && Math.abs(my-(rect.y+rect.h))<handleSize) return "br";
            return null;
        }}

        canvas.onmousemove = function(e) {{
            var mx = e.offsetX;
            var my = e.offsetY;

            // Change cursor depending on hover
            var handle = getHandle(mx,my);
            if (handle) {{
                canvas.style.cursor = "nwse-resize";
            }} else if (mx>rect.x && mx<rect.x+rect.w && my>rect.y && my<rect.y+rect.h) {{
                canvas.style.cursor = "move";
            }} else {{
                canvas.style.cursor = "default";
            }}

            if (dragging) {{
                rect.x = Math.max(0, Math.min(canvas.width-rect.w, mx - offsetX));
                rect.y = Math.max(0, Math.min(canvas.height-rect.h, my - offsetY));
                drawRect();
            }} else if (resizing && currentHandle) {{
                switch(currentHandle) {{
                    case "tl":
                        var newX = Math.min(rect.x+rect.w-10, Math.max(0, mx));
                        var newY = Math.min(rect.y+rect.h-10, Math.max(0, my));
                        rect.w = rect.x + rect.w - newX;
                        rect.h = rect.y + rect.h - newY;
                        rect.x = newX;
                        rect.y = newY;
                        break;
                    case "tr":
                        var newX = Math.max(rect.x+10, Math.min(canvas.width, mx));
                        var newY = Math.min(rect.y+rect.h-10, Math.max(0, my));
                        rect.w = newX - rect.x;
                        rect.h = rect.y + rect.h - newY;
                        rect.y = newY;
                        break;
                    case "bl":
                        var newX = Math.min(rect.x+rect.w-10, Math.max(0, mx));
                        var newY = Math.max(rect.y+10, Math.min(canvas.height, my));
                        rect.w = rect.x + rect.w - newX;
                        rect.h = newY - rect.y;
                        rect.x = newX;
                        break;
                    case "br":
                        rect.w = Math.max(10, Math.min(canvas.width-rect.x, mx - rect.x));
                        rect.h = Math.max(10, Math.min(canvas.height-rect.y, my - rect.y));
                        break;
                }}
                drawRect();
            }}
        }};

        canvas.onmousedown = function(e) {{
            var mx = e.offsetX;
            var my = e.offsetY;
            currentHandle = getHandle(mx,my);
            if (currentHandle) {{
                resizing = true;
            }} else if (mx>rect.x && mx<rect.x+rect.w && my>rect.y && my<rect.y+rect.h) {{
                dragging = true;
                offsetX = mx - rect.x;
                offsetY = my - rect.y;
            }}
        }};

        canvas.onmouseup = function(e) {{
            dragging = false;
            resizing = false;
            currentHandle = null;
        }};

        function submitCoords() {{
            const coords = JSON.stringify(rect);
            alert("Rectangle coordinates: " + coords);
        }}
        </script>
        """

        st.components.v1.html(html_code, height=frame_rgb.shape[0]+120, scrolling=True)
