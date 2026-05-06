from __future__ import annotations

import tkinter as tk

import cv2 as cv
import numpy as np

from process import GestureBackend

PERF_INFO: bool = False
"""Print per-frame timing stats on close when True."""

MAX_DISPLAY_WIDTH: int = 400
"""Maximum width of the camera preview in pixels.  Height is derived
from the frame's aspect ratio."""


def bgr_to_tk_image(
    frame: np.ndarray, max_w: int = MAX_DISPLAY_WIDTH
) -> tuple[tk.PhotoImage, int, int]:
    """Convert an OpenCV BGR frame to a Tkinter PhotoImage.

    The frame is scaled so its width does not exceed *max_w*, keeping
    the original aspect ratio.

    Args:
        frame: Source BGR image from OpenCV.
        max_w: Maximum display width in pixels.

    Returns:
        A ``(photo, w, h)`` tuple giving the PhotoImage and its
        dimensions.
    """
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        w = max_w
        h = int(h * scale)
    resized = cv.resize(frame, (w, h), interpolation=cv.INTER_AREA)
    rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    ppm_header = f"P6 {w} {h} 255 ".encode("ascii")
    data = ppm_header + rgb.tobytes()
    return tk.PhotoImage(data=data, format="PPM"), w, h


def build_gui(backend: GestureBackend, perf_info: bool = False) -> tk.Tk:
    """Build the Tkinter GUI window.

    Args:
        backend: The gesture-processing backend instance.
        perf_info: If True, print timing stats on close.

    Returns:
        The root Tk window (call ``mainloop()`` on it to run).
    """
    root = tk.Tk()
    root.title("Gesture Recognition")

    title = tk.Label(root, text="Gesture Recognition", font=("Segoe UI", 14, "bold"))
    title.pack(pady=(10, 8))

    status_var = tk.StringVar(value="Processing: stopped")
    status_label = tk.Label(root, textvariable=status_var, font=("Segoe UI", 10))
    status_label.pack(pady=(6, 8))

    button_row = tk.Frame(root)
    button_row.pack(pady=6)

    def on_start() -> None:
        backend.start_processing()
        status_var.set("Processing: running")

    def on_stop() -> None:
        backend.stop_processing()
        status_var.set("Processing: stopped")

    start_btn = tk.Button(
        button_row,
        text="Start Processing",
        width=16,
        command=on_start,
        bg="#2e7d32",
        fg="white",
    )
    start_btn.grid(row=0, column=0, padx=6)

    stop_btn = tk.Button(
        button_row,
        text="Stop",
        width=16,
        command=on_stop,
        bg="#c62828",
        fg="white",
    )
    stop_btn.grid(row=0, column=1, padx=6)

    rotate_ccw_btn = tk.Button(
        button_row,
        text="\u21ba",  # ↺
        width=4,
        command=backend.rotate_ccw,
    )
    rotate_ccw_btn.grid(row=0, column=2, padx=2)

    rotate_cw_btn = tk.Button(
        button_row,
        text="\u21bb",  # ↻
        width=4,
        command=backend.rotate_cw,
    )
    rotate_cw_btn.grid(row=0, column=3, padx=2)

    camera_label = tk.Label(root, bg="black", relief="sunken", bd=1)
    camera_label.pack(padx=10, pady=(10, 8))

    _sized: bool = False

    def update_camera_view() -> None:
        nonlocal _sized
        frame = backend.get_latest_display_frame()
        if frame is not None:
            image, dw, dh = bgr_to_tk_image(frame)
            camera_label.configure(image=image, width=dw, height=dh)
            camera_label.image = image  # ty: ignore[unresolved-attribute]

            if not _sized:
                _sized = True
                control_height = (
                    title.winfo_reqheight()
                    + status_label.winfo_reqheight()
                    + start_btn.winfo_reqheight()
                    + 60
                )
                root.geometry(f"{dw + 20}x{dh + control_height}")
                root.resizable(False, False)

        if not backend.app_stop_event.is_set():
            root.after(20, update_camera_view)

    def on_close() -> None:
        backend.stop_all()
        if perf_info:
            backend.print_average_timings()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(20, update_camera_view)
    return root


def main() -> None:
    """Application entry point.

    Creates the gesture backend, starts the camera, builds the GUI,
    and enters the Tkinter main loop.
    """
    backend = GestureBackend(perf_info=PERF_INFO)
    backend.start_camera()

    gui = build_gui(backend, perf_info=PERF_INFO)
    gui.mainloop()


if __name__ == "__main__":
    main()
