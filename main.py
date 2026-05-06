from __future__ import annotations

import tkinter as tk

import cv2 as cv
import numpy as np

from process import GestureBackend

PERF_INFO: bool = False


def bgr_to_tk_image(
    frame: np.ndarray, display_w: int = 320, display_h: int = 240
) -> tk.PhotoImage:
    """Convert an OpenCV BGR frame to a Tkinter PhotoImage for display.

    Args:
        frame: Source BGR image from OpenCV.
        display_w: Display width after resize.
        display_h: Display height after resize.

    Returns:
        A Tkinter-compatible PhotoImage.
    """
    resized = cv.resize(frame, (display_w, display_h), interpolation=cv.INTER_AREA)
    rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ppm_header = f"P6 {w} {h} 255 ".encode("ascii")
    data = ppm_header + rgb.tobytes()
    return tk.PhotoImage(data=data, format="PPM")


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
    root.geometry("400x430")
    root.resizable(False, False)

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

    camera_label = tk.Label(
        root, width=320, height=240, bg="black", relief="sunken", bd=1
    )
    camera_label.pack(padx=10, pady=(10, 8))

    def update_camera_view() -> None:
        frame = backend.get_latest_display_frame()
        if frame is not None:
            image = bgr_to_tk_image(frame)
            camera_label.configure(image=image)
            camera_label.image = image  # ty: ignore[unresolved-attribute]

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
