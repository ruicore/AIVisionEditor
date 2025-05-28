import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, colorchooser, simpledialog
from PIL import Image, ImageTk, ImageFilter, ImageDraw, ImageFont
import cv2
import os
import io
import numpy as np
import torch
from torchvision import models, transforms
import requests
from io import BytesIO
import pygame
import base64
from gradio_client import Client, handle_file
import tempfile

# Initialize pygame for sound
pygame.mixer.init()

TOKEN  ='1b4fd031-9ed5-4df4-a648-3e625e5fc3c9'

class PhotoEditorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("AI Smart Photo Editor")
        self.geometry("860x860")
        self.resizable(True, True)
        
        # State management
        self.current_page = "Home"
        self.dark_mode = True
        self.current_view = "load_image"
        self.brush_color = "black"
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.bg_option = ctk.StringVar(value="White")
        self.photobooth_images = []
        self.strip_created = False
        self.drawing = False
        self.is_cropping = False
        self.mask = None
        self.image =None

        self.basicUrl = "https://genai.hkbu.edu.hk/general/rest"
        self.modelName = "gpt-4-o-mini" 
        self.apiVersion = "2024-05-01-preview"
        self.apiKey = os.environ['GENAI_API_KEY']
        self.deepAi = os.environ['DEEP_AI_API_KEY']

        self.undo_stack = []
        self.redo_stack = []

        # self.shutter_sound = pygame.mixer.Sound("shutter_sound.mp3")
        
        # Load shutter sound ONCE
        try:
            self.shutter_sound = pygame.mixer.Sound("data/shutter_sound.mp3")
        except Exception as e:
            print("Could not load shutter sound:", e)

        # Initialize UI
        self.configure_layout()
        self.create_left_sidebar()
        self.create_top_menu()
        self.create_main_content()
        
        # Set default theme
        self.set_theme()
        self.show_load_image_view()

    def configure_layout(self):
        """Set up the grid layout structure"""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.grid_columnconfigure(0, weight=1)  # Left sidebar column
        self.grid_columnconfigure(1, weight=7)
        
        # Top menu spans all columns
        self.top_menu = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.top_menu.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        # Left sidebar in column 0
        self.left_sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.left_sidebar.grid(row=1, column=0, sticky="nsew")
        
        # Main content in column 1
        self.main_content = ctk.CTkFrame(self, width=700, corner_radius=0)  # Set explicit width
        self.main_content.grid(row=1, column=1, sticky="nsew")

    def create_top_menu(self):
        """Create the top menu bar with controls"""
        # Left spacer to align with sidebar
        ctk.CTkLabel(
            self.top_menu,
            text="",
            width=200
        ).pack(side="left")
        
        # Theme toggle
        self.theme_switch = ctk.CTkSwitch(
            self.top_menu,
            text="Dark Mode",
            command=self.toggle_theme
        )
        self.theme_switch.pack(side="left", padx=10)
        
        # Menu buttons
        menu_buttons = ["File", "Edit", "View", "Window", "Help"]
        for btn_text in menu_buttons:
            ctk.CTkButton(
                self.top_menu,
                text=btn_text,
                width=60,
                height=20,
                fg_color="#2A3F54",
                hover_color="#3D5A80",
                corner_radius=0,
                font=ctk.CTkFont(size=12)
            ).pack(side="left", padx=0)
        
        # Current page indicator
        self.page_label = ctk.CTkLabel(
            self.top_menu,
            text=f"Current: {self.current_page}",
            font=ctk.CTkFont(size=12)
        )
        self.page_label.pack(side="right", padx=20)

    def create_left_sidebar(self):
        """Create the editor tools sidebar with all functionality"""
        # Undo and Redo buttons
        undo_redo_frame = ctk.CTkFrame(self.left_sidebar)
        undo_redo_frame.pack(pady=(10, 5), padx=10, fill="x")

        ctk.CTkButton(
            undo_redo_frame,
            text="Undo",
            command=self.undo,
            width=80,
            height=30,
            fg_color="#2A3F54",
            hover_color="#3D5A80",
            corner_radius=4,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            undo_redo_frame,
            text="Redo",
            command=self.redo,
            width=80,
            height=30,
            fg_color="#2A3F54",
            hover_color="#3D5A80",
            corner_radius=4,
        ).pack(side="left", padx=5)

        # Editor sections data
        sections = [
            {
                "title": "IMAGE FILTERS",
                "buttons": [
                    ("Rotate", self.rotate_90),
                    ("Crop", self.crop),
                    ("Blur", lambda: self.apply_filter(ImageFilter.BLUR)),
                    ("Sharpen", lambda: self.apply_filter(ImageFilter.SHARPEN)),
                    ("Contrast", lambda: self.apply_filter(ImageFilter.DETAIL)),
                    ("B&W", self.apply_grayscale),
                ],
            },
            {
                "title": "PROCESSING",
                "buttons": [
                    ("Remove BG", self.remove_background),
                    ("Blur BG", self.blur_background),
                    ("Replace BG", self.replace_background),
                    # ("Eraser", self.activate_eraser),
                    # ("Inpainting", self.inpaint_image),
                    ("Face Detection", self.detect_faces_viola),
                    ("Draw", self.enable_drawing_mode),
                    ("Add Text", self.add_text),
                    ("Create Strip", self.create_single_image_strip),
                ],
            },
        ]

        # Build UI components for each section
        for section in sections:
            # Section header
            ctk.CTkLabel(
                self.left_sidebar,
                text=section["title"],
                font=ctk.CTkFont(weight="bold"),
                anchor="w",
            ).pack(pady=(15, 5), padx=10, fill="x")

            # Section buttons
            for btn_text, command in section["buttons"]:
                ctk.CTkButton(
                    self.left_sidebar,
                    text=btn_text,
                    command=command,
                    anchor="w",
                    width=180,
                    height=30,
                    fg_color="#2A3F54",
                    hover_color="#3D5A80",
                    corner_radius=4,
                ).pack(pady=2, padx=5)

        # Move Strip Background dropdown between AI Tools and Processing
        bg_frame = ctk.CTkFrame(self.left_sidebar)
        bg_frame.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(
            bg_frame,
            text="Strip Background:",
            font=ctk.CTkFont(size=12),
        ).pack()

        self.bg_dropdown = ctk.CTkOptionMenu(
            bg_frame,
            variable=self.bg_option,
            values=["White", "Pink", "Sky Blue", "Polka Dots", "Cartoon"],
            command=self.update_strip_background,
        )
        self.bg_dropdown.pack(pady=5)

        ai_tools_section = {
            "title": "AI TOOLS",
            "buttons": [
                ("Auto-Enhance", self.auto_enhance),
                ("AI Assistant", self.ask_chatbot),
                ("AI Editing", self.show_load_image_view),

            ],
        }

        ctk.CTkLabel(
            self.left_sidebar,
            text=ai_tools_section["title"],
            font=ctk.CTkFont(weight="bold"),
            anchor="w",
        ).pack(pady=(15, 5), padx=10, fill="x")

        for btn_text, command in ai_tools_section["buttons"]:
            ctk.CTkButton(
                self.left_sidebar,
                text=btn_text,
                command=command,
                anchor="w",
                width=180,
                height=30,
                fg_color="#2A3F54",
                hover_color="#3D5A80",
                corner_radius=4,
            ).pack(pady=2, padx=5)

    def create_main_content(self):
        """Create the main content area with image handling"""
        # Create a canvas with scrollbar
        self.canvas = Canvas(self.main_content, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Scrollbar
        scrollbar = ctk.CTkScrollbar(self.main_content, orientation="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Content frame inside canvas
        self.content_frame = ctk.CTkFrame(self.canvas)
        self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        
        # View toggle buttons
        btn_frame = ctk.CTkFrame(self.content_frame)
        btn_frame.pack(pady=10)
        
        self.load_view_btn = ctk.CTkButton(
            btn_frame,
            text="Load Image View",
            command=self.toggle_load_image_buttons,
            fg_color="#990000"  # Light blue
        )
        self.load_view_btn.pack(side="left", padx=5)
        
        self.photo_booth_btn = ctk.CTkButton(
            btn_frame,
            text="Photo Booth View",
            command=self.toggle_photo_booth_buttons,
            fg_color="#990000"  # Light blue
        )
        self.photo_booth_btn.pack(side="left", padx=5)
        
        # Little hint label
        # self.view_hint = ctk.CTkLabel(self.content_frame, text="👆 Click above to choose your view", text_color="gray")
        # self.view_hint.pack(pady=(2, 10))

        # Image handling buttons
        self.img_btn_frame = ctk.CTkFrame(self.content_frame)
        self.img_btn_frame.pack(pady=10)
        
        self.load_btn = ctk.CTkButton(
            self.img_btn_frame,
            text="Load Image",
            command=self.load_image
        )
        
        self.capture_btn = ctk.CTkButton(
            self.img_btn_frame,
            text="Capture Image",
            command=self.capture_photobooth
        )
        
        self.reset_btn = ctk.CTkButton(
            self.img_btn_frame,
            text="Reset Image",
            command=self.reset_image
        )
        
        self.delete_btn = ctk.CTkButton(
            self.img_btn_frame,
            text="Delete Image",
            command=self.delete_image
        )
        
        # Image canvas
        self.image_canvas = Canvas(self.content_frame, bg="white", width=600, height=600)
        self.image_canvas.pack(pady=10)
        
        # Save button
        self.save_btn = ctk.CTkButton(
            self.content_frame,
            text="Save Image",
            command=self.save_image
        )
        self.save_btn.pack(pady=10)
        
        # Chatbot UI
        self.chat_frame = ctk.CTkFrame(self.content_frame)
        self.chat_frame.pack_forget()  # Hide until button is clicked
        self.chat_widgets_initialized = False
        # self.question_entry = ctk.CTkEntry(self.chat_frame, placeholder_text="Ask about the image...")
        # self.question_entry.pack(pady=5)
        
        # self.response_box = ctk.CTkTextbox(self.chat_frame, height=100)
        # self.response_box.pack(pady=5)
        
        # Update scroll region
        self.content_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    # "#1f6aa5"
    def highlight_active_view(self, active="load"):
        if active == "load":
            self.load_view_btn.configure(fg_color="#4CAF50")  # Green for active
            self.photo_booth_btn.configure(fg_color="#990000")  # Blue for inactive
        else:
            self.photo_booth_btn.configure(fg_color="#4CAF50")
            self.load_view_btn.configure(fg_color="#990000")


    # ===== IMAGE HANDLING METHODS =====
    def display_image(self):
        """Displays the current image in the canvas"""
        if hasattr(self, 'image') and self.image:
            aspect_ratio = self.image.width / self.image.height
            new_width = 600
            new_height = int(new_width / aspect_ratio)
            self.resized_image = self.image.resize((new_width, new_height), Image.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(self.resized_image)
            self.image_canvas.config(width=new_width, height=new_height)
            self.image_canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
            self.image_canvas.image = self.image_tk

    def load_image(self):
        """Load an image from file"""
        if self.photobooth_images:
            messagebox.showwarning("Warning", "Please delete the photobooth strip before uploading a new image.")
            return
    
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.original_image = Image.open(file_path)
            self.image = self.original_image.copy()
            self.mask = np.zeros((self.image.height, self.image.width), dtype=np.uint8)
            self.strip_created = False
            self.display_image()

    def reset_image(self):
        """Reset to original image"""
        if hasattr(self, 'original_image') and self.original_image:
            self.image = self.original_image.copy()
            self.strip_created = False 
            self.display_image()

    def delete_image(self):
        """Delete the current image or photobooth strip"""
        if hasattr(self, 'image') and self.image:
            self.image = None
            self.original_image = None
            self.image_canvas.delete("all")
            self.strip_created = False
            self.photobooth_images = []
            self.strip = None
            messagebox.showinfo("Success", "Image deleted successfully.")
        elif self.photobooth_images:
            self.photobooth_images = []
            self.strip = None
            self.image_canvas.delete("all")
            self.strip_created = False
            messagebox.showinfo("Success", "Photobooth strip deleted successfully.")
        else:
            messagebox.showwarning("Warning", "No image or photobooth strip to delete!")
            
    def save_image(self):
        """Save the current image into a 'result' directory"""
        if hasattr(self, 'image') and self.image:
            # Ensure the 'result' directory exists
            result_dir = "result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            # Ask the user for a file name
            file_name = filedialog.asksaveasfilename(
                initialdir=result_dir,
                title="Save Image As",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                defaultextension=".png"
            )
            if not file_name:
                return  # User canceled the save dialog

            # Save the image
            try:
                # Convert RGBA to RGB if saving as PNG
                if self.image.mode == "RGBA":
                    image_to_save = self.image.convert("RGB")
                else:
                    image_to_save = self.image

                image_to_save.save(file_name)
                messagebox.showinfo("Success", f"Image saved successfully as '{file_name}'")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")
        else:
            messagebox.showwarning("Warning", "No image to save!")

    # ===== IMAGE PROCESSING METHODS =====
    def apply_filter(self, filter_type):
        """Apply a filter to the image"""
        if hasattr(self, 'image') and self.image:
            self.apply_action()
            self.image = self.image.filter(filter_type)
            self.display_image()
        else: 
            messagebox.showwarning("Warning", "No image to delete!")

    def apply_grayscale(self):
        """Convert image to grayscale"""
        if hasattr(self, 'image') and self.image:
            self.apply_action()
            self.image = self.image.convert("L").convert("RGB")
            self.display_image()
        else: 
            messagebox.showwarning("Warning", "No image to delete!")

    def rotate_90(self):
        """Rotate image 90 degrees"""
        if hasattr(self, 'image') and self.image:
            self.apply_action()
            self.image = self.image.rotate(90, expand=True)
            self.display_image()
        else: 
            messagebox.showwarning("Warning", "No image to delete!")

    def crop(self):
        """Enable cropping mode"""
        if not hasattr(self, 'image') or not self.image:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        self.apply_action()
        self.is_cropping = True
        self.image_canvas.bind("<ButtonPress-1>", self.start_crop)
        self.image_canvas.bind("<B1-Motion>", self.draw_crop)
        self.image_canvas.bind("<ButtonRelease-1>", self.end_crop)

    def start_crop(self, event):
        """Start crop selection"""
        self.start_x = event.x
        self.start_y = event.y
        self.rectangle = self.image_canvas.create_rectangle(
            self.start_x, self.start_y, 
            self.start_x, self.start_y, 
            outline='red'
        )

    def draw_crop(self, event):
        """Update crop selection"""
        self.image_canvas.coords(
            self.rectangle, 
            self.start_x, self.start_y, 
            event.x, event.y
        )

    def end_crop(self, event):
        """Apply crop to image"""
        if hasattr(self, 'rectangle') and self.rectangle:
            x0, y0, x1, y1 = self.image_canvas.coords(self.rectangle)
            scale_x = self.image.width / self.resized_image.width
            scale_y = self.image.height / self.resized_image.height
            
            orig_x0 = int(x0 * scale_x)
            orig_y0 = int(y0 * scale_y)
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int(y1 * scale_y)
            
            # Ensure valid crop area
            if orig_x1 > orig_x0 and orig_y1 > orig_y0:
                self.image = self.image.crop((orig_x0, orig_y0, orig_x1, orig_y1))
                self.display_image()
            else:
                messagebox.showwarning("Warning", "Invalid crop area")
            
            # Clean up
            self.image_canvas.delete(self.rectangle)
            self.image_canvas.unbind("<ButtonPress-1>")
            self.image_canvas.unbind("<B1-Motion>")
            self.image_canvas.unbind("<ButtonRelease-1>")
            self.is_cropping = False

    # ===== DRAWING METHODS =====
    # def enable_drawing_mode(self):
    #     """Enable drawing on the image"""
    #     self.image_canvas.bind("<Button-1>", self.start_drawing)
    #     self.image_canvas.bind("<B1-Motion>", self.draw)
    #     self.image_canvas.bind("<ButtonRelease-1>", self.stop_drawing)


    def choose_brush_settings(self):
        """Popup to choose brush color and size with animated preview"""
        dialog = tk.Toplevel()
        dialog.title("Brush Settings")
        dialog.geometry("350x400")
        dialog.configure(bg="#fdf6f0")
        dialog.grab_set()

        # Cute Fonts
        font_title = ("Helvetica", 14, "bold")
        font_label = ("Helvetica", 11)

        # Title
        tk.Label(dialog, text="Customize Your Brush", font=font_title, bg="#fdf6f0", fg="#444").pack(pady=(15, 5))

        # --- Color Picker ---
        tk.Label(dialog, text="Pick Brush Color:", font=font_label, bg="#fdf6f0").pack()
        color_preview = tk.Canvas(dialog, width=60, height=30, bg="#fdf6f0", highlightthickness=0)
        color_preview.pack()
        selected_color = [self.brush_color or "#000000"]

        def pick_color():
            color = colorchooser.askcolor()[1]
            if color:
                selected_color[0] = color
                color_preview.delete("all")
                color_preview.create_rectangle(0, 0, 60, 30, fill=color, outline="#aaa")

        tk.Button(dialog, text="Choose Color", command=pick_color, bg="#ffe8d6", fg="#333", relief="ridge").pack(pady=(3, 10))

        # --- Brush Size Slider ---
        tk.Label(dialog, text="Brush Size:", font=font_label, bg="#fdf6f0").pack()

        size_var = tk.IntVar(value=self.brush_size if hasattr(self, 'brush_size') else 5)
        size_slider = tk.Scale(
            dialog, from_=1, to=50, orient="horizontal", variable=size_var,
            bg="#fdf6f0", fg="#444", highlightthickness=0, troughcolor="#ffddd2",
            sliderrelief="raised", length=200
        )
        size_slider.pack()

        # --- Live Preview (Animated) ---
        tk.Label(dialog, text="Preview:", font=font_label, bg="#fdf6f0").pack()
        preview_canvas = tk.Canvas(dialog, width=120, height=120, bg="#fdf6f0", highlightthickness=0)
        preview_canvas.pack()

        dot_id = None
        def animate_dot():
            nonlocal dot_id
            preview_canvas.delete("all")
            r = size_var.get()
            dot_id = preview_canvas.create_oval(
                60 - r, 60 - r,
                60 + r, 60 + r,
                fill=selected_color[0],
                outline=""
            )

        def live_update(*args):
            animate_dot()

        size_var.trace_add("write", live_update)
        animate_dot()

        # --- OK Button ---
        def apply_and_close():
            self.brush_color = selected_color[0]
            self.brush_size = size_var.get()  # Ensure brush size is updated here
            dialog.destroy()

        tk.Button(dialog, text="Apply", command=apply_and_close, bg="#cbf3f0", fg="#222", relief="groove").pack(pady=15)


    def enable_drawing_mode(self):
        """Enable drawing after choosing settings"""
        if hasattr(self, 'image') and self.image:
            self.apply_action()  # Save the current image state before enabling drawing
            self.choose_brush_settings()
            self.drawing_enabled = True
            self.image_canvas.bind("<Button-1>", self.start_drawing)
            self.image_canvas.bind("<B1-Motion>", self.draw)
            self.image_canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        else:
            messagebox.showwarning("Warning", "No image loaded to draw on!")

    def start_drawing(self, event):
        """Start drawing"""
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def stop_drawing(self, event):
        """Stop drawing"""
        if self.drawing:
            self.apply_action()  # Save the current image state after drawing
        self.drawing = False

    def draw(self, event):
        """Draw on the image"""
        if self.drawing and hasattr(self, 'image') and self.image:
            # Draw on canvas
            self.image_canvas.create_line(
                self.last_x, self.last_y, 
                event.x, event.y,
                fill=self.brush_color, 
                width=self.brush_size, 
                capstyle=ctk.ROUND, 
                smooth=True
            )
            
            # Draw on actual image
            scale_x = self.image.width / self.resized_image.width
            scale_y = self.image.height / self.resized_image.height
            
            real_last_x = int(self.last_x * scale_x)
            real_last_y = int(self.last_y * scale_y)
            real_x = int(event.x * scale_x)
            real_y = int(event.y * scale_y)
            
            draw = ImageDraw.Draw(self.image)
            draw.line(
                [(real_last_x, real_last_y), (real_x, real_y)],
                fill=self.brush_color,
                width=self.brush_size
            )
            
            self.last_x, self.last_y = event.x, event.y

    def set_brush_color(self):
        """Change brush color"""
        color_code = colorchooser.askcolor(title="Choose color")[1]
        if color_code:
            self.brush_color = color_code
            
    def add_text(self):
        """Add text to image with color and size picker"""
        def on_click(event):
            # Unbind after first click
            self.image_canvas.unbind("<Button-1>")

            # Save the current image state before adding text
            self.apply_action()

            # Ask for text
            text = simpledialog.askstring("Input", "Enter text:")
            if not text:
                return

            # Ask for color
            color = colorchooser.askcolor(title="Choose Text Color")[1]
            if not color:
                return

            # Create size chooser popup with preview
            def apply_text_size():
                size = size_slider.get()
                preview_window.destroy()

                # Scale click coordinates to image coordinates
                scale_x = self.image.width / self.resized_image.width
                scale_y = self.image.height / self.resized_image.height
                image_x = int(event.x * scale_x)
                image_y = int(event.y * scale_y)

                # Try font
                try:
                    font = ImageFont.truetype("Arial.ttf", size)
                except:
                    font = ImageFont.load_default()

                # Add text to original image
                draw = ImageDraw.Draw(self.image)
                draw.text((image_x, image_y), text, fill=color, font=font)

                # Show on canvas
                self.image_canvas.create_text(event.x, event.y, text=text, fill=color, font=("Arial", size))
                self.display_image()

            # Create size chooser window
            preview_window = tk.Toplevel()
            preview_window.title("Choose Text Size")
            preview_window.geometry("350x400")
            preview_window.configure(bg="#f0f0ff")

            tk.Label(preview_window, text="Brush Size", bg="#f0f0ff", font=("Arial", 12, "bold")).pack(pady=5)
            size_slider = tk.Scale(preview_window, from_=10, to=80, orient="horizontal", length=250, bg="#e6e6ff")
            size_slider.set(30)
            size_slider.pack()

            # Live preview
            preview_canvas = tk.Canvas(preview_window, width=150, height=60, bg="white", bd=1, relief="solid")
            preview_canvas.pack(pady=10)
            text_id = preview_canvas.create_text(75, 30, text="ABCD", font=("Arial", 30))

            def update_preview(val):
                preview_canvas.itemconfig(text_id, font=("Arial", int(val)))

            size_slider.config(command=update_preview)

            # Apply button
            apply_btn = tk.Button(preview_window, text="Apply", command=apply_text_size, bg="#b3b3ff", font=("Arial", 10, "bold"))
            apply_btn.pack(pady=5)

        self.image_canvas.bind("<Button-1>", on_click)

    # ===== PHOTOBOOTH METHODS =====

    def start_camera_preview(self):
        """Start live preview from webcam on canvas."""
        self.preview_active = True
        self.cap = cv2.VideoCapture(0)

        def update_frame():
            if self.preview_active and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    img = img.resize((500, 400))  # Resize to match your canvas
                    self.tk_preview_image = ImageTk.PhotoImage(img)
                    self.image_canvas.create_image(0, 0, anchor='nw', image=self.tk_preview_image)
                self.after(30, update_frame)  # Continuously call
        update_frame()

    def show_countdown(self, countdown_label, count, callback):
        """Show countdown before capture"""
        if count > 0:
            countdown_label.configure(text=f"Capturing in {count}...")
            self.after(1000, self.show_countdown, countdown_label, count-1, callback)
        else:
            countdown_label.configure(text="Capturing now!")
            self.after(500, callback)

    def capture_photobooth(self):
        """Capture 3 images with countdowns, live preview, and display final strip."""
        if self.image:
            messagebox.showwarning("Warning", "Please delete the uploaded image before capturing photobooth images.")
            return
        
        self.photobooth_images = []
        self.preview_active = True

        # Create webcam object
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return

        # Create label for countdown text
        countdown_label = ctk.CTkLabel(self.content_frame, text="", font=("Arial", 30), text_color="red")
        countdown_label.pack(pady=20)

        # Start live preview
        def update_preview():
            if self.preview_active and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)  # Flip horizontally to fix inverted camera
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame).resize((500, 400))
                    self.tk_preview_image = ImageTk.PhotoImage(img)
                    self.image_canvas.create_image(0, 0, anchor='nw', image=self.tk_preview_image)
                self.after(30, update_preview)
        update_preview()

        def capture_single_image():
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)  # Flip again for capture consistency
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    self.photobooth_images.append(img)

                    # 🔥 FLASH EFFECT (white overlay rectangle)
                    flash = self.image_canvas.create_rectangle(
                        0, 0,
                        self.image_canvas.winfo_width(),
                        self.image_canvas.winfo_height(),
                        fill='white',
                        outline=''
                    )
                    self.image_canvas.tag_raise(flash)
                    self.after(150, lambda: self.image_canvas.delete(flash))

                     # Play camera shutter sound
                    try:
                        self.shutter_sound.play()
                    except Exception as e:
                        print("Sound error:", e)

                    
                    # Flash effect
                    # self.image_canvas.configure(bg="white")
                    # self.after(100, lambda: self.image_canvas.configure(bg="black"))  # Set canvas back to black after 100ms

        def show_countdown(count, next_step):
            if count > 0:
                countdown_label.configure(text=f"{count}")
                self.after(1000, show_countdown, count - 1, next_step)
            else:
                countdown_label.configure(text="Snap!")
                self.after(500, next_step)

        def take_photos(index=0):
            if index < 3:
                show_countdown(3, lambda: [
                    capture_single_image(),
                    take_photos(index + 1)
                ])
            else:
                # Done capturing: cleanup
                self.preview_active = False
                self.cap.release()
                countdown_label.destroy()
                self.create_photobooth_strip(self.photobooth_images)

        take_photos()


    # def capture_photobooth(self):
    #     """Captures 3 images with countdowns and stacks them into a strip."""
    #     # Clear the photobooth images list before starting a new session
    #     self.photobooth_images = []

    #     cap = cv2.VideoCapture(0)
    #     if not cap.isOpened():
    #         messagebox.showerror("Error", "Could not open webcam")
    #         return

    #     countdown_label = ctk.CTkLabel(self.content_frame, text="Get Ready!", font=("Arial", 30), text_color="blue")
    #     countdown_label.pack(pady=20)

    #     def capture_image():
    #         ret, frame = cap.read()
    #         if ret:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             img = Image.fromarray(frame)
    #             self.photobooth_images.append(img)  # Append the captured image to the list

    #     def take_photos(index=0):
    #         if index < 3:
    #             self.show_countdown(countdown_label, 3, lambda: [capture_image(), take_photos(index + 1)])
    #         else:
    #             cap.release()
    #             countdown_label.destroy()
    #             self.create_photobooth_strip(self.photobooth_images)  # Create the strip from the captured images

    #     take_photos()

    def create_photobooth_strip(self, images):
        """Create strip from photobooth images."""
        if len(images) != 3:
            messagebox.showwarning("Warning", "Exactly 3 images are required to create a photobooth strip.")
            return

        self.apply_action()
        # Define strip dimensions
        strip_width = images[0].width + 120
        strip_height = sum(img.height for img in images) + 60 + 60 + 30 * (len(images) - 1)

        # Create background
        bg_choice = self.bg_option.get()
        if bg_choice == "White":
            strip = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))
        elif bg_choice == "Pink":
            strip = Image.new("RGB", (strip_width, strip_height), (255, 192, 203))
        elif bg_choice == "Sky Blue":
            strip = Image.new("RGB", (strip_width, strip_height), (135, 206, 235))
        elif bg_choice == "Polka Dots":
            if os.path.exists("data/polka.jpg"):
                bg_img = Image.open("data/polka.jpg").convert("RGB")
                bg_img = bg_img.resize((strip_width, strip_height))
                strip = bg_img.copy()
            else:
                messagebox.showwarning("Warning", "Polka Dots background not found. Using default white background.")
                strip = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))
        elif bg_choice == "Cartoon":
            if os.path.exists("data/cartoon.jpg"):
                bg_img = Image.open("data/cartoon.jpg").convert("RGB")
                bg_img = bg_img.resize((strip_width, strip_height))
                strip = bg_img.copy()
            else:
                messagebox.showwarning("Warning", "Cartoon background not found. Using default white background.")
                strip = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))
        else:
            strip = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))

        # Paste images onto the strip
        y_offset = 60
        for img in images:
            img_resized = img.resize((images[0].width, img.height))
            left_padding = 60
            strip.paste(img_resized, (left_padding, y_offset))
            y_offset += img_resized.height + 30

        # Update the main image and display it
        self.image = strip  # Update self.image with the created strip
        self.display_image()  # Display the strip on the canvas

    def create_single_image_strip(self):
        """Create a strip from a single image"""
        if self.strip_created:
            return

        if not hasattr(self, 'image') or not self.image:
            messagebox.showwarning("Warning", "No image loaded")
            return

        self.apply_action()

        strip_width = self.image.width + 100
        strip_height = self.image.height + 100

        # Create background
        bg_choice = self.bg_option.get()
        if bg_choice == "White":
            strip = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))
        elif bg_choice == "Pink":
            strip = Image.new("RGB", (strip_width, strip_height), (255, 192, 203))
        elif bg_choice == "Sky Blue":
            strip = Image.new("RGB", (strip_width, strip_height), (135, 206, 235))
        elif bg_choice == "Polka Dots":
            if os.path.exists("data/polka.jpg"):
                bg_img = Image.open("data/polka.jpg").convert("RGB")
                bg_img = bg_img.resize((strip_width, strip_height))
                strip = bg_img.copy()
            else:
                messagebox.showwarning("Warning", "Polka Dots background not found. Using default white background.")
                strip = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))
        elif bg_choice == "Cartoon":
            if os.path.exists("data/cartoon.jpg"):
                bg_img = Image.open("data/cartoon.jpg").convert("RGB")
                bg_img = bg_img.resize((strip_width, strip_height))
                strip = bg_img.copy()
            else:
                messagebox.showwarning("Warning", "Cartoon background not found. Using default white background.")
                strip = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))
        else:
            strip = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))

        # Paste image
        strip.paste(self.image, (50, 50))
        self.image = strip
        self.display_image()

        # Mark the strip as created
        self.strip_created = True

    def update_strip_background(self, selected_value=None):
        """Update the background of the existing strip based on the selected option."""
        if not self.strip_created:
            messagebox.showwarning("Warning", "No strip has been created yet!")
            return

        if not hasattr(self, 'image') or self.image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        self.apply_action()

        # Get the current strip dimensions
        strip_width, strip_height = self.image.width, self.image.height

        selected_bg = self.bg_option.get()
        # Create a new background based on the selected option
        if selected_bg == "White":
            new_bg = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))
        elif selected_bg == "Pink":
            new_bg = Image.new("RGB", (strip_width, strip_height), (255, 192, 203))
        elif selected_bg == "Sky Blue":
            new_bg = Image.new("RGB", (strip_width, strip_height), (135, 206, 235))
        elif selected_bg == "Polka Dots":
            bg_path = "data/polka.jpg"
            if os.path.exists(bg_path):
                bg_img = Image.open(bg_path).convert("RGB").resize((strip_width, strip_height))
                new_bg = bg_img
            else:
                messagebox.showwarning("Warning", "Polka Dots background not found.")
                new_bg = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))
        elif selected_bg == "Cartoon":
            bg_path = "data/cartoon.jpg"
            if os.path.exists(bg_path):
                bg_img = Image.open(bg_path).convert("RGB").resize((strip_width, strip_height))
                new_bg = bg_img
            else:
                messagebox.showwarning("Warning", "Cartoon background not found.")
                new_bg = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))
        else:
            new_bg = Image.new("RGB", (strip_width, strip_height), (255, 255, 255))

        # Extract the content of the strip (excluding the old background)
        # Assuming the content is centered with padding
        content_x = 50  # Left padding for the content
        content_y = 50  # Top padding for the content
        content_width = strip_width - 100  # Subtract left and right padding
        content_height = strip_height - 100  # Subtract top and bottom padding

        # Crop the content from the current strip
        content = self.image.crop((content_x, content_y, content_x + content_width, content_y + content_height))

        # Paste the content onto the new background
        new_bg.paste(content, (content_x, content_y))

        # Update the image and display it
        self.image = new_bg
        self.display_image()

    # ===== AI METHODS =====
    def remove_background(self):
        """Remove image background using AI"""
        if not hasattr(self, 'image') or not self.image:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        self.apply_action()
        image_np = np.array(self.image.convert("RGB"))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image_np).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
        mask = (output.argmax(0).cpu().numpy() == 15).astype(np.uint8) * 255

        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
        foreground = cv2.bitwise_and(image_np, image_np, mask=mask)
        rgba_image = np.dstack((foreground, mask))
        self.image = Image.fromarray(rgba_image, mode="RGBA")
        self.display_image()

    def replace_background(self):
        """Automatically remove background and replace it with selected image"""
        if not hasattr(self, 'image') or not self.image:
            messagebox.showwarning("Warning", "No image loaded")
            return

        # If image doesn't have alpha, remove background first
        image_np = np.array(self.image)
        if image_np.shape[2] < 4:
            self.remove_background()
            image_np = np.array(self.image)  # Refresh image_np after background removal

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return  # Cancelled

        try:
            # Load new background image and resize to match current image size
            new_bg = Image.open(file_path).convert("RGB")
            width, height = self.image.size
            new_bg = new_bg.resize((width, height))
            bg_np = np.array(new_bg)

            # Foreground and alpha mask
            fg_rgb = image_np[:, :, :3]
            alpha = image_np[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)

            # Broadcast alpha properly to match RGB shape
            composite = (fg_rgb * alpha + bg_np * (1 - alpha)).astype(np.uint8)

            self.image = Image.fromarray(composite)
            self.display_image()

        except Exception as e:
            messagebox.showerror("Error", f"Could not replace background:\n{e}")

    def blur_background(self):
        """Blur image background using AI"""
        if not hasattr(self, 'image') or not self.image:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        self.apply_action()
        image_np = np.array(self.image.convert("RGB"))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image_np).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
        mask = (output.argmax(0).cpu().numpy() == 15).astype(np.uint8) * 255

        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
        blurred = cv2.GaussianBlur(image_np, (21, 21), 0)
        foreground = cv2.bitwise_and(image_np, image_np, mask=mask)
        background = cv2.bitwise_and(blurred, blurred, mask=cv2.bitwise_not(mask))
        final = cv2.add(foreground, background)
        self.image = Image.fromarray(final)
        self.display_image()

    def image_to_bytes(self, image):
        """Convert image to bytes"""
        byte_arr = BytesIO()
        image.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

    def setup_chat_frame(self):
        """Setup the input and output widgets inside the chat frame"""
        self.question_entry = ctk.CTkEntry(
            self.chat_frame,
            width=300,
            placeholder_text="Ask about the image..."
        )
        self.question_entry.pack(pady=5)

        self.response_box = ctk.CTkTextbox(
            self.chat_frame,
            width=400,
            height=150
        )
        self.response_box.pack(pady=5)

        self.send_button = ctk.CTkButton(
        self.chat_frame,
        text="Send",
        command=self.ask_chatbot,
        fg_color="#2A3F54",
        hover_color="#3D5A80"
        )
        self.send_button.pack(pady=5)
    
    def ask_chatbot(self):
        """Query ChatGPT API about the image"""
        if not self.chat_frame.winfo_ismapped():
            # Show the chat frame and initialize widgets if not already done
            self.chat_frame.pack(pady=10)
            if not self.chat_widgets_initialized:
                self.setup_chat_frame()
                self.chat_widgets_initialized = True
            return

        if not hasattr(self, 'image') or not self.image:
            messagebox.showwarning("Warning", "No image loaded")
            return

        question = self.question_entry.get()
        if not question.strip():
            messagebox.showwarning("Warning", "Please enter a question")
            return

        # Convert the current image (self.image) to a temporary file path
        temp_image_path = "temp_image.jpg"
        self.image.save(temp_image_path)

        # Prepare the conversation payload
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ]

        # Call the ChatGPT API
        response = self.call_chat_gpt_api(conversation, temp_image_path)

        # Display the response
        self.response_box.delete("1.0", "end")
        self.response_box.insert("1.0", str(response))

        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    def ai_editing(self):
        with open(text_path, 'rb') as txt_file:
            response = requests.post(
                'https://api.deepai.org/api/image-editor',
                headers={'api-key': api_key},
                files={
                    'image': self.image,
                    'text': txt_file
                }
            )

        # Output the result
        if response.status_code == 200:
            result = response.json()
            print("Edited image URL:", result.get('output_url'))
        else:
            print("Error:", response.status_code)
            print(response.text)


    def call_chat_gpt_api(self, conversation, image_path):
        """Call the ChatGPT API with optional image input"""
        if image_path:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                conversation[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                })

        url = f"{self.basicUrl}/deployments/{self.modelName}/chat/completions?api-version={self.apiVersion}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.apiKey
        }
        payload = {"messages": conversation}

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            assistant_reply = data["choices"][0]["message"]["content"]
            return assistant_reply
        else:
            return f"Error: {response.status_code} - {response.text}"
        
    # ===== VIEW MANAGEMENT =====
    def toggle_load_image_buttons(self):
        """Switch to load image view"""
        self.highlight_active_view("load")
        if self.current_view == "load_image":
            return
        self.current_view = "Home"
        self.page_label.configure(text=f"Current: {self.current_view}")
        self.show_load_image_view()

    def toggle_photo_booth_buttons(self):
        """Switch to photobooth view"""
        self.highlight_active_view("photo")
        if self.current_view == "photo_booth":
            return
        self.current_view = "Photo Booth"
        self.page_label.configure(text=f"Current: {self.current_view}")
        self.show_photo_booth_view()

    def show_load_image_view(self):
        """Show load image interface"""
        self.load_btn.pack(side="left", padx=5)
        self.capture_btn.pack_forget()
        self.reset_btn.pack(side="left", padx=5)
        self.delete_btn.pack(side="left", padx=5)
        self.save_btn.pack()
        self.chat_frame.pack_forget()

    def show_photo_booth_view(self):
        """Show photobooth interface"""
        self.load_btn.pack_forget()
        self.capture_btn.pack(side="left", padx=5)
        self.reset_btn.pack(side="left", padx=5)
        self.delete_btn.pack(side="left", padx=5)
        self.save_btn.pack()
        self.chat_frame.pack_forget()

    # ===== THEME MANAGEMENT =====
    def toggle_theme(self):
        """Toggle between dark and light theme"""
        self.dark_mode = not self.dark_mode
        self.set_theme()

    def set_theme(self):
        """Apply current theme settings"""
        if self.dark_mode:
            ctk.set_appearance_mode("dark")
            base_color = "#1E2A38"
            hover_color = "#2A3F54"
            self.theme_switch.select()
        else:
            ctk.set_appearance_mode("light")
            base_color = "#2A3F54"
            hover_color = "#3D5A80"
            self.theme_switch.deselect()
        
        # Update all buttons
        self.update_button_colors(base_color, hover_color)

    def update_button_colors(self, base_color, hover_color):
        """Update button colors throughout app"""
        for widget in self.left_sidebar.winfo_children():
            if isinstance(widget, ctk.CTkButton):
                widget.configure(fg_color=base_color, hover_color=hover_color)
        
        for widget in self.top_menu.winfo_children():
            if isinstance(widget, ctk.CTkButton):
                widget.configure(fg_color=base_color, hover_color=hover_color)

    def auto_enhance(self):
        """Enhance the image using the Finegrain Image Enhancer"""
        if self.image:
            try:
                # Convert the image to bytes
                img_byte_arr = io.BytesIO()
                self.image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
    
                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file.write(img_byte_arr)
                    temp_file_path = temp_file.name
    
                # Initialize the client
                client = Client("finegrain/finegrain-image-enhancer")
                result = client.predict(
                    input_image=handle_file(temp_file_path),  # Pass the file path
                    prompt="Enhance",
                    negative_prompt="",
                    seed=42,
                    upscale_factor=1,  # Reduce upscale factor
                    controlnet_scale=0.5,  # Reduce controlnet scale
                    controlnet_decay=0.8,  # Reduce controlnet decay
                    condition_scale=4,  # Reduce condition scale
                    tile_width=64,  # Reduce tile width
                    tile_height=64,  # Reduce tile height
                    denoise_strength=0.2,  # Reduce denoise strength
                    num_inference_steps=10,  # Reduce inference steps
                    solver="DDIM",
                    api_name="/process"
                )
                print(result)
    
                # Load the enhanced image from the result
                enhanced_image_data = result[0]
                enhanced_image = Image.open(io.BytesIO(enhanced_image_data))
                self.image = enhanced_image
                self.display_image()
                messagebox.showinfo("Success", "Image enhanced successfully!")
    
                # Clean up the temporary file
                if temp_file_path:
                    os.remove(temp_file_path)
    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to enhance image: {e}")
        else:
            messagebox.showwarning("Warning", "No image loaded")


    def detect_faces_viola(self):
        """Detect faces using Viola-Jones"""
        if self.image:
            try:
                self.apply_action()

                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'data/haarcascade_frontalface_default.xml')
                img_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
                self.image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                self.display_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to detect faces: {e}")
        else:
            messagebox.showwarning("Warning", "No image loaded")

    def activate_eraser(self):
        """Activate eraser tool"""
        self.image_canvas.bind("<Button-1>", self.start_eraser)
        self.image_canvas.bind("<B1-Motion>", self.erase)
        self.image_canvas.bind("<ButtonRelease-1>", self.stop_eraser)
    
    def start_eraser(self, event):
        """Start erasing"""
        self.drawing = True
        scale_x = self.image.width / self.resized_image.width
        scale_y = self.image.height / self.resized_image.height
        self.last_x = int(event.x * scale_x)
        self.last_y = int(event.y * scale_y)
    
    def stop_eraser(self, event):
        """Stop erasing"""
        self.drawing = False
        self.last_x, self.last_y = None, None
    
    def erase(self, event):
        """Erase part of the image"""
        if self.drawing:
            self.apply_action()

            # Scale click coordinates to image coordinates
            scale_x = self.image.width / self.resized_image.width
            scale_y = self.image.height / self.resized_image.height
            x = int(event.x * scale_x)
            y = int(event.y * scale_y)
    
            # Draw a line between the last and current positions on the mask
            if self.last_x is not None and self.last_y is not None:
                cv2.line(self.mask, (self.last_x, self.last_y), (x, y), 255, 10)
    
            # Update the last position
            self.last_x = x
            self.last_y = y
    
            # Update the display
            img_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            img_cv[self.mask == 255] = (255, 255, 255)
            self.image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            self.display_image()
    
    def inpaint_image(self):
        """Inpaint the image"""
        if self.image:
            self.apply_action()

            img_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            inpainted_img = cv2.inpaint(img_cv, self.mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            self.image = Image.fromarray(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
            self.display_image()
        else:
            messagebox.showwarning("Warning", "No image loaded to inpaint!")

    def apply_action(self):
        """Save the current image state to the undo stack before applying an action."""
        if hasattr(self, 'image') and self.image:
            self.undo_stack.append(self.image.copy())  # Save the current image state
            self.redo_stack.clear()  # Clear the redo stack

    def undo(self):
        """Undo the last action."""
        if self.undo_stack:
            # Save the current state to the redo stack
            self.redo_stack.append(self.image.copy())
            # Restore the last state from the undo stack
            self.image = self.undo_stack.pop()
            self.display_image()
        else:
            messagebox.showwarning("Warning", "No actions to undo.")

    def redo(self):
        """Redo the last undone action."""
        if self.redo_stack:
            # Save the current state to the undo stack
            self.undo_stack.append(self.image.copy())
            # Restore the last state from the redo stack
            self.image = self.redo_stack.pop()
            self.display_image()
        else:
            messagebox.showwarning("Warning", "No actions to redo.")

if __name__ == "__main__":
    app = PhotoEditorApp()
    app.mainloop()