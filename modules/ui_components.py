#!/usr/bin/env python3
"""
UI Components Module
Shared UI components and utilities for the GGUF WebUI
"""

import gradio as gr
from typing import List, Dict, Any, Optional, Tuple
import time

class UIComponents:
    """Shared UI components and utilities"""
    
    def __init__(self):
        self.theme = self.create_custom_theme()
    
    def create_custom_theme(self):
        """Create a custom theme for the UI"""
        return gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="gray",
            font=[
                gr.themes.GoogleFont("Inter"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif"
            ]
        )
    
    def create_status_display(self, initial_status: str = "Ready") -> gr.Textbox:
        """Create a status display component"""
        return gr.Textbox(
            label="Status",
            value=initial_status,
            interactive=False,
            show_copy_button=False,
            container=True
        )
    
    def create_log_display(self, lines: int = 10) -> gr.Textbox:
        """Create a log display component"""
        return gr.Textbox(
            label="Activity Log",
            lines=lines,
            max_lines=20,
            interactive=False,
            show_copy_button=True,
            container=True,
            placeholder="Activity logs will appear here..."
        )
    
    def create_progress_bar(self) -> gr.Progress:
        """Create a progress bar component"""
        return gr.Progress(track_tqdm=True)
    
    def create_file_upload(self, file_types: List[str] = None, 
                          multiple: bool = False) -> gr.File:
        """Create a file upload component"""
        return gr.File(
            label="Upload Files",
            file_types=file_types,
            file_count="multiple" if multiple else "single",
            interactive=True
        )
    
    def create_model_selector(self, choices: List[str] = None) -> gr.Dropdown:
        """Create a model selector dropdown"""
        return gr.Dropdown(
            label="Select Model",
            choices=choices or [],
            value=None,
            interactive=True,
            info="Choose a model from the list"
        )
    
    def create_parameter_slider(self, label: str, minimum: float, maximum: float, 
                              value: float, step: float = 0.1) -> gr.Slider:
        """Create a parameter slider"""
        return gr.Slider(
            label=label,
            minimum=minimum,
            maximum=maximum,
            value=value,
            step=step,
            interactive=True
        )
    
    def create_chat_interface(self, height: int = 500) -> Tuple[gr.Chatbot, gr.Textbox, gr.Button]:
        """Create a chat interface"""
        chatbot = gr.Chatbot(
            label="Conversation",
            height=height,
            show_copy_button=True,
            bubble_full_width=False,
            show_share_button=False,

        )
        return chatbot, gr.Textbox(label="Type your message here..."), gr.Button("Send", variant="primary")
    
    
