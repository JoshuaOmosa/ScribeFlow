# ScribeFlow: Local AI Transcription Engine

ScribeFlow is a Python-based utility designed to transform unstructured audio data into structured documentation. Built with a focus on local privacy and architectural decoupling, it leverages OpenAI's Whisper model to process consultations or meetings without relying on cloud-based APIs.

## Features
- **Privacy-First:** All processing is done locally on your machine.
- **Architectural Separation:** Decouples the transcription logic from the formatting layer, allowing for easy expansion to different output formats (JSON, Markdown, PDF).
- **Scalable Model Support:** Easily switch between `base`, `small`, `medium`, or `large` models depending on hardware capabilities.

## Tech Stack
- **Language:** Python 3.9+
- **AI Model:** OpenAI Whisper
- **Processing:** PyTorch (CPU/GPU)

## Installation & Setup

1. **System Requirement:** Ensure you have `ffmpeg` installed on your system.
   - *Windows:* `choco install ffmpeg`
   - *Mac:* `brew install ffmpeg`

2. **Clone the Repo:**
   ```bash
   git clone (https://github.com/JoshuaOmosa/ScribeFlow.git)
   cd ScribeFlow