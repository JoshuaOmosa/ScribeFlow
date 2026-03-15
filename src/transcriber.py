import whisper
import os
import warnings

# Suppress the FP16 warning for a cleaner terminal output
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

def process_consultation(file_path):
    print(f"--- Loading Model (Base) ---")
    # We specify device='cpu' to be explicit
    model = whisper.load_model("base", device="cpu")
    
    print(f"--- Transcribing: {file_path} ---")
    # fp16=False prevents the warning you saw earlier
    result = model.transcribe(file_path, fp16=False)
    
    formatted_note = f"# Consultation Note\n\n## Transcript\n{result['text']}\n"
    return formatted_note

if __name__ == "__main__":
    input_file = "data/input/sample.mp3" 
    
    if os.path.exists(input_file):
        try:
            note = process_consultation(input_file)
            with open("output_note.md", "w", encoding="utf-8") as f:
                f.write(note)
            print("\n✅ Success: Generated output_note.md")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
    else:
        print(f"❌ Error: File not found at {input_file}")