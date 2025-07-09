# To install the required libraries, run these commands in your terminal:
# pip install youtube-transcript-api
# pip install gensim

# If you want to use the more advanced 'abstractive' summarization (generates new sentences),
# you will also need to install the 'transformers' library and a deep learning framework like PyTorch or TensorFlow:
# pip install transformers torch  # For PyTorch backend (recommended for abstractive summarization)
# OR
# pip install transformers tensorflow # For TensorFlow backend

import sys
import re
from youtube_transcript_api import YouTubeTranscriptApi
from gensim.summarization import summarize as gensim_summarize

# --- IMPORTANT: Uncomment the following lines if you want to use abstractive summarization ---
# from transformers import pipeline
# global_summarizer_pipeline = None # Initialize globally to load the model once

# --- YouTube Transcript Fetching Function ---
def get_youtube_transcript(video_url):
    """Fetches the transcript for a given YouTube video URL."""
    try:
        # Extract video ID from URL
        video_id = ""
        # Handle common YouTube URL formats
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            print("Invalid YouTube URL format. Please provide a URL with 'v=' or 'youtu.be/'.")
            return None

        print(f"Attempting to fetch transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([d['text'] for d in transcript_list])
        return transcript
    except Exception as e:
        print(f"Error fetching transcript for {video_url}: {e}")
        print("Possible reasons: Video has no English captions, captions are disabled, or incorrect video ID.")
        return None

# --- Summarization Functions ---

def summarize_extractive(text, ratio=0.2):
    """
    Summarizes text using Gensim's TextRank algorithm (extractive).
    ratio: float, percentage of the original text to keep (e.g., 0.2 means 20%)
    """
    if not text or len(text.strip()) < 50: # Gensim needs a reasonable amount of text to work
        return "Text is too short for extractive summarization or is empty."
    try:
        summary = gensim_summarize(text, ratio=ratio)
        return summary if summary else "Extractive summarization yielded an empty result. Text might be too repetitive or short for effective extraction."
    except ValueError as e:
        return f"Could not summarize (extractive): {e}. Text might be too short for the specified ratio."
    except Exception as e:
        return f"An unexpected error occurred during extractive summarization: {e}"

# --- IMPORTANT: Uncomment the entire 'summarize_abstractive' function if you want to use it ---
# def summarize_abstractive(text, max_length=150, min_length=40):
#     """
#     Summarizes text using a pre-trained abstractive summarization model (e.g., BART).
#     This function will download the model on first run if not already present.
#     """
#     global global_summarizer_pipeline
#     if global_summarizer_pipeline is None:
#         print("Initializing abstractive summarizer model (this may take a moment and download files)...")
#         try:
#             # 'sshleifer/distilbart-cnn-12-6' is a lighter, faster BART version
#             global_summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
#             print("Abstractive summarizer initialized.")
#         except Exception as e:
#             print(f"Error loading abstractive summarization model: {e}")
#             print("Please ensure 'transformers' and 'torch' (or 'tensorflow') are installed.")
#             return "Failed to load summarization model."

#     if not text or len(text.strip()) < 50:
#         return "Text is too short for abstractive summarization or is empty."

#     # Transformer models have an input token limit (e.g., 512 or 1024 tokens).
#     # For very long transcripts, simple truncation might lose important context.
#     # A more robust solution for very long texts would involve
#     # splitting the text into smaller segments, summarizing each, then summarizing those summaries.
#     # For demonstration, we'll use a basic character-based truncation.
#     max_model_input_chars = 4000 # Rough estimate for models like distilbart-cnn (~512 tokens)
#     original_length = len(text)
#     if original_length > max_model_input_chars:
#         print(f"Warning: Transcript ({original_length} chars) is very long for abstractive summarizer. Truncating to {max_model_input_chars} characters.")
#         text = text[:max_model_input_chars]

#     try:
#         summary_list = global_summarizer_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
#         return summary_list[0]['summary_text']
#     except Exception as e:
#         return f"An error occurred during abstractive summarization: {e}"

# --- Main Program Logic ---
def main():
    print("YouTube Transcript Summarizer")
    print("----------------------------")

    # Command-line argument parsing
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <youtube_video_url> [summary_type] [ratio/max_length]")
        print("  <youtube_video_url>: The full URL of the YouTube video.")
        print("  [summary_type]: Optional. 'extractive' (default) or 'abstractive'.")
        print("  [ratio/max_length]: Optional. Parameter for summarization.")
        print("    For 'extractive': [ratio] - e.g., 0.2 for 20% of original text length (float between 0.01 and 0.99).")
        print("    For 'abstractive': [max_length] - maximum length of the generated summary (integer).")
        print("\nExamples:")
        print("  Extractive (default ratio): python youtube_summarizer.py \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\"")
        print("  Extractive (15% summary): python youtube_summarizer.py \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" 0.15")
        print("  Abstractive (default length): python youtube_summarizer.py \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" abstractive")
        print("  Abstractive (max 100 words): python youtube_summarizer.py \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" abstractive 100")
        return

    video_url = sys.argv[1]
    summary_type = "extractive" # Default to extractive
    param = None # Ratio for extractive, max_length for abstractive

    # Parse optional arguments
    if len(sys.argv) > 2:
        arg2 = sys.argv[2].lower()
        if arg2 in ['extractive', 'abstractive']:
            summary_type = arg2
            if len(sys.argv) > 3:
                try:
                    if summary_type == 'extractive':
                        param = float(sys.argv[3])
                        if not (0.01 <= param <= 0.99):
                            print("Warning: Extractive ratio should be between 0.01 and 0.99. Using default 0.2.")
                            param = None
                    elif summary_type == 'abstractive':
                        param = int(sys.argv[3])
                        if not (20 <= param <= 500): # Reasonable limits for generated summary length
                            print("Warning: Abstractive max_length should be between 20 and 500. Using default 150.")
                            param = None
                except ValueError:
                    print(f"Warning: Invalid parameter for {summary_type} summarization. Using default.")
                    param = None
        else:
            # If second arg is not a type, assume it's the ratio for extractive
            try:
                param = float(arg2)
                if not (0.01 <= param <= 0.99):
                    print("Warning: Extractive ratio should be between 0.01 and 0.99. Using default 0.2.")
                    param = None
            except ValueError:
                print(f"Warning: Unrecognized summary type or invalid ratio '{arg2}'. Defaulting to extractive summary with default ratio.")

    print(f"Fetching transcript for: {video_url}")
    transcript = get_youtube_transcript(video_url)

    if transcript:
        print(f"\nTranscript fetched. Length: {len(transcript)} characters.")
        print(f"Summarizing using '{summary_type}' method...")

        summary = ""
        if summary_type == "extractive":
            ratio_val = param if param is not None else 0.2
            summary = summarize_extractive(transcript, ratio=ratio_val)
            print(f"\n--- Extractive Summary (ratio: {ratio_val}) ---")
        elif summary_type == "abstractive":
            # Check if the abstractive function is enabled (uncommented)
            if 'summarize_abstractive' in globals() and callable(globals()['summarize_abstractive']):
                max_len_val = param if param is not None else 150
                # min_length is often set to about half of max_length for good results
                min_len_val = int(max_len_val * 0.4)
                summary = globals()['summarize_abstractive'](transcript, max_length=max_len_val, min_length=min_len_val)
                print(f"\n--- Abstractive Summary (max_length: {max_len_val}, min_length: {min_len_val}) ---")
            else:
                print("\nAbstractive summarization is not enabled. Please uncomment the 'summarize_abstractive' function and its imports in the script, and install 'transformers' and 'torch'/'tensorflow'.")
                summary = "Abstractive summarization not available. Using default extractive."
                # Fallback to extractive if abstractive is requested but not enabled
                ratio_val = 0.2
                summary = summarize_extractive(transcript, ratio=ratio_val)
                print(f"\n--- Fallback to Extractive Summary (ratio: {ratio_val}) ---")

        print(summary)
    else:
        print("\nCould not generate summary as transcript could not be fetched.")

if __name__ == "__main__":
    main()
