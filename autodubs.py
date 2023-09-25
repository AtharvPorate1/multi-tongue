import streamlit as st
from pytube import YouTube
from pydub import AudioSegment
import pandas as pd
import whisper
import openai
import io
import os
# import xi
from elevenlabs import generate, set_api_key
import subprocess
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

api_key = st.secrets["YOUR_OPENAI_API_KEY"]
xi_api_key = st.secrets["xi_api_key"]

def shorten_audio(input_filename):
    output_filename = "cut_audio.mp4"
    audio = AudioSegment.from_file(input_filename)
    cut_audio = audio[:60 * 1000]
    cut_audio.export(output_filename, format="mp4")
    return output_filename
                    
def generate_translation(original_text, destination_language):
    prompt = (
    "Translate the following video transcript into " + destination_language +
    ". You will see the translation immediately after the prompt 'The translation is:'"
    "The transcript is as follows: " + original_text +
    "The translation is:"
)
    
    # Use your OpenAI API key here
    
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1500,
        n=1,
        # stop=None,
        # temperature=0.7,
        api_key=api_key
    )
    
    translation = response.choices[0].text
    return translation

def generate_dubs(text):
    filename = "output.mp3"
    
    set_api_key(xi_api_key)

    audio = generate(
        text=text,
        voice="Antoni",
        model='eleven_multilingual_v1'
    )

    audio_io = io.BytesIO(audio)  # You need to import io for this
    insert_audio = AudioSegment.from_file(audio_io, format='mp3')
    insert_audio.export(filename, format="mp3")
    
    return filename

def combine_video(video_filename, audio_filename):
    ffmpeg_extract_subclip(video_filename, 0, 60, targetname="cut_video.mp4")
    output_filename = "output.mp4"
    command = ["ffmpeg", "-y", "-i", "cut_video.mp4", "-i", audio_filename, "-c:v", "copy", "-c:a", "aac", output_filename]
    subprocess.run(command)
    return output_filename

st.title("AutoDubs 📺🎵")

link = st.text_input("Link to Youtube Video", key="link")
language = st.selectbox("Translate to", ("Hindi", "Marathi", "Tamil", "telugu", "Spanish"))

if st.button("Transcribe!"):
    print(f"downloading from link: {link}")
    
    model = whisper.load_model("base")
    yt = YouTube(link)
    
    if yt is not None:
        st.subheader(yt.title)
        st.image(yt.thumbnail_url)
        audio_name = st.caption("Downloading audio stream...")
        audio_streams = yt.streams.filter(only_audio=True)
        filename = audio_streams.first().download()
        
        if filename:
            audio_name.caption(filename)
            cut_audio = shorten_audio(filename)
            transcription = model.transcribe(cut_audio)
            print(transcription)
            
            if transcription:
                df = pd.DataFrame(transcription['segments'], columns=['start', 'end', 'text'])
                st.dataframe(df)
                
                dubbing_caption = st.caption("Generating translation...")
                translation = generate_translation(transcription['text'], language)
                dubbing_caption = st.caption("Begin dubbing...")
                dubs_audio = generate_dubs(translation)
                dubbing_caption.caption("Dubs generated! combining with the video...")
                
                video_streams = yt.streams.filter(only_video=True)
                video_filename = video_streams.first().download()
                if video_filename:
                    dubbing_caption.caption("Video downloaded! Combining the video and the dubs...")
                    output_filename = combine_video(video_filename, dubs_audio)

                    if os.path.exists(output_filename):
                        dubbing_caption.caption("Video successfully dubbed! Enjoy! 😀")
                        st.video(output_filename)

 
