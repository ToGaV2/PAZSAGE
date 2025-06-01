#!/usr/bin/env python


#import the libraries
# from kokoro import KPipeline
# import torch


#from IPython.display import display, Audio
from pydub import AudioSegment
import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup
import zipfile
import warnings
import requests
import os
warnings.filterwarnings('ignore')

import soundfile as sf
import json

import random

import re
import wget


# ---------------------------------------------------------------

def set_openrouterkey(keys):
    # Set the OpenRouter API key and openrouter front door
    # input a key and get a tuple of key and url...
    api_key = userdata.get('openrouter')
    url = "https://openrouter.ai/api/v1/chat/completions"
    return (api_key,url)

# ---------------------------------------------------------------

def openroute(question:str,context:str,model:str,api_key:str,url:str)->dict:
    '''
    Function to call the model via OpenRouter
    Args:
        question: text question to ask of the context
        context: the text of the paper or report
        model: the model to use
        api_key: the OpenRouter API key
        url: the URL of the OpenRouter "OpenAI" style Chat Completions API

    Returns: JSON Dictionary with text based summary inside.
    '''
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Define the request payload
    data = {
        #"model": qwen/qwen3-32b
        "model": model,
        # "provider": {
        #     "only": ["Cerebras"]
        # },
        "messages": [
            {"role": "system", "content": "You are a helpful phd assistant."},
            {"role": "user", "content": f"Question:{question}\n\nContext:{context}"}
        ]
    }
    # Send the POST request
    response = requests.post(url, headers=headers, json=data)
    # Print the response
    return response.json()

# ---------------------------------------------------------------

def make_summary_report(text:str)->str:
    '''
    Function to generate summary texts from a single paper, report, or publication document
    Arguments:
        text: the exctracted text of the paper or document
    '''
    summary = ''
    questions = []
    questions.append("Summarize the main aim of the paper as well as the main specific questions asked in the paper. Keep each question to one sentence. Use the provided paper as context. Focus solely on the aims and questions.")
    questions.append("Summarize the statistical and analytical methods to a few sentences. Exactly name the statistical and analytica methods. Use the provided paper as context. Focus solely on statistical and analytical methods and keep the description of each method to two sentences.")
    questions.append("Summarize three main results from the provided paper. Results are defined as specific quantitative or qualitative results. If a qualitative result is returned, report the exact number from the provided context.")
    questions.append("Summarize three main conclusions from the paper. Conclusions are defined as broad context findings. Do not report specific quantitative or qualitative results. Each conclusion should not exceed three sentences.")
    questions.append("Based on the paper, list only the key recommendations and future directions for research. Do not include any specific results, data, numbers, statistics, or findings. Focus exclusively on qualitative guidance and proposed next steps and keep every recommendation and future direction to one sentence.")

    for i in questions:
        try:
            response = openroute(i,text)
            summary = summary + "\n\n" + response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Summary Error: {e}")
            summary = summary + "\n\n" + "Error: " + str(e)
            continue
    return summary

# ---------------------------------------------------------------

def read_pdf(file_path:str)->str:
    """Reads text from a PDF file."""
    try:
        with fitz.open(file_path) as doc:
            text = ''
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        print(f"Failed to read PDF: {e}")
        return None

def read_docx(file_path:str)->str:
    """Reads text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Failed to read DOCX: {e}")
        return None

# ---------------------------------------------------------------

def read_html(file_path:str)->str:
    """Reads text from an HTML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
    except Exception as e:
        print(f"Failed to read HTML: {e}")
        return None

# ---------------------------------------------------------------

def read_file_text(file_path:str)->str:
    """Dispatches to the appropriate file reader based on the file extension."""
    if not os.path.isfile(file_path):
        print("File not found.")
        return None

    file_ext = os.path.splitext(file_path)[-1].lower()
    if file_ext == '.pdf':
        return read_pdf(file_path)
    elif file_ext in ['.docx' , '.doc']:
        return read_docx(file_path)
    elif file_ext in ['.html', '.htm']:
        return read_html(file_path)
    else:
        print("Unsupported file format.")
        return None

# ---------------------------------------------------------------

def join_wavs_to_mp3(root, wav_files, output_mp3):
    """
    Joins multiple WAV files into a single MP3 file.

    Args:
        wav_files: A list of paths to the WAV files to join.
        output_mp3: The path to save the output MP3 file.
    """
    combined = AudioSegment.empty()
    for wav_file in wav_files:
        if os.path.exists(os.path.join(root,wav_file)):
            try:
                audio = AudioSegment.from_wav(os.path.join(root,wav_file))
                combined += audio
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
        else:
            print(f"File not found: {wav_file}")

    combined.export(output_mp3, format="mp3")

# ---------------------------------------------------------------

def zip_folder(folder_path, output_path):
    """
    Zips the contents of a folder into a zip file.

    Args:
        folder_path (str): The path to the folder to be zipped.
        output_path (str): The path to the output zip file.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

# ---------------------------------------------------------------


if __name__ == "__main__":
    pass
    # run the program