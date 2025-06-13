import tkinter as tk
import threading
import numpy as np
import fitz
import pymupdf # PyMuPDF
import docx
from bs4 import BeautifulSoup
import zipfile
import warnings
import requests
import os
import wget
import re
import shutil
from kokoro import KPipeline
import soundfile as sf
import torch
pipeline = KPipeline(lang_code='a')
warnings.filterwarnings('ignore')

def openroute(question:str,context:str,model:str,api_key2:str,url2:str)->dict:
    """
    Function to call the model via OpenRouter
    Args:
        question: text question to ask of the paper
        context: the text of the paper or report
        model: the model to use
        api_key2: the OpenRouter API key
        url2: the URL of the OpenRouter "OpenAI" style Chat Completions API

    Returns: JSON Dictionary with text based summary inside.
    """
    headers = {
        "Authorization": f"Bearer {api_key2}",
        "Content-Type": "application/json"
    }
    # Define the request payload
    data = {
        "model": model,
        # "provider": {
        #     "only": ["Cerebras"]
        # },
        "messages": [
            {"role": "system", "content": "You are a helpful phd assistant."},
            {"role": "user", "content": f"Question:{question}\n\nContext:{context}"}
        ],
        'provider': {
            'ignore': [
                'GMICloud'
            ]
        }
    }
    try:
        # Send the POST request
        response = requests.post(url2, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error connecting to OpenRouter: {e}"}
    except ValueError:  # Includes JSONDecodeError
        return {"error": "Invalid JSON response from OpenRouter."}


def make_summary_report(text:str,model:str,api_key2:str,url2:str)->str:
    """
    Function to generate summary texts from a single paper, report, or publication document
    Arguments:
        text: the extracted text of the paper or document
        model: the model to use
        api_key2: the OpenRouter API key
        url2: the URL of the OpenRouter "OpenAI" style Chat Completions API
    """
    summary = ''
    questions = [
        "Summarize the main aim of the paper as well as the main specific questions asked in the paper. Keep each question to one sentence. Use the provided paper as context. Focus solely on the aims and questions.",
        "Summarize the statistical and analytical methods to a few sentences. Exactly name the statistical and analytica methods. Use the provided paper as context. Focus solely on statistical and analytical methods and keep the description of each method to two sentences.",
        "Summarize three main results from the provided paper. Results are defined as specific quantitative or qualitative results. If a qualitative result is returned, report the exact number from the provided context.",
        "Summarize three main conclusions from the paper. Conclusions are defined as broad context findings. Do not report specific quantitative or qualitative results. Each conclusion should not exceed three sentences.",
        "Based on the paper, list only the key recommendations and future directions for research. Do not include any specific results, data, numbers, statistics, or findings. Focus exclusively on qualitative guidance and proposed next steps and keep every recommendation and future direction to one sentence."]

    for i in range(len(questions)):
        try:
            response = openroute(question=questions[i],context=text,model=model,api_key2=api_key2,url2=url2)

            if "error" in response:
                print(f"API Error for question {i+1}: {response['error']}")
                summary = summary + "\n\n" + f"Error generating part of summary: {response['error']}"
                continue

            #print(str(response) +  "\n" +  "-"*50 + "\n")
            try:
                content = response['choices'][0]['message']['content']
                summary = f"{summary}\n\n{str(content)}"
            except (KeyError, IndexError) as e:
                print(f"Error extracting content from API response: {e}. Response: {response}")
                summary = summary + "\n\n" + "Error: Unexpected API response format."
                continue

            #print(str(summary) + "\n" + "-" * 50 + "\n")
            print(f"Step {i + 1} Completed.")
        except Exception as e: # Fallback for other unexpected errors
            print(f"Unexpected error in make_summary_report for question {i+1}: {e}")
            summary = summary + "\n\n" + "Error: An unexpected error occurred while generating part of the summary."
            continue

    return summary

def read_pdf(file_path:str)->tuple[str | None, str | None]:
    """Reads text from a PDF file."""
    try:
        with fitz.open(file_path) as doc:
            text = ''
            for page in doc:
                text += page.get_text("text", flags=pymupdf.TEXT_INHIBIT_SPACES)
            #print(f"Character Length of PDF: {str(len(text))}")
            text = text.encode("utf-8")
            return text, None
    except Exception as e:
        print(f"Failed to read PDF {file_path}: {e}")
        return None, f"Error reading PDF {file_path}: {e}"

def read_docx(file_path:str)->tuple[str | None, str | None]:
    """Reads text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text, None
    except Exception as e:
        print(f"Failed to read DOCX {file_path}: {e}")
        return None, f"Error reading DOCX {file_path}: {e}"


def read_html(file_path:str)->tuple[str | None, str | None]:
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
            return text, None
    except Exception as e:
        print(f"Failed to read HTML {file_path}: {e}")
        return None, f"Error reading HTML {file_path}: {e}"

# ---------------------------------------------------------------

def read_file_text(file_path:str)->tuple[str | None, str | None]:
    """Dispatches to the appropriate file reader based on the file extension."""
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None, f"File not found: {file_path}"

    file_ext = os.path.splitext(file_path)[-1].lower()
    if file_ext == '.pdf':
        return read_pdf(file_path)
    elif file_ext in ['.docx' , '.doc']:
        return read_docx(file_path)
    elif file_ext in ['.html', '.htm']:
        return read_html(file_path)
    else:
        print(f"Unsupported file format: {file_ext} for file {file_path}")
        return None, f"Unsupported file format: {file_ext}"

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

class DocumentProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Processor")
        self.working_label_text = tk.StringVar()
        self.working_label_text.set("")
        self.working = False

        # RIS File
        ris_file_label = tk.Label(root, text="RIS File Location:")
        ris_file_label.grid(row=0, column=0, padx=5, pady=5)
        self.ris_file_entry = tk.Entry(root, width=50)
        self.ris_file_entry.grid(row=0, column=1, padx=5, pady=5)
        self.ris_file_entry.insert("0",os.path.join(os.getcwd(),"Agrivoltaics_RIS_open.ris"))
        #self.ris_file_entry.insert("0", "/path/to/backupfile.ris")

        # Document Folder
        doc_folder_label = tk.Label(root, text="Document Folder:")
        doc_folder_label.grid(row=1, column=0, padx=5, pady=5)
        self.doc_folder_entry = tk.Entry(root, width=50)
        self.doc_folder_entry.grid(row=1, column=1, padx=5, pady=5)
        self.doc_folder_entry.insert("0",os.getcwd())
        #self.doc_folder_entry.insert("0", "/path/to/files/folder")

        # Output Folder
        out_folder_label = tk.Label(root, text="Output Files to Folder:")
        out_folder_label.grid(row=2, column=0, padx=5, pady=5)
        self.out_folder_entry = tk.Entry(root, width=50)
        self.out_folder_entry.grid(row=2, column=1, padx=5, pady=5)
        self.out_folder_entry.insert("0",os.path.join(os.getcwd(),"output/"))
        #self.out_folder_entry.insert("0", "/path/to/output/folder")

        # API Key
        api_key_label = tk.Label(root, text="OpenRouter API Key:")
        api_key_label.grid(row=3, column=0, padx=5, pady=5)
        self.api_key_entry = tk.Entry(root, width=50, show="*")
        self.api_key_entry.grid(row=3, column=1, padx=5, pady=5)
        # self.api_key_entry.insert("0","API_KEY_GOES_HERE")

        # Model
        model_label = tk.Label(root, text="Model:")
        model_label.grid(row=4, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar()
        self.model_var.set("meta-llama/llama-4-maverick")
        model_option = tk.OptionMenu(root, self.model_var,  "meta-llama/llama-4-maverick", "meta-llama/llama-3.1-70b-instruct")
        model_option.grid(row=4, column=1, padx=5, pady=5)

        # Voice
        voice_label = tk.Label(root, text="Voice:")
        voice_label.grid(row=5, column=0, padx=5, pady=5)
        self.voice_var = tk.StringVar()
        self.voice_var.set("female")
        voice_option = tk.OptionMenu(root, self.voice_var,  "female","male")
        voice_option.grid(row=5, column=1, padx=5, pady=5)

        # Go Button
        go_button = tk.Button(root, text="Start Processing Files", command=self.start_process)
        go_button.grid(row=6, column=1, padx=5, pady=5)

        # Output Zip Files
        output_zip_label = tk.Label(root, text="Output Zip Files:")
        output_zip_label.grid(row=7, column=0, padx=5, pady=5)
        self.output_zip1_label = tk.Label(root, text="")
        self.output_zip1_label.grid(row=7, column=1, padx=5, pady=5)
        self.output_zip2_label = tk.Label(root, text="")
        self.output_zip2_label.grid(row=8, column=1, padx=5, pady=5)

        # Working Label
        self.working_label = tk.Label(root, textvariable=self.working_label_text)
        self.working_label.grid(row=9, column=1, padx=5, pady=5)

    def animate_working_label(self):
        if self.working:
            current_text = self.working_label_text.get()
            if current_text == "":
                self.working_label_text.set("Working.")
            elif current_text == "Working.":
                self.working_label_text.set("Working..")
            elif current_text == "Working..":
                self.working_label_text.set("Working...")
            else:
                self.working_label_text.set("Working.")
            self.root.after(500, self.animate_working_label)

    def start_process(self):
        if not self.working:
            self.working = True
            self.working_label_text.set("Working.")
            self.animate_working_label()
            thread = threading.Thread(target=self.process)
            thread.start()

    def process(self):

        # get tk form variables
        ris_file2 = self.ris_file_entry.get()
        doc_folder2 = self.doc_folder_entry.get()
        out_folder2 = self.out_folder_entry.get()
        api_key2 = self.api_key_entry.get()
        model2 = self.model_var.get()
        voice2 = self.voice_var.get()

        # Placeholder for your actual processing function
        print("\n\n" + "-" * 50)
        print("Starting process with:")
        print(f"RIS File: {ris_file2}")
        print(f"Document Folder: {doc_folder2}")
        print(f"Output Folder: {out_folder2}")
        print(f"API Key: {api_key2}")
        print(f"Model: {model2}")
        print(f"Voice: {voice2}")
        print("-" * 50 + "\n" + "-" * 50)

        # set Variables
        chat_url2 = "https://openrouter.ai/api/v1/chat/completions"

        try:
            # open the RIS file
            if ris_file2.startswith("http"):
                try:
                    fdown = wget.download(ris_file2, out=os.getcwd())
                    if not os.path.isfile(fdown):
                        self.root.after(0, self.update_ui, "", "", "Error: Failed to download RIS file. Check URL and network.")
                        return
                except Exception as e:
                    print(f"Detailed download error: {e}")
                    self.root.after(0, self.update_ui, "", "", "Error downloading RIS file. See console for details.")
                    return
                with (open(fdown,'r',encoding="utf-8")) as f:
                    fileris = f.read()
            else:
                try:
                    if not os.path.isfile(ris_file2):
                        self.root.after(0, self.update_ui, "", "", f"Error: Local RIS file not found. Check path.")
                        return
                except Exception as e: # Should catch if os.path.isfile fails for some reason
                    print(f"Detailed local file access error: {e}")
                    self.root.after(0, self.update_ui, "", "", f"Error accessing local RIS file. See console for details.")
                    return
                with (open(ris_file2, "r",encoding="utf-8")) as f:
                    fileris = f.read()

            # # Process the RIS file text
            # test for data
            if not fileris: # Check if fileris is empty
                self.root.after(0, self.update_ui, "", "", "Error: RIS file is empty.")
                return

            # split it into items - we ate the end tag so don't forget to put that back if re-uploading
            items = fileris.split('\nER  - \n\n')
            if not items or (len(items) == 1 and items[0].strip() == "ER  -"): # Check if items is empty or only contains the end tag
                self.root.after(0, self.update_ui, "", "", "Error: No items found in RIS file.")
                return

            # split each line into a sub-item in those lists...
            for i in range(len(items)):
                items[i] = items[i].split('\n')

            # check the last item, if blank, pop it
            if items[-1] == ['']:
                items.pop()

            # proper next line pattern
            pattern = r"[0-9A-Z]{2}\s\s-\s"
            fixes = []

            # test em and fix em
            for i in range(len(items)):
                holder = []
                for j in range(len(items[i])):
                    if re.search(pattern, items[i][j]):
                        holder.append(items[i][j])
                    else:
                        cursor = len(holder) - 1
                        fixes.append(1)
                        holder[cursor] = str(holder[cursor]) + str(items[i][j])
                items[i] = holder
            print(f"Preprocessing Fixed  {len(fixes)} lines.")
            # end RIS file preprocessing

            # make a fresh file folder series
            folders = ['media','summaries','audio','staging']
            try:
                for folder in folders:
                    folder_path = os.path.join(os.getcwd(), folder)
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
                    os.mkdir(folder_path)
            except OSError as e:
                print(f"Detailed folder operation error: {e}")
                self.root.after(0, self.update_ui, "", "", f"Error with output folders. Check permissions or close files. See console.")
                return

            # check they exist
            for folder in folders:
                if not os.path.exists(os.path.join(os.getcwd(), folder)):
                    self.root.after(0, self.update_ui, "", "", f"Error: Failed to create folder '{folder}'. Check permissions.")
                    return

            # build human-readable counter
            y = 1

            # Notify UI: Starting processing
            self.root.after(0, self.update_ui, "", "", f"Starting processing for {len(items)} documents...")

            # iterate through the documents and build summaries
            for i_loop_idx, item_content in enumerate(items): # Use enumerate for index and content
                # set dummy vars
                pub_year = 0000
                titleofpaper = 'Unknown Title' # Default title
                authors, summaries = [], []

                # Attempt to find title early for UI update
                for line in item_content:
                    if line.startswith('TI  - '):
                        titleofpaper = line[6:]
                        break

                # Update UI with current document
                self.root.after(0, self.working_label_text.set, f"Processing doc {y}/{len(items)}: {titleofpaper[:30]}...")

                # iterate through the document items
                for j_loop_idx, line_content in enumerate(item_content): # Use enumerate for line content

                    # find the title (already found for UI, but ensure it's set for logic)
                    if line_content.startswith('TI  - '):
                        titleofpaper = line_content[6:]

                    # look for the authors and append them to the list
                    if line_content.startswith('AU  - '):
                        authors.append(line_content[6:])

                    # look for the year and add to a variable
                    if line_content.startswith('PY  - '):
                        pub_year = line_content[6:]

                    # look for L1 and L2 lines and process the files...
                    if (line_content.startswith('L1  - ')) or (line_content.startswith('L2  - ')):
                        loc = os.path.join(doc_folder2, line_content[6:])
                        print(f"Attempting to process file for: {titleofpaper} at {loc}")
                        try:
                            text_content, error_msg = read_file_text(loc)

                            if error_msg:
                                print(f"Skipping item {titleofpaper} due to file error: {error_msg}")
                                # Optionally, update GUI here or log to a file for user review
                                # For now, just printing to console as per instructions
                                continue

                            if text_content is None:
                                print(f"Skipping item {titleofpaper} as no text content was extracted (file might be empty or unreadable).")
                                continue

                            print(f"Successfully read file for {titleofpaper}, generating summary...")
                            summary = make_summary_report(text = text_content,
                                                      model = model2,
                                                      api_key2 = api_key2,
                                                      url2 = chat_url2)
                            summaries.append(summary)

                        except Exception as e:
                            print(f"Error during processing or summary generation for {titleofpaper}: {e}")
                            # Consider whether to inform the user via GUI here as well
                            continue

                # after going through all the items, collate and output the results...
                etal = '' if len(authors) == 1 else ' et al'

                # use counter to order file outputs
                if y > 99:
                    z = str(y)
                elif (y > 9) and (y < 100):
                    z = '0' + str(y)
                else:  # y < 10
                    z = '00' + str(y)

                sumname = str(z) + str(authors[0]) + etal + "-" + pub_year + "-" + titleofpaper + ".txt"
                if len(summaries) > 0:
                    sumfin = titleofpaper + " Authors: " + " ".join(authors) + summaries[0]
                    with open(os.path.join(str(os.getcwd()) , "summaries" , sumname), "w") as f:
                        f.write(sumfin)
                else:
                    print(f"No summaries generated for Number {str(y)}")

                # push the counter
                y += 1
                # end building the summaries loop

        except Exception as e:
            print(f"Detailed summary build section error: {e}")
            self.root.after(0, self.update_ui, "", "", f"Unexpected error during summary processing. See console.")

        # Notify UI before starting audio generation
        self.root.after(0, self.working_label_text.set, "Generating audio files...")
        print("Summaries Completed!\n\nGenerating Audio...")
        # # build audio files

        # set the voice for audio from form
        if voice2 =="female":
            voiceset = 'af_heart'

        elif voice2 =="male":
            voiceset = 'am_echo'

        else:
            voiceset = "af_bella"

        # loop through summaries
        try:
            summary_files = [f for f in os.listdir(os.path.join(os.getcwd(),'summaries')) if not f.startswith('.')]
            for idx, j_file_name in enumerate(summary_files): # Use enumerate for progress
                self.root.after(0, self.working_label_text.set, f"Audio for {j_file_name[:30]}... ({idx+1}/{len(summary_files)})")
                if not j_file_name.startswith('.'): # This check is redundant due to list comprehension above but kept for safety
                    print(j_file_name)
                    with open(os.path.join(os.getcwd(),'summaries',j_file_name), 'r', encoding='utf-8') as f:
                        text = f.read()

                    # clean the text a bit
                    text = text.replace('\n', ' ')
                    text = text.replace('\r', ' ')
                    text = text.replace('**', ' ')
                    text = text.replace('*', ' ')

                    # define the generator
                    generator = pipeline(text, voice=voiceset)

                    # create a variable to populate pieces into
                    fullaudio = []

                    # create audio clips
                    for i, (gs, ps, audio) in enumerate(generator):
                        print(f"Working on Clip {i} ...")
                        fullaudio.append(audio)

                    # concatenate the clips and dump the tmp var to get memory
                    audio2 = np.concatenate(fullaudio)
                    del fullaudio

                    # write the file
                    filename = str(j)[:-4]
                    filename = filename + ".wav"
                    fileloc = os.path.join(os.getcwd(),"audio",filename)
                    print(f"Writing {fileloc}")
                    sf.write(fileloc, audio2,24000)

        except Exception as e:
            print(f"Detailed audio build error: {e}")
            self.root.after(0, self.update_ui, "", "", f"Error during audio generation. See console.")

        try:
            # Notify UI before zipping
            self.root.after(0, self.working_label_text.set, "Zipping output files...")
            # # Pop zip locations
            output_zip1 = os.path.join(out_folder2,'audio.zip')
            output_zip2 = os.path.join(out_folder2, 'summaries.zip')

            zip_folder(os.path.join(os.getcwd(),'audio'), output_zip1)
            zip_folder(os.path.join(os.getcwd(), 'summaries'), output_zip2)

            print("The Program Successfully Completed. Close the GUI or hit CNTRL+C to stop the program.")

            self.root.after(0, self.update_ui, output_zip1, output_zip2, "Completed - You can Close the Program.")
        except Exception as e:
            print(f"Detailed zipping error: {e}")
            self.root.after(0, self.update_ui, "", "", f"Error during file zipping. See console.")

    def update_ui(self, output_zip1, output_zip2, status):
        self.working = False
        self.working_label_text.set(status)
        self.output_zip1_label.config(text=output_zip1, fg="blue")
        #self.output_zip1_label.config(cursor="hand2")
        #self.output_zip1_label.bind("<1>", lambda e: os.system(f'open "{output_zip1}"'))

        self.output_zip2_label.config(text=output_zip2, fg="blue")
        #self.output_zip2_label.config(cursor="hand2")
        #self.output_zip2_label.bind("<1>", lambda e: os.system(f'open "{output_zip2}"'))

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentProcessor(root)
    root.mainloop()
