import tkinter as tk
import threading
import time
from pydub import AudioSegment
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
    response = requests.post(url2, headers=headers, json=data)
    # Print the response
    return response.json()

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

    for i in questions:
        try:
            response = openroute(question=i,context=text,model=model,api_key2=api_key2,url2=url2)
            print(str(response) +  "\n" +  "-"*50 + "\n")
            summary = f"{summary}\n\n{str(response['choices'][0]['message']['content'])}"
            print(str(summary) + "\n" + "-" * 50 + "\n")
        except Exception as e:
            print(f"Summary Error: {e}")
            summary = summary + "\n\n" + "Error: " + str(e)
            continue

    return summary

def read_pdf(file_path:str)->[str,None]:
    """Reads text from a PDF file."""
    try:
        with fitz.open(file_path) as doc:
            text = ''
            for page in doc:
                text += page.get_text("text", flags=pymupdf.TEXT_INHIBIT_SPACES)
            #print(f"Character Length of PDF: {str(len(text))}")
            text = text.encode("utf-8")
            return text
    except Exception as e:
        print(f"Failed to read PDF: {e}")
        return None

def read_docx(file_path:str)->[str,None]:
    """Reads text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Failed to read DOCX: {e}")
        return None


def read_html(file_path:str)->[str,None]:
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

def read_file_text(file_path:str)->[str,None]:
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

def join_wavs_to_mp3(rootpath, wav_files, output_mp3):
    """
    Joins multiple WAV files into a single MP3 file.

    Args:
        wav_files: A list of paths to the WAV files to join.
        output_mp3: The path to save the output MP3 file.
        rootpath: The root directory of the WAV files.
    """
    combined = AudioSegment.empty()
    for wav_file in wav_files:
        if os.path.exists(os.path.join(rootpath,wav_file)):
            try:
                audio = AudioSegment.from_wav(os.path.join(rootpath,wav_file))
                combined += audio
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
        else:
            print(f"File not found: {wav_file}")

    combined.export(output_mp3, format="mp3")


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
        self.ris_file_entry.insert("0","/Users/toddgardiner/PAZSAGE/PAZSAGE/Agrivoltaics_RIS_open.ris")
        #self.ris_file_entry.insert("0", "/path/to/backupfile.ris")

        # Document Folder
        doc_folder_label = tk.Label(root, text="Document Folder:")
        doc_folder_label.grid(row=1, column=0, padx=5, pady=5)
        self.doc_folder_entry = tk.Entry(root, width=50)
        self.doc_folder_entry.grid(row=1, column=1, padx=5, pady=5)
        self.doc_folder_entry.insert("0","/Users/toddgardiner/PAZSAGE/PAZSAGE/")
        #self.doc_folder_entry.insert("0", "/path/to/files/folder")

        # Output Folder
        out_folder_label = tk.Label(root, text="Output Files to Folder:")
        out_folder_label.grid(row=2, column=0, padx=5, pady=5)
        self.out_folder_entry = tk.Entry(root, width=50)
        self.out_folder_entry.grid(row=2, column=1, padx=5, pady=5)
        self.out_folder_entry.insert("0","/Users/toddgardiner/PAZSAGE/output/")
        #self.out_folder_entry.insert("0", "/path/to/output/folder")

        # API Key
        api_key_label = tk.Label(root, text="OpenRouter API Key:")
        api_key_label.grid(row=3, column=0, padx=5, pady=5)
        self.api_key_entry = tk.Entry(root, width=50, show="*")
        self.api_key_entry.grid(row=3, column=1, padx=5, pady=5)
        self.api_key_entry.insert("0","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")

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
                fdown = wget.download(ris_file2, out=os.getcwd())
                assert (os.path.isfile(fdown)), "The system isn't currently downloading a .ris file."
                with (open(fdown,'r',encoding="utf-8")) as f:
                    fileris = f.read()
            else:
                assert(os.path.isfile(ris_file2)), "The system isn't currently accessing a local .ris file."
                with (open(ris_file2, "r",encoding="utf-8")) as f:
                    fileris = f.read()

            # # Process the RIS file text
            # test for data
            assert len(fileris) > 0, "RIS file is empty"

            # split it into items - we ate the end tag so don't forget to put that back if re-uploading
            items = fileris.split('\nER  - \n\n')
            assert len(items) > 0, "Items list is empty"

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
            for folder in folders:
                folder_path = os.path.join(os.getcwd(), folder)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                os.mkdir(folder_path)

            # check they exist
            for folder in folders:
                assert os.path.exists(os.path.join(os.getcwd(), folder)), f"Error at {folder} folder generation"

            # build human-readable counter
            y = 1

            # iterate through the documents
            for i in range(len(items)):
                # set dummy vars
                pub_year = 0000
                titleofpaper = ''
                authors, summaries = [], []

                # iterate through the document items
                for j in range(len(items[i])):

                    # find the title
                    if items[i][j].startswith('TI  - '):
                        titleofpaper = items[i][j][6:]

                    # look for the authors and append them to the list
                    if items[i][j].startswith('AU  - '):
                        authors.append(items[i][j][6:])

                    # look for the year and add to a variable
                    if items[i][j].startswith('PY  - '):
                        pub_year = items[i][j][6:]

                    # look for L1 and L2 lines and process the files...
                    if (items[i][j].startswith('L1  - ')) or (items[i][j].startswith('L2  - ')):
                        loc = os.path.join(doc_folder2, items[i][j][6:])
                        print(titleofpaper)
                        try:

                            text = read_file_text(loc)
                            if text is not None:
                                summary = make_summary_report(text = text,
                                                          model = model2,
                                                          api_key2 = api_key2,
                                                          url2 = chat_url2)
                                summaries.append(summary)

                        except Exception as e:
                            print(f"Items PyMUDF or Summary Generation Error: {e}")
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
                    print(f"No summaries generated for Number {str(y)} - {titleofpaper}")

                # push the counter
                y += 1
                # end building the summaries loop






            # Create dummy zip files for demonstration
            output_zip1 = "output1.zip"
            output_zip2 = "output2.zip"
            with zipfile.ZipFile(output_zip1, 'w') as zip1, zipfile.ZipFile(output_zip2, 'w') as zip2:
                zip1.writestr('dummy1.txt', 'This is a dummy file.')
                zip2.writestr('dummy2.txt', 'This is another dummy file.')

            time.sleep(4)
            self.root.after(0, self.update_ui, output_zip1, output_zip2, "Completed")
        except Exception as e:
            self.root.after(0, self.update_ui, "", "", f"Error: {e}")

    def update_ui(self, output_zip1, output_zip2, status):
        self.working = False
        self.working_label_text.set(status)
        self.output_zip1_label.config(text=output_zip1, fg="blue")
        self.output_zip1_label.config(cursor="hand2")
        self.output_zip1_label.bind("<1>", lambda e: os.system(f'open "{output_zip1}"'))

        self.output_zip2_label.config(text=output_zip2, fg="blue")
        self.output_zip2_label.config(cursor="hand2")
        self.output_zip2_label.bind("<1>", lambda e: os.system(f'open "{output_zip2}"'))

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentProcessor(root)
    root.mainloop()