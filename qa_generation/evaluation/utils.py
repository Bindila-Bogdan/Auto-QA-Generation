import csv
import os
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import inch
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import subprocess


def create_pdfs_from_file(filename, output_folder):
    """
    Create a PDF for each line in the file, handling multiline strings and preserving line breaks.
    Skip the first row, reduce font size and margins.

    Args:
    filename (str): The path to the CSV file.
    output_folder (str): The folder where the PDFs will be saved.
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.fontSize = 10  # Reducing the font size

    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        next(reader, None)  # Skip the header row
        for i, row in enumerate(reader, 1):  # Start counting from 1
            # Extracting the third field for the page content
            page_content = row[2] if len(row) > 2 else ""

            # Replace newline characters with HTML break tags
            page_content = page_content.replace('\n', '<br/>')

            pdf_path = os.path.join(output_folder, f"page_{i}.pdf")

            doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                                    leftMargin=0.5 * inch,
                                    rightMargin=0.5 * inch,
                                    topMargin=0.5 * inch,
                                    bottomMargin=0.5 * inch)
            Story = []

            # Adding the content as a Paragraph, which wraps text automatically
            content = Paragraph(page_content, style=normal_style)
            Story.append(content)
            Story.append(Spacer(1, 0.2 * inch))

            doc.build(Story)


def upload_files(api_url, directory):
    """
    Uploads all PDF files in a given directory to a specified REST API endpoint.

    Args:
    api_url (str): The URL of the API endpoint.
    directory (str): The directory containing the PDF files to upload.
    headers (dict): The headers to be used in the request.
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Filter for PDF files only
    files_to_upload = [f for f in os.listdir(directory) if
                       os.path.isfile(os.path.join(directory, f)) and f.lower().endswith('.pdf')]

    for filename in files_to_upload:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            # The key 'documents' is used here as that's what's indicated in the screenshot
            # Set the correct MIME type for PDF
            files = {'documents': (filename, f, 'application/pdf')}
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                print(f"Successfully uploaded {filename}")
            else:
                print(f"Failed to upload {filename}. Status code: {response.status_code} Response: {response.text}")


def get_response_and_extract_text(url, payload):
    try:
        # Sending the request
        response = requests.post(url, json=payload)

        # Check if the response status is OK
        response.raise_for_status()

        # Try to parse the response as JSON
        data = response.json()

        # Extracting 'text' from the response
        if 'responses' in data and data['responses']:
            text_response = data['responses'][0].get('text', 'No text found')
            return text_response
        else:
            return 'No responses found in the data'

    except requests.exceptions.HTTPError:
        # Return None in case of HTTP errors
        return None
    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection issues)
        print(f"Request error occurred: {req_err}")
        return f"Request error occurred: {req_err}"
    except ValueError as json_err:
        # Handle JSON decode error (e.g., if the response is not in JSON format)
        print(f"JSON decoding error occurred: {json_err}")
        return f"JSON decoding error occurred: {json_err}"
    except Exception as err:
        # Handle other unforeseen errors
        print(f"An error occurred: {err}")
        return f"An error occurred: {err}"


def send_question_to_api(question):
    url = 'http://localhost:5055/webhook'  # Replace with the actual URL of your API
    payload = {
        "sender_id": "14314",
        "next_action": "action_langchain",
        "tracker": {
            "sender_id": "14314",
            "slots": {
                "feedback": None  # Include only if your action uses this slot
            },
            "latest_message": {
                "intent": {"name": "user_intent_name", "confidence": 0.98},
                "entities": [],
                "text": question,
                "sender_id": "14314"
            },
            "active_loop": {},
            "latest_action_name": None
        }
    }

    text_from_response = get_response_and_extract_text(url, payload)
    return text_from_response


def run_command_in_directory(command, directory):
    """Run a shell command in a different directory."""
    try:
        # Changing the directory and running the command
        completed_process = subprocess.run(
            f"cd {directory} && {command}",
            check=True,  # Raises CalledProcessError if command returns non-zero exit status
            shell=True,  # Needed to run the command through the shell (to interpret &&, etc.)
            text=True,  # For Python 3.7+, ensures stdout and stderr are strings
            capture_output=True  # Captures stdout and stderr
        )
        return completed_process.stdout
    except subprocess.CalledProcessError as e:
        # If a non-zero exit status was returned
        print(f"Error: {e.returncode}, {e.stderr}")
        return None


# Function to calculate BLEU score for a single pair of response and answer
def calculate_bleu(response, answer):
    reference = [answer.split()]  # BLEU reference should be tokenized and put in a list
    candidate = response.split()  # BLEU candidate should be tokenized
    score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))  # Equal weights for 1-gram to 4-gram
    return score


# Function to calculate ROUGE scores for a single pair of response and answer
def calculate_rouge(response, answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(answer, response)
    return scores
