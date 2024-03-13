from flair.data import Sentence
from flair.models import SequenceTagger
from quantulum3 import parser
import re
import pyap

tagger = SequenceTagger.load('flair/ner-dutch-large')

def extract_entities(text):
    # Load the Dutch NER model

    # Prepare the text
    sentence = Sentence(text)

    # Predict the entities
    tagger.predict(sentence)

    # Extract entities
    entities = []
    for entity in sentence.get_spans('ner'):
        entities.append((entity.text, entity.tag))

    return entities


def extract_measurements(text):
    # Parse the text for quantities
    quantities = parser.parse(text)

    # Extract and format the measurements
    measurements = [f"{quantity.value} {quantity.unit.name}" for quantity in quantities]

    return measurements


def extract_addresses(text):
    ##addresses = pyap.parse(text, country='NL')
    pattern = r'\w+straat\s+\d+,\s+\d{4}\w{2}\s+\w+'
    adresses = re.findall(pattern, text)
    return adresses



def extract_opening_hours(text):
    pattern = r'\b(maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag)\s*tot\s*(maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag)?:?\s*\d{1,2}(?::\d{2})?\s*uur\s*tot\s*\d{1,2}(?::\d{2})?\s*uur\b'

    # Find all matches in the text
    matches = re.findall(pattern, text)

    return matches

import re

def extract_emails(text):
    # Regular expression for matching email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Find all matches in the text
    emails = re.findall(email_pattern, text)

    return emails