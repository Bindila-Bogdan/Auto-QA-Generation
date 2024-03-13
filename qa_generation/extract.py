from enum import Enum
from phone import extract_phone_numbers
from ner import extract_entities, extract_measurements, extract_addresses, extract_opening_hours, extract_emails

# Define the EntityType Enum
class EntityType(Enum):
    PHONE = 'Phone'
    EMAIL = 'Email'
    OPENING_HOUR = 'Opening hour'
    ADDRESS = 'Address'
    MEASUREMENT = 'Measurement'

    
def extract_all_entities(text):
    entities = []

    # Extract phones
    for phone in extract_phone_numbers(text):
        entities.append({'entityType': EntityType.PHONE.value, 'value': phone})

    # Extract emails
    for email in extract_emails(text):
        entities.append({'entityType': EntityType.EMAIL.value, 'value': email})
    
    for entity_text, flair_type in extract_entities(text):
        entities.append({'entityType': flair_type, 'value': entity_text})

    #for address in extract_addresses(text):
    #    entities.append({'entityType': EntityType.ADDRESS.value, 'value': address})

    for hours in extract_opening_hours(text):
        entities.append({'entityType': EntityType.OPENING_HOUR.value, 'value': hours})

    #for measurement in extract_measurements(text):
    #    entities.append({'entityType': EntityType.MEASUREMENT.value, 'value': measurement})

    return entities

