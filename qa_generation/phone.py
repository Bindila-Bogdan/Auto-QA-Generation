import re

def extract_and_normalize_phone_numbers(text):
    """
    Extracts and normalizes Dutch phone numbers from the provided text.

    Args:
    text (str): A string containing the text to be searched for phone numbers.

    Returns:
    list: A list of found and normalized phone numbers.
    """
    # Regex pattern for Dutch phone numbers
    pattern = r'\+31\s6\s\d{8}|\+31\s\d{2}\s\d{6,7}|\d{2}-\d{8}|06-\d{8}|\d{3}\s\d{3}\s\d{2}\s\d{2}'

    # Find all matches
    phone_numbers = re.findall(pattern, text)

    # Normalize each phone number found
    normalized_numbers = [normalize_phone_number(num) for num in phone_numbers]

    return normalized_numbers

def normalize_phone_number(phone_number):
    """
    Normalizes the Dutch phone number to a standard format for easy comparison.

    Args:
    phone_number (str): A Dutch phone number in various possible formats.

    Returns:
    str: A normalized phone number in a standard format.
    """
    # Removing spaces and hyphens
    phone_number = phone_number.replace(' ', '').replace('-', '')

    # Ensuring the country code is present
    if not phone_number.startswith('+31'):
        phone_number = '+31' + phone_number.lstrip('0')

    return phone_number
