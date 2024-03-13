import openai
from langchain.llms.openai import OpenAI


def estimate_relevance(question, context):
    # Create the prompt for the GPT model
    prompt = f'''
    Context: {context}\nVraag: {question}\n
     Je hoeft niet te scoren hoe goed de vraag is.
     100 - interessante vraag, relevant voor het onderwerp, goed geformuleerd
     50 - niet zo interessant, maar toch enigszins relevant en ok geformuleerd
     0 - niet gerelateerd aan de context, niet correct geformuleerd.
     0 - de vraag is niet logisch zonder de context, omdat deze alleen op een te algemene manier naar de tekst verwijst met "dit" of "afspraak".
     Op een schaal van 1 tot 100, beoordeel
     hoe relevant de vraag is voor het contextantwoord met een getal:
    '''
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct",
                 openai_api_key=""
)  ##ChatOpenAI(model_name="gpt-3.5-turbo-instruct")
    # Query the GPT model
    response = llm(prompt)
    relevance_score = None
    # Extract the relevance score from the response
    # Here you might need more sophisticated parsing depending on the model's response
    try:
        relevance_score = float(response.strip())
    except ValueError:
        # Handle cases where the response is not a valid number
        try:
            relevance_score =  float(llm(prompt).strip())
        except:
            relevance_score = None

    return relevance_score