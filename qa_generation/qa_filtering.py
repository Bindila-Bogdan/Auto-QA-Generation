import openai
from langchain_community.llms.openai import OpenAI


def estimate_relevance(question, context):
    # Create the prompt for the GPT model
    prompt = f"Context: {context}\nQuestion: {question}\nOn a scale from 1 to 100, rate how relevant the question is to the context:"
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")  ##ChatOpenAI(model_name="gpt-3.5-turbo-instruct")
    # Query the GPT model
    response = llm(prompt)
    # Extract the relevance score from the response
    # Here you might need more sophisticated parsing depending on the model's response
    try:
        relevance_score = float(response.strip())
    except ValueError:
        # Handle cases where the response is not a valid number
        relevance_score = None

    return relevance_score

# Example usage
# relevance_score = estimate_relevance("What is the capital of France?", "France is a country in Europe.", your_openai_api_key)
