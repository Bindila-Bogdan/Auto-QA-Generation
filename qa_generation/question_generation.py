import openai
from langchain.llms.openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel
from typing import List


class QuestionGenerator:
    def __init__(self, temperature=0.5, max_tokens=1024) -> None:
        """
        Initializes the QAGenerator with a given prompt and GPT-3.5 API configuration.
        :param prompt: The base prompt used for question generation.
        :param temperature: Controls randomness in the generation.se Lower is more deterministic.
        :param max_tokens: The maximum number of tokens to generate in each request.
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize the OpenAI API and Langchain LLM
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    def generate_questions(self, prompt: PromptTemplate) -> [str]:
        """
        Generates a set of questions for each provided source document.
        :param prompt: The prompt to be used for question generation.
        :return: A dictionary with source document identifiers and their generated questions.
        """
        # parser = PydanticOutputParser(pydantic_object=QuestionsArrayModel)
        parser = CommaSeparatedListOutputParser()

        prompt = prompt.format(format=parser.get_format_instructions())
        print(prompt)
        # Use LangChain's LLM to generate output
        raw_output = self.llm(prompt)
        print(raw_output)
        # Parse the result with PydanticOutputParser
        parsed_output = parser.parse(raw_output)

        return parsed_output
