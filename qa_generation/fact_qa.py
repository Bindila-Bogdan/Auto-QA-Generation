import langchain
import openai
from langchain.output_parsers import PydanticOutputParser
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser
from typing import List

from pydantic import BaseModel


class QAModel(BaseModel):
    question: str
    answer: str


class QAModelList(BaseModel):
    qa_list: List[QAModel]

fixer_prompt_template = '''
            Instructions:
            --------------
            Please correct the following output to match the expected format. The format should conform to:
            {format_instructions}
            --------------
            Completion:
            --------------
            {completion}
            --------------
            Error:
            --------------
            {error}
            --------------
            Based on the error described above and the format instructions, please correct the output. Only provide an answer that satisfies the constraints laid out in the format instructions.
            '''


class FactQAGenerator:
    def __init__(self, temperature=0.8, max_tokens=2048) -> None:
        """
        Initializes the QAGenerator with a given prompt and GPT-3.5 API configuration.
        :param prompt: The base prompt used for question generation.
        :param temperature: Controls randomness in the generation. Lower is more deterministic.
        :param max_tokens: The maximum number of tokens to generate in each request.
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize the OpenAI API and Langchain LLM
        self.llm = OpenAI(model_name="gpt-3.5-turbo-instruct",
                              temperature=temperature,
                              max_tokens=max_tokens,
                              openai_api_key=""
                         )
        self.llm_fixer = self.llm

    def generate_question_for_fact(self, prompt: PromptTemplate, fact) -> List[QAModel]:
        parsed_output = {'qa_list': []}
        
        parser = PydanticOutputParser(pydantic_object=QAModelList)

        prompt = prompt.format(format=parser.get_format_instructions(), 
                               fact=fact)
        raw_output = self.llm(prompt)
        try:
            # Parse the result with PydanticOutputParser
            parsed_output = parser.parse(raw_output)
        except Exception as e:
            fixer_prompt = langchain.PromptTemplate(
                template=fixer_prompt_template,
                input_variables=['format_instructions', 'completion', 'error']
                )

            new_parser = OutputFixingParser.from_llm(
                parser=parser,
                llm=self.llm_fixer, prompt=fixer_prompt.partial(format_instructions=parser.get_format_instructions())
            )
            parsed_output = new_parser.parse(raw_output)
        finally:
            return parsed_output.qa_list

