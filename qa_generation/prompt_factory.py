import random
import langchain

class PromptFactory:
    def __init__(self, prompt , roles = ['patiënt']):
        self.roles = roles
        self.prompt = langchain.PromptTemplate(
            template=prompt,
            input_variables=['role', 'n', 'document', 'format']
        )

    def generate_prompt(self, source_document: str, num_questions_per_doc: int = 1) -> langchain.PromptTemplate:
        """
        Generates a set of questions for each provided source document.
        :param source_document: The source document to generate questions for.
        :param num_questions_per_doc: Number of questions to generate per document.
        :return: A dictionary with source document identifiers and their generated questions.
        """
        return self.prompt.partial(
            role=random.choice(self.roles),
            n=num_questions_per_doc,
            document=source_document,
        )



