# Automated Question-Answer Generation

## Project structure
The code developed for the Automated Question-Answer Generation for Evaluating RAG-based Chatbots project has the structure below.

- analyze_evaluation
    - analyze the results of the human evaluation
    - in the data directory are stored the following files of interest:
        - comments.txt - various comments written by annotators
        - questions_and_answers.csv - questions together with the answers and the source section
        - QA - evaluation (x_3) (Responses).xlsx - these 3 files contain the responses received to the form
- datascraping
    - extract, filter, and group the sections into pages
    - data directory contains the extracted sections that are stored in flyers_wo_outliers.json and the grouped sections into emulated pages that are stored in grouped_sections.json
- embeddings
    - generate embeddings at section or document level and analyze them visually
- folds_creation
    - analyze the topics, group them into folds, and retrieve the pages used during cross-validation
    - in the rag_pages and rag_pages_used_split are the pages used during cross-validation, being divided into train-test folds
    - in the folds_data directory are the train-test folds of sections created based on two sets of topics
- qa_generation
    - create question-answer pairs, filter them, and perform the cross-validation procedure
- topics_creation
    - run BERTopic with many sets of parameters to group sections into topics and then evaluate the quality of the found topics
- presentation.pdf - project presentation

> **Note**: The intermediary files such as the ones that store embeddings of sections, input topics, or the retrieved flyers have not been included because they are too large.
