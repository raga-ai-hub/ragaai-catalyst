import os
from groq import Groq
import google.generativeai as genai
import openai
import dotenv
import PyPDF2
import csv
import markdown
import pandas as pd
import json
from ragaai_catalyst import proxy_call
import ast

# dotenv.load_dotenv()

class SyntheticDataGeneration:
    """
    A class for generating synthetic data using various AI models and processing different document types.
    """

    def __init__(self):
        """
        Initialize the SyntheticDataGeneration class with API clients for Groq, Gemini, and OpenAI.
        """
        # self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_qna(self, text, question_type="simple", n=5,model_config=dict(),api_key=None):
        """
        Generate questions based on the given text using the specified model and provider.

        Args:
            text (str): The input text to generate questions from.
            question_type (str): The type of questions to generate ('simple', 'mcq', or 'complex').
            model (str): The specific model to use for generation.
            provider (str): The AI provider to use ('groq', 'gemini', or 'openai').
            n (int): The number of question/answer pairs to generate.

        Returns:
            pandas.DataFrame: A DataFrame containing the generated questions and answers.

        Raises:
            ValueError: If an invalid provider is specified.
        """
        provider = model_config.get("provider")
        model = model_config.get("model")
        api_base = model_config.get("api_base")



        system_message = self._get_system_message(question_type, n)
        if provider == "groq":
            if api_key is None and os.getenv("GROQ_API_KEY") is None:
                raise ValueError("API key must be provided for Groq.")
            self.groq_client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
            return self._generate_groq(text, system_message, model)
        elif provider == "gemini":
            genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
            if api_base is None:
                if api_key is None and os.getenv("GEMINI_API_KEY") is None:
                    raise ValueError("API key must be provided for Gemini.")
                genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
                return self._generate_gemini(text, system_message, model)
            else:
                messages=[
                {'role': 'user', 'content': system_message+text}
            ]
                a= proxy_call.api_completion(messages=messages ,model=model ,api_base=api_base)
                b= ast.literal_eval(a[0])
                return pd.DataFrame(b)
        elif provider == "openai":
            if api_key is None and os.getenv("OPENAI_API_KEY") is None:
                raise ValueError("API key must be provided for OpenAI.")
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
            return self._generate_openai(text, system_message, model)
        else:
            raise ValueError("Invalid provider. Choose 'groq', 'gemini', or 'openai'.")

    def _get_system_message(self, question_type, n):
        """
        Get the appropriate system message for the specified question type.

        Args:
            question_type (str): The type of questions to generate ('simple', 'mcq', or 'complex').
            n (int): The number of question/answer pairs to generate.

        Returns:
            str: The system message for the AI model.

        Raises:
            ValueError: If an invalid question type is specified.
        """
        if question_type == 'simple':
            return f'''Generate a set of {n} very simple questions answerable in a single phrase. 
                Also return the answers for the generated questions.
                Return the response in a list of object format. 
                Each object in list should have Question and corresponding answer.
                Do not return any extra strings. Return Generated text strictly in below format.  
                [{{"Question":"question,"Answer":"answer"}}]
            '''
        elif question_type == 'mcq':
            return f'''Generate a set of {n} questions with 4 probable answers from the given text. 
                The options should not be longer than a phrase. There should be only 1 correct answer.
                There should not be any ambiguity between correct and incorrect options.
                Return the response in a list of object format. 
                Each object in list should have Question and a list of options. 
                Do not return any extra strings. Return Generated text strictly in below format. 
                [{{"Question":"question","Options":[option1,option2,option3,option4]}}]
            '''
        elif question_type == 'complex':
            return f'''Can you generate a set of {n} complex questions answerable in long form from the below texts.
                Make sure the questions are important and provide new information to the user.
                Return the response in a list of object format. Enclose any quotes in single quote. 
                Do not use double quotes within questions or answers.
                Each object in list should have Question and corresponding answer.
                Do not return any extra strings. Return generated text strictly in below format.
                [{{"Question":"question","Answer":"answers"}}]
            '''
        else:
            raise ValueError("Invalid question type")

    def _generate_groq(self, text, system_message, model):
        """
        Generate questions using the Groq API.

        Args:
            text (str): The input text to generate questions from.
            system_message (str): The system message for the AI model.
            model (str): The specific Groq model to use.

        Returns:
            pandas.DataFrame: A DataFrame containing the generated questions and answers.
        """
        response = self.groq_client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': text}
            ]
        )
        return self._parse_response(response, provider="groq")

    def _generate_gemini(self, text, system_message, model):
        """
        Generate questions using the Gemini API.

        Args:
            text (str): The input text to generate questions from.
            system_message (str): The system message for the AI model.
            model (str): The specific Gemini model to use.

        Returns:
            pandas.DataFrame: A DataFrame containing the generated questions and answers.
        """
        model = genai.GenerativeModel(model)
        response = model.generate_content([system_message, text])
        return self._parse_response(response, provider="gemini")

    def _generate_openai(self, text, system_message, model):
        """
        Generate questions using the OpenAI API.

        Args:
            text (str): The input text to generate questions from.
            system_message (str): The system message for the AI model.
            model (str): The specific OpenAI model to use.

        Returns:
            pandas.DataFrame: A DataFrame containing the generated questions and answers.
        """
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ]
        )
        return self._parse_response(response, provider="openai")

    def _parse_response(self, response, provider):
        """
        Parse the response from the AI model and return it as a DataFrame.

        Args:
            response (str): The response from the AI model.
            provider (str): The AI provider used ('groq', 'gemini', or 'openai').
        Returns:
            pandas.DataFrame: The parsed response as a DataFrame.
        """
        if provider == "openai":
            data = response.choices[0].message.content
        elif provider == "gemini":
            data = response.candidates[0].content.parts[0].text
        elif provider == "groq":
            data = response.choices[0].message.content.replace('\n', '')
            list_start_index = data.find('[')  # Find the index of the first '['
            substring_data = data[list_start_index:] if list_start_index != -1 else data  # Slice from the list start
            data = substring_data

        else:
            raise ValueError("Invalid provider. Choose 'groq', 'gemini', or 'openai'.")
        try:
            json_data = json.loads(data)
            return pd.DataFrame(json_data)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a DataFrame with a single column
            return pd.DataFrame({'content': [data]})

    def process_document(self, input_data):
        """
        Process the input document and extract its content.

        Args:
            input_data (str): Either a file path or a string of text.

        Returns:
            str: The extracted text content from the document.

        Raises:
            ValueError: If the input is neither a valid file path nor a string of text.
        """
        if isinstance(input_data, str):
            if os.path.isfile(input_data):
                # If input_data is a file path
                _, file_extension = os.path.splitext(input_data)
                if file_extension.lower() == '.pdf':
                    return self._read_pdf(input_data)
                elif file_extension.lower() == '.txt':
                    return self._read_text(input_data)
                elif file_extension.lower() == '.md':
                    return self._read_markdown(input_data)
                elif file_extension.lower() == '.csv':
                    return self._read_csv(input_data)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")
            else:
                # If input_data is a string of text
                return input_data
        else:
            raise ValueError("Input must be either a file path or a string of text")

    def _read_pdf(self, file_path):
        """
        Read and extract text from a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text content from the PDF.
        """
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def _read_text(self, file_path):
        """
        Read the contents of a text file.

        Args:
            file_path (str): The path to the text file.

        Returns:
            str: The contents of the text file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _read_markdown(self, file_path):
        """
        Read and convert a Markdown file to HTML.

        Args:
            file_path (str): The path to the Markdown file.

        Returns:
            str: The HTML content converted from the Markdown file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
            html_content = markdown.markdown(md_content)
            return html_content

    def _read_csv(self, file_path):
        """
        Read and extract text from a CSV file.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            str: The extracted text content from the CSV, with each row joined and separated by newlines.
        """
        text = ""
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                text += " ".join(row) + "\n"
        return text

    def get_supported_qna(self):
        """
        Get a list of supported question types.

        Returns:
            list: A list of supported question types.
        """
        return ['simple', 'mcq', 'complex']

    def get_supported_providers(self):
        """
        Get a list of supported AI providers.

        Returns:
            list: A list of supported AI providers.
        """
        return ['groq', 'gemini', 'openai']

# Usage:
# from synthetic_data_generation import SyntheticDataGeneration
# synthetic_data_generation = SyntheticDataGeneration()
# text = synthetic_data_generation.process_document(input_data=text_file)
# result = synthetic_data_generation.generate_question(text)
# supported_question_types = synthetic_data_generation.get_supported_question_types()
# supported_providers = synthetic_data_generation.get_supported_providers()
