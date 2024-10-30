import os
from groq import Groq
import google.generativeai as genai
import openai
import PyPDF2
import csv
import markdown
import pandas as pd
import json
from litellm import completion
import internal_api_completion
# from ragaai_catalyst import proxy_call
import proxy_call
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
    def generate_qna(self, text, question_type="simple", n=5,model_config=dict(),api_key=None,**kwargs):
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
        # if "internal_llm_proxy" in kwargs.keys():
        api_base = model_config.get("api_base")

        system_message = self._get_system_message(question_type, n)
        if "internal_llm_proxy" not in kwargs.keys():
            if provider == "groq":
                if api_key is None and os.getenv("GROQ_API_KEY") is None:
                    raise ValueError("API key must be provided for Groq.")
                self.groq_client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
                return self._generate_llm_response(text, system_message,model_config,api_key)
            elif provider == "gemini":
                genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
                if api_base is None:
                    if api_key is None and os.getenv("GEMINI_API_KEY") is None:
                        raise ValueError("API key must be provided for Gemini.")
                    genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
                    return self._generate_llm_response(text, system_message,model_config,api_key)
                else:
                    messages=[
                    {'role': 'user', 'content': system_message+text}
                ]
                    a= proxy_call.api_completion(messages=messages,model= model,api_base=api_base)
                    b= ast.literal_eval(a[0])
                    return pd.DataFrame(b)
            elif provider == "openai":
                if api_key is None and os.getenv("OPENAI_API_KEY") is None:
                    raise ValueError("API key must be provided for OpenAI.")
                openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
                return self._generate_llm_response(text, system_message,model_config,api_key)
            else:
                raise ValueError("Invalid provider. Choose 'groq', 'gemini', or 'openai'.")
        else:
            messages=[
                {'role': 'user', 'content': system_message+text}]
            response= internal_api_completion.api_completion(messages=messages,model_config =model_config, kwargs=kwargs)
            json_data = json.loads(response.replace('\n', ''))
            return pd.DataFrame(json_data)



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

    def _generate_llm_response(self, text, system_message, model_config, api_key=None):
        """
        Generate questions using LiteLLM which supports multiple providers (OpenAI, Groq, Gemini, etc.).

        Args:
            text (str): The input text to generate questions from.
            system_message (str): The system message for the AI model.
            model_config (dict): Configuration dictionary containing model details.
                Required keys:
                - model: The model identifier (e.g., "gpt-4", "gemini-pro", "mixtral-8x7b-32768")
                Optional keys:
                - api_base: Custom API base URL if needed
                - max_tokens: Maximum tokens in response
                - temperature: Temperature for response generation
            api_key (str, optional): The API key for the model provider.

        Returns:
            pandas.DataFrame: A DataFrame containing the generated questions and answers.

        Raises:
            Exception: If there's an error in generating the response.
        """
        try:
            # Prepare the messages in the format expected by LiteLLM
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ]

            # Set up the completion parameters
            completion_params = {
                "model": model_config["model"],
                "messages": messages,
                "api_key": api_key
            }

            # Add optional parameters if they exist in model_config
            if "api_base" in model_config:
                completion_params["api_base"] = model_config["api_base"]
            if "max_tokens" in model_config:
                completion_params["max_tokens"] = model_config["max_tokens"]
            if "temperature" in model_config:
                completion_params["temperature"] = model_config["temperature"]

            # Make the API call using LiteLLM
            try:
                response = completion(**completion_params)
            except Exception as e:
                if any(error in str(e).lower() for error in ["invalid api key", "incorrect api key", "unauthorized", "authentication"]):
                    raise ValueError(f"Invalid API key provided for {model_config.get('provider', 'the specified')} provider")
                raise Exception(f"Error calling LLM API: {str(e)}")

            # Extract the content from the response
            content = response.choices[0].message.content

            # Clean the response if needed (remove any prefix before the JSON list)
            list_start_index = content.find('[')
            if list_start_index != -1:
                content = content[list_start_index:]

            try:
                # Parse the JSON response
                json_data = json.loads(content)
                return pd.DataFrame(json_data)
            except json.JSONDecodeError:
                # If JSON parsing fails, return a DataFrame with the raw content
                return pd.DataFrame({'content': [content]})

        except Exception as e:
            raise Exception(f"Error generating response with LiteLLM: {str(e)}")

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
        return ['gemini', 'openai']

# Usage:
# from synthetic_data_generation import SyntheticDataGeneration
# synthetic_data_generation = SyntheticDataGeneration()
# text = synthetic_data_generation.process_document(input_data=text_file)
# result = synthetic_data_generation.generate_question(text)
# supported_question_types = synthetic_data_generation.get_supported_question_types()
# supported_providers = synthetic_data_generation.get_supported_providers()
