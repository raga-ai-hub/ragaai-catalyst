import os
import requests
import json
import re
from .ragaai_catalyst import RagaAICatalyst

class PromptManager:
    def __init__(self, project_name):
        """
        Initialize the PromptManager with a project name.

        Args:
            project_name (str): The name of the project.
        """
        self.project_name = project_name
        self.headers = {
                "Content-Type": "application/json",
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Name": self.project_name
            }
        self.base_url = f"{RagaAICatalyst.BASE_URL}/playground/prompt"
        self.timeout = 10

    def get_response(self, prompt_name, version=None):
        """
        Get the response for a specific prompt.

        Args:
            prompt_name (str): The name of the prompt.
            version (str, optional): The version of the prompt. Defaults to None.

        Returns:
            dict: The JSON response containing the prompt data.
        """
        response = requests.get(f"{RagaAICatalyst.BASE_URL}/playground/prompt/version/{prompt_name}",
                                headers=self.headers, timeout=10)
        return response.json()

    def list_prompts(self):
        """
        List all available prompts.

        Returns:
            list: A list of prompt names.
        """
        prompt = Prompt()
        prompt_list = prompt.list_prompts(self.base_url, self.headers, self.timeout)
        return prompt_list
    
    def get_prompt(self, prompt_name, version=None):
        """
        Get a specific prompt.

        Args:
            prompt_name (str): The name of the prompt.
            version (str, optional): The version of the prompt. Defaults to None.

        Returns:
            PromptObject: An object representing the prompt.
        """
        prompt = Prompt()
        prompt_object = prompt.get_prompt(self.base_url, self.headers, self.timeout, prompt_name, version)
        return prompt_object

    def list_prompt_versions(self, prompt_name):
        """
        List all versions of a specific prompt.

        Args:
            prompt_name (str): The name of the prompt.

        Returns:
            dict: A dictionary mapping version names to prompt texts.
        """
        prompt = Prompt()
        prompt_versions = prompt.list_prompt_versions(self.base_url, self.headers, self.timeout, prompt_name)
        return prompt_versions


class Prompt:
    def __init__(self):
        """
        Initialize the Prompt class.
        """
        pass

    def list_prompts(self, url, headers, timeout):
        """
        List all available prompts.

        Args:
            url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.

        Returns:
            list: A list of prompt names.
        """
        response = requests.get(url, headers=headers, timeout=timeout)
        prompt_list = [prompt["name"] for prompt in response.json()["data"]]                        
        return prompt_list

    def get_prompt_by_version(self, base_url, headers, timeout, prompt_name, version):
        """
        Get a specific version of a prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.
            version (str): The version of the prompt.

        Returns:
            str: The text of the prompt.
        """
        response = requests.get(f"{base_url}/version/{prompt_name}?version={version}",
                                headers=headers, timeout=timeout)
        prompt = ""
        for text_field in response.json()["data"]["docs"][0]["textFields"]:
            prompt += text_field['content']
        return prompt

    def get_prompt(self, base_url, headers, timeout, prompt_name, version=None):
        """
        Get a prompt, optionally specifying a version.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.
            version (str, optional): The version of the prompt. Defaults to None.

        Returns:
            PromptObject: An object representing the prompt.
        """
        if version:
            prompt_text = self.get_prompt_by_version(base_url, headers, timeout, prompt_name, version)
        else:
            response = requests.get(f"{base_url}/version/{prompt_name}",
                                headers=headers, timeout=timeout)
            prompt_text = ""
            for text_field in response.json()["data"]["docs"][0]["textFields"]:
                prompt_text += text_field['content']
        
        return PromptObject(prompt_text)

    def list_prompt_versions(self, base_url, headers, timeout, prompt_name):
        """
        List all versions of a specific prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.

        Returns:
            dict: A dictionary mapping version names to prompt texts.
        """
        response = requests.get(f"{base_url}/{prompt_name}/version",
                                headers=headers, timeout=timeout)
        version_names = [version["name"] for version in response.json()["data"]]
        prompt_versions = {}
        for version in version_names:
            prompt_versions[version] = self.get_prompt_by_version(base_url, headers, timeout, prompt_name, version)
        return prompt_versions

class PromptObject:
    def __init__(self, text):
        """
        Initialize a PromptObject with the given text.

        Args:
            text (str): The text of the prompt.
        """
        self.text = text
        self.variables = self._extract_variables()

    def _extract_variables(self):
        """
        Extract variables from the prompt text.

        Returns:
            list: A list of variable names found in the prompt text.
        """
        return [var.strip('{}') for var in self.text.split('{{')[1:]]
    
    def compile(self, **kwargs):
        """
        Compile the prompt by replacing variables with provided values.

        Args:
            **kwargs: Keyword arguments where keys are variable names and values are their replacements.

        Returns:
            str: The compiled prompt with variables replaced.
        """
        compiled_prompt = self.text
        for key, value in kwargs.items():
            compiled_prompt = compiled_prompt.replace(f"{{{{{key}}}}}", str(value))
        return compiled_prompt
    
    def get_variables(self):
        """
        Get all variables in the prompt text.

        Returns:
            list: A list of variable names found in the prompt text.
        """
        pattern = r'\{\{(.*?)\}\}'
        matches = re.findall(pattern, self.text)
        return [match.strip() for match in matches if '"' not in match]
