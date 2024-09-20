import os
import requests
import json
import re
from .ragaai_catalyst import RagaAICatalyst
import copy

class PromptManager:
    NUM_PROJECTS = 100
    TIMEOUT = 10

    def __init__(self, project_name):
        """
        Initialize the PromptManager with a project name.

        Args:
            project_name (str): The name of the project.

        Raises:
            requests.RequestException: If there's an error with the API request.
            ValueError: If the project is not found.
        """
        self.project_name = project_name
        self.headers = {
                "Content-Type": "application/json",
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Name": self.project_name
            }
        self.base_url = f"{RagaAICatalyst.BASE_URL}/playground/prompt"
        self.timeout = 10

        try:
            response = requests.get(
                f"{RagaAICatalyst.BASE_URL}/projects",
                params={
                    "size": str(self.NUM_PROJECTS),
                    "page": "0",
                    "type": "llm",
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                },
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching projects: {str(e)}")

        try:
            project_list = [
                project["name"] for project in response.json()["data"]["content"]
            ]
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing project list: {str(e)}")

        if self.project_name not in project_list:
            raise ValueError("Project not found. Please enter a valid project name")


    def list_prompts(self):
        """
        List all available prompts.

        Returns:
            list: A list of prompt names.

        Raises:
            requests.RequestException: If there's an error with the API request.
        """
        prompt = Prompt()
        try:
            prompt_list = prompt.list_prompts(self.base_url, self.headers, self.timeout)
            return prompt_list
        except requests.RequestException as e:
            raise requests.RequestException(f"Error listing prompts: {str(e)}")
    
    def get_prompt(self, prompt_name, version=None):
        """
        Get a specific prompt.

        Args:
            prompt_name (str): The name of the prompt.
            version (str, optional): The version of the prompt. Defaults to None.

        Returns:
            PromptObject: An object representing the prompt.

        Raises:
            ValueError: If the prompt or version is not found.
            requests.RequestException: If there's an error with the API request.
        """
        try:
            prompt_list = self.list_prompts()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt list: {str(e)}")

        if prompt_name not in prompt_list:
            raise ValueError("Prompt not found. Please enter a valid prompt name")

        try:
            prompt_versions = self.list_prompt_versions(prompt_name)
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt versions: {str(e)}")

        if version and version not in prompt_versions.keys():
            raise ValueError("Version not found. Please enter a valid version name")

        prompt = Prompt()
        try:
            prompt_object = prompt.get_prompt(self.base_url, self.headers, self.timeout, prompt_name, version)
            return prompt_object
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt: {str(e)}")

    def list_prompt_versions(self, prompt_name):
        """
        List all versions of a specific prompt.

        Args:
            prompt_name (str): The name of the prompt.

        Returns:
            dict: A dictionary mapping version names to prompt texts.

        Raises:
            ValueError: If the prompt is not found.
            requests.RequestException: If there's an error with the API request.
        """
        try:
            prompt_list = self.list_prompts()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt list: {str(e)}")

        if prompt_name not in prompt_list:
            raise ValueError("Prompt not found. Please enter a valid prompt name")
        
        prompt = Prompt()
        try:
            prompt_versions = prompt.list_prompt_versions(self.base_url, self.headers, self.timeout, prompt_name)
            return prompt_versions
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt versions: {str(e)}")


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

        Raises:
            requests.RequestException: If there's an error with the API request.
        """
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            prompt_list = [prompt["name"] for prompt in response.json()["data"]]                        
            return prompt_list
        except requests.RequestException as e:
            raise requests.RequestException(f"Error listing prompts: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing prompt list: {str(e)}")

    def get_response_by_version(self, base_url, headers, timeout, prompt_name, version):
        """
        Get a specific version of a prompt.

        Args:
            base_url (str): The base URL for the API.
            headers (dict): The headers to be used in the request.
            timeout (int): The timeout for the request.
            prompt_name (str): The name of the prompt.
            version (str): The version of the prompt."""
        try:
            response = requests.get(f"{base_url}/version/{prompt_name}?version={version}",
                                    headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt version: {str(e)}")
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Error parsing prompt version: {str(e)}")
        return response

    def get_response(self, base_url, headers, timeout, prompt_name):
        try:
            response = requests.get(f"{base_url}/version/{prompt_name}",
                                headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt version: {str(e)}")
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Error parsing prompt version: {str(e)}")
        return response

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

        Raises:
            requests.RequestException: If there's an error with the API request.
        """
        response = self.get_response_by_version(base_url, headers, timeout, prompt_name, version)
        prompt_text = response.json()["data"]["docs"][0]["textFields"]
        return prompt_text

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

        Raises:
            requests.RequestException: If there's an error with the API request.
        """
        if version:
            response = self.get_response_by_version(base_url, headers, timeout, prompt_name, version)
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            prompt_parameters = response.json()["data"]["docs"][0]["modelSpecs"]["parameters"]
            model = response.json()["data"]["docs"][0]["modelSpecs"]["model"]
        else:
            response = self.get_response(base_url, headers, timeout, prompt_name)
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            prompt_parameters = response.json()["data"]["docs"][0]["modelSpecs"]["parameters"]
            model = response.json()["data"]["docs"][0]["modelSpecs"]["model"]
        return PromptObject(prompt_text, prompt_parameters, model)


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

        Raises:
            requests.RequestException: If there's an error with the API request.
        """
        try:
            response = requests.get(f"{base_url}/{prompt_name}/version",
                                    headers=headers, timeout=timeout)
            response.raise_for_status()
            version_names = [version["name"] for version in response.json()["data"]]
            prompt_versions = {}
            for version in version_names:
                prompt_versions[version] = self.get_prompt_by_version(base_url, headers, timeout, prompt_name, version)
            return prompt_versions
        except requests.RequestException as e:
            raise requests.RequestException(f"Error listing prompt versions: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing prompt versions: {str(e)}")


class PromptObject:
    def __init__(self, text, parameters, model):
        """
        Initialize a PromptObject with the given text.

        Args:
            text (str): The text of the prompt.
            parameters (dict): The parameters of the prompt.
            model (str): The model of the prompt.
        """
        self.text = text
        self.parameters = parameters
        self.model = model

    def _extract_content(self):
        """
        Extract variables from the prompt text.

        Returns:
            list: A list of variable names found in the prompt text.
        """
        system_content = ""
        user_content = ""
        assistant_content = ""
        for item in self.text:
            if item.get("role") == "user":
                user_content = item.get("content", "")
            elif item.get("role") == "system":
                system_content = item.get("content", "")
            elif item.get("role") == "assistant":
                assistant_content = item.get("content", "")
        

        return system_content, user_content, assistant_content
    
    def compile(self, **kwargs):
        """
        Compile the prompt by replacing variables with provided values.

        Args:
            **kwargs: Keyword arguments where keys are variable names and values are their replacements.

        Returns:
            str: The compiled prompt with variables replaced.

        Raises:
            ValueError: If there are missing or extra variables, or if a value is not a string.
        """
        all_variables = self.get_variables()
        # print(all_variables)
        system_variables = all_variables["system"]
        user_variables = all_variables["user"]
        assistant_variables = all_variables["assistant"]

        required_variables = system_variables + user_variables + assistant_variables
        required_variables = list(set(required_variables))
        provided_variables = set(kwargs.keys())
        # print(required_variables)
        # print(provided_variables)

        missing_variables = [item for item in required_variables if item not in provided_variables]
        extra_variables = [item for item in provided_variables if item not in required_variables]
        # print(missing_variables)
        # print(extra_variables)
        if missing_variables:
            raise ValueError(f"Missing variable(s): {', '.join(missing_variables)}")
        if extra_variables:
            raise ValueError(f"Extra variable(s) provided: {', '.join(extra_variables)}")


        system_content, user_content, assistant_content = self._extract_content()
        # user_content = next(item["content"] for item in self.text if item["role"] == "user")

        for key, value in kwargs.items():
            if not isinstance(value, str):
                raise ValueError(f"Value for variable '{key}' must be a string, not {type(value).__name__}")
            if key in user_variables:                
                user_content = user_content.replace(f"{{{{{key}}}}}", value)
            if key in system_variables:
                system_content = system_content.replace(f"{{{{{key}}}}}", value)
            if key in assistant_variables:
                assistant_content = assistant_content.replace(f"{{{{{key}}}}}", value)

        updated_text = copy.deepcopy(self.text)

        for item in updated_text:
            if item.get('role') == 'system':
                item['content'] = system_content
            elif item['role'] == 'user':
                item['content'] = user_content
            elif item["role"] == "assistant":
                item["content"] = assistant_content

        return updated_text
    
    def get_variables(self):
        """
        Get all variables in the prompt text.

        Returns:
            list: A list of variable names found in the prompt text.
        """
        system_content, user_content, assistant_content = self._extract_content()
        pattern = r'\{\{(.*?)\}\}'

        matches_system = re.findall(pattern, system_content)
        matches_user = re.findall(pattern, user_content)
        matches_assistant = re.findall(pattern, assistant_content)

        system_variables = [match.strip() for match in matches_system if '"' not in match]
        user_variables = [match.strip() for match in matches_user if '"' not in match]
        assistant_variables = [match.strip() for match in matches_assistant if '"' not in match]

        return {"system": system_variables, "user": user_variables, "assistant": assistant_variables}
    
    # Function to convert value based on type
    def _convert_value(self, value, type_):
        if type_ == "float":
            return float(value)
        elif type_ == "int":
            return int(value)
        return value  # Default case, return as is

    def get_parameters(self):
        """
        Get all parameters in the prompt text.

        Returns:
            dict: A dictionary of parameters found in the prompt text.
        """
        parameters = {param["name"]: self._convert_value(param["value"], param["type"]) for param in self.parameters}
        parameters["model"] = self.model
        return parameters
    
    