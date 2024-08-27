import os
import logging
import requests
from typing import Dict, Optional, Union

logger = logging.getLogger("RagaAICatalyst")


class RagaAICatalyst:
    BASE_URL = None
    TIMEOUT = 10  # Default timeout in seconds

    def __init__(
        self,
        access_key,
        secret_key,
        api_keys: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initializes a new instance of the RagaAICatalyst class.

        Args:
            access_key (str): The access key for the RagaAICatalyst.
            secret_key (str): The secret key for the RagaAICatalyst.
            api_keys (Optional[Dict[str, str]]): A dictionary of API keys for different services. Defaults to None.

        Raises:
            ValueError: If the RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY environment variables are not set.

        Returns:
            None
        """
        if base_url:
            RagaAICatalyst.BASE_URL = base_url
            os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url

        if not access_key or not secret_key:
            logger.error(
                "RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY environment variables must be set"
            )
            raise ValueError(
                "RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY environment variables must be set"
            )

        self.access_key, self.secret_key = self._set_access_key_secret_key(
            access_key, secret_key
        )
        RagaAICatalyst.BASE_URL = (
            os.getenv("RAGAAI_CATALYST_BASE_URL")
            if os.getenv("RAGAAI_CATALYST_BASE_URL")
            else "https://llm-platform.dev4.ragaai.ai/api"
        )
        os.environ["RAGAAI_CATALYST_ACCESS_KEY"] = access_key
        os.environ["RAGAAI_CATALYST_SECRET_KEY"] = secret_key

        self.api_keys = api_keys or {}
        self.get_token()
        if self.api_keys:
            self._upload_keys()

        if base_url:
            RagaAICatalyst.BASE_URL = base_url
            os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url

    def _set_access_key_secret_key(self, access_key, secret_key):
        os.environ["RAGAAI_CATALYST_ACCESS_KEY"] = access_key
        os.environ["RAGAAI_CATALYST_SECRET_KEY"] = secret_key

        return access_key, secret_key

    def _upload_keys(self):
        """
        Uploads API keys to the server for the RagaAICatalyst.

        This function uploads the API keys stored in the `api_keys` attribute of the `RagaAICatalyst` object to the server. It sends a POST request to the server with the API keys in the request body. The request is authenticated using a bearer token obtained from the `RAGAAI_CATALYST_TOKEN` environment variable.

        Parameters:
            None

        Returns:
            None

        Raises:
            ValueError: If the `RAGAAI_CATALYST_ACCESS_KEY` or `RAGAAI_CATALYST_SECRET_KEY` environment variables are not set.

        Side Effects:
            - Sends a POST request to the server.
            - Prints "API keys uploaded successfully" if the request is successful.
            - Logs an error message if the request fails.

        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        }
        secrets = [
            {"type": service, "key": service, "value": key}
            for service, key in self.api_keys.items()
        ]
        json_data = {"secrets": secrets}
        response = requests.post(
            f"{RagaAICatalyst.BASE_URL}/v1/llm/secrets/upload",
            headers=headers,
            json=json_data,
            timeout=RagaAICatalyst.TIMEOUT,
        )
        if response.status_code == 200:
            print("API keys uploaded successfully")
        else:
            logger.error("Failed to upload API keys")

    def add_api_key(self, service: str, key: str):
        """Add or update an API key for a specific service."""
        self.api_keys[service] = key

    def get_api_key(self, service: str) -> Optional[str]:
        """Get the API key for a specific service."""
        return self.api_keys.get(service)

    @staticmethod
    def get_token() -> Union[str, None]:
        """
        Retrieves a token from the server using the provided access key and secret key.

        Returns:
            - A string representing the token if successful.
            - None if the access key or secret key is not set or if there is an error retrieving the token.

        Raises:
            - requests.exceptions.HTTPError: If there is an HTTP error while retrieving the token.
            - requests.exceptions.RequestException: If there is an error while retrieving the token.
            - ValueError: If there is a JSON decoding error.
            - Exception: If there is an unexpected error while retrieving the token.
        """
        access_key = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
        secret_key = os.getenv("RAGAAI_CATALYST_SECRET_KEY")

        if not access_key or not secret_key:
            logger.error(
                "RAGAAI_CATALYST_ACCESS_KEY or RAGAAI_CATALYST_SECRET_KEY is not set"
            )
            return None

        headers = {"Content-Type": "application/json"}
        json_data = {
            "accessKey": access_key,
            "secretKey": secret_key,
        }

        try:
            response = requests.post(
                f"{ RagaAICatalyst.BASE_URL}/token",
                headers=headers,
                json=json_data,
                timeout=RagaAICatalyst.TIMEOUT,
            )
            response.raise_for_status()

            token_response = response.json()

            if not token_response.get("success", False):
                logger.error(
                    "Token retrieval was not successful: %s",
                    token_response.get("message", "Unknown error"),
                )
                return None

            token = token_response.get("data", {}).get("token")
            if token:
                os.environ["RAGAAI_CATALYST_TOKEN"] = token
                print("Token(s) set successfully")
                return token
            else:
                logger.error("Token(s) not set")
                return None

        except requests.exceptions.HTTPError as http_err:
            logger.error(
                "HTTP error occurred while retrieving token: %s", str(http_err)
            )
            if response.status_code == 500:
                error_message = response.json().get("message", "Unknown server error")
                logger.error("Server error: %s", error_message)
            return None
        except requests.exceptions.RequestException as req_err:
            logger.error("Error occurred while retrieving token: %s", str(req_err))
            return None
        except ValueError as json_err:
            logger.error("JSON decoding error: %s", str(json_err))
            return None
        except Exception as e:
            logger.error("Unexpected error occurred while retrieving token: %s", str(e))
            return None

    def create_project(self, project_name, type="llm", description=""):
        """
        Creates a project with the given project_name, type, and description.

        Parameters:
            project_name (str): The name of the project to be created.
            type (str, optional): The type of the project. Defaults to "llm".
            description (str, optional): Description of the project. Defaults to "".

        Returns:
            str: A message indicating the success or failure of the project creation.
        """
        json_data = {"name": project_name, "type": type, "description": description}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        try:
            response = requests.post(
                f"{RagaAICatalyst.BASE_URL}/projects",
                headers=headers,
                json=json_data,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            print(
                f"Project Created Successfully with name {response.json()['data']['name']}"
            )
            return f'Project Created Successfully with name {response.json()["data"]["name"]}'

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                self.get_token()
                headers["Authorization"] = (
                    f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'
                )
                try:
                    response = requests.post(
                        f"{RagaAICatalyst.BASE_URL}/projects",
                        headers=headers,
                        json=json_data,
                        timeout=self.TIMEOUT,
                    )
                    response.raise_for_status()
                    print(
                        "Project Created Successfully with name %s after token refresh",
                        response.json()["data"]["name"],
                    )
                    return f'Project Created Successfully with name {response.json()["data"]["name"]}'
                except requests.exceptions.HTTPError as refresh_http_err:
                    logger.error(
                        "Failed to create project after token refresh: %s",
                        str(refresh_http_err),
                    )
                    return f"Failed to create project: {response.json().get('message', 'Authentication error after token refresh')}"
            else:
                logger.error("Failed to create project: %s", str(http_err))
                return f"Failed to create project: {response.json().get('message', 'Unknown error')}"
        except requests.exceptions.Timeout as timeout_err:
            logger.error(
                "Request timed out while creating project: %s", str(timeout_err)
            )
            return "Failed to create project: Request timed out"
        except Exception as general_err1:
            logger.error(
                "Unexpected error while creating project: %s", str(general_err1)
            )
            return "An unexpected error occurred while creating the project"

    def list_projects(self, num_projects=100):
        """
        Retrieves a list of projects with the specified number of projects.

        Parameters:
            num_projects (int, optional): Number of projects to retrieve. Defaults to 100.

        Returns:
            list: A list of project names retrieved successfully.
        """
        params = {
            "size": str(num_projects),
            "page": "0",
            "type": "llm",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        try:
            response = requests.get(
                f"{RagaAICatalyst.BASE_URL}/projects",
                params=params,
                headers=headers,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            logger.debug("Projects list retrieved successfully")
            # return [project["name"] for project in response.json()["data"]["content"]]
            project_list = [
                project["name"] for project in response.json()["data"]["content"]
            ]

            return project_list
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                self.get_token()
                headers["Authorization"] = (
                    f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'
                )
                try:
                    response = requests.get(
                        f"{RagaAICatalyst.BASE_URL}/projects",
                        params=params,
                        headers=headers,
                        timeout=self.TIMEOUT,
                    )
                    response.raise_for_status()
                    logger.debug(
                        "Projects list retrieved successfully after token refresh"
                    )
                    project_df = pd.DataFrame(
                        [
                            {"project": project["name"]}
                            for project in response.json()["data"]["content"]
                        ]
                    )
                    return project_df

                except requests.exceptions.HTTPError as refresh_http_err:
                    logger.error(
                        "Failed to list projects after token refresh: %s",
                        str(refresh_http_err),
                    )
                    return f"Failed to list projects: {response.json().get('message', 'Authentication error after token refresh')}"
            else:
                logger.error("Failed to list projects: %s", str(http_err))
                return f"Failed to list projects: {response.json().get('message', 'Unknown error')}"
        except requests.exceptions.Timeout as timeout_err:
            logger.error(
                "Request timed out while listing projects: %s", str(timeout_err)
            )
            return "Failed to list projects: Request timed out"
        except Exception as general_err2:
            logger.error(
                "Unexpected error while listing projects: %s", str(general_err2)
            )
            return "An unexpected error occurred while listing projects"

    def list_metrics(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        try:
            response = requests.get(
                f"{RagaAICatalyst.BASE_URL}/v1/llm/llm-metrics",
                headers=headers,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            logger.debug("Metrics list retrieved successfully")

            metrics = response.json()["data"]["metrics"]
            # For each dict in metric only return the keys: `name`, `category`
            sub_metrics = [metric["name"] for metric in metrics]
            return sub_metrics

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                self.get_token()
                headers["Authorization"] = (
                    f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'
                )
                try:
                    response = requests.get(
                        f"{RagaAICatalyst.BASE_URL}/v1/llm/llm-metrics",
                        headers=headers,
                        timeout=self.TIMEOUT,
                    )
                    response.raise_for_status()
                    logger.debug(
                        "Metrics list retrieved successfully after token refresh"
                    )
                    metrics = [
                        project["name"]
                        for project in response.json()["data"]["metrics"]
                    ]
                    # For each dict in metric only return the keys: `name`, `category`
                    sub_metrics = [
                        {
                            "name": metric["name"],
                            "category": metric["category"],
                        }
                        for metric in metrics
                    ]
                    return sub_metrics

                except requests.exceptions.HTTPError as refresh_http_err:
                    logger.error(
                        "Failed to list metrics after token refresh: %s",
                        str(refresh_http_err),
                    )
                    return f"Failed to list metrics: {response.json().get('message', 'Authentication error after token refresh')}"
            else:
                logger.error("Failed to list metrics: %s", str(http_err))
                return f"Failed to list metrics: {response.json().get('message', 'Unknown error')}"
