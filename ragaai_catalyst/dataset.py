import os
import requests
from .utils import response_checker
from typing import Union
import logging
from .ragaai_catalyst import RagaAICatalyst

logger = logging.getLogger(__name__)
get_token = RagaAICatalyst.get_token


class Dataset:
    BASE_URL = None
    TIMEOUT = 10

    def __init__(self, project_name):
        self.project_name = project_name
        Dataset.BASE_URL = (
            os.getenv("RAGAAI_CATALYST_BASE_URL")
            if os.getenv("RAGAAI_CATALYST_BASE_URL")
            else "https://llm-platform.dev4.ragaai.ai/api"
        )

    def list_datasets(self):
        """
        Retrieves a list of datasets for a given project.

        Returns:
            list: A list of dataset names.

        Raises:
            None.
        """

        def make_request():
            headers = {
                "accept": "application/json, text/plain, */*",
                "authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name
            }
            params = {
                "projectName": self.project_name,
            }
            response = requests.get(
                f"{Dataset.BASE_URL}/v1/llm/sub-datasets",
                headers=headers,
                params=params,
                timeout=Dataset.TIMEOUT,
            )
            return response

        response = make_request()
        response_checker(response, "Dataset.list_datasets")
        if response.status_code == 401:
            get_token()  # Fetch a new token and set it in the environment
            response = make_request()  # Retry the request
        if response.status_code != 200:
            return {
                "status_code": response.status_code,
                "message": response.json(),
            }
        datasets = response.json()["data"]["content"]
        sub_datasets = [dataset["name"] for dataset in datasets]
        return sub_datasets

    def create_dataset(self, dataset_name, filter_list):
        """
        Creates a new dataset with the given `dataset_name` and `filter_list`.

        Args:
            dataset_name (str): The name of the dataset to be created.
            filter_list (list): A list of filters to be applied to the dataset.

        Returns:
            str: A message indicating the success of the dataset creation and the name of the created dataset.

        Raises:
            None

        """

        def make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }
            json_data = {
                "projectName": self.project_name,
                "subDatasetName": dataset_name,
                "filterList": filter_list,
            }
            response = requests.post(
                f"{Dataset.BASE_URL}/v1/llm/sub-dataset",
                headers=headers,
                json=json_data,
                timeout=Dataset.TIMEOUT,
            )
            return response

        response = make_request()
        response_checker(response, "Dataset.create_dataset")
        if response.status_code == 401:
            get_token()  # Fetch a new token and set it in the environment
            response = make_request()  # Retry the request
        if response.status_code != 200:
            return response.json()["message"]
        message = response.json()["message"]
        return f"{message} {dataset_name}"
