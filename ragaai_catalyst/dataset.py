import os
import requests
from .utils import response_checker
from typing import Union
import logging
from .ragaai_catalyst import RagaAICatalyst
import pandas as pd

logger = logging.getLogger(__name__)
get_token = RagaAICatalyst.get_token


class Dataset:
    BASE_URL = None
    TIMEOUT = 30

    def __init__(self, project_name):
        self.project_name = project_name
        Dataset.BASE_URL = (
            os.getenv("RAGAAI_CATALYST_BASE_URL")
            if os.getenv("RAGAAI_CATALYST_BASE_URL")
            else "https://catalyst.raga.ai/api"
        )
        headers = {
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
        }
        try:
            response = requests.get(
                f"{Dataset.BASE_URL}/v2/llm/projects",
                headers=headers,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            logger.debug("Projects list retrieved successfully")

            self.project_id = [
                project["id"] for project in response.json()["data"]["content"] if project["name"]==project_name
            ][0]

        except:
            pass

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
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            json_data = {"size": 12, "page": "0", "projectId": str(self.project_id), "search": ""}
            response = requests.post(
                f"{Dataset.BASE_URL}/v2/llm/dataset",
                headers=headers,
                json=json_data,
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
        dataset_list = [dataset["name"] for dataset in datasets]
        
        return dataset_list

    def get_schema_mapping(self):
        return ["traceid", "prompt", "context", "response", "expected_response", "expected_context", "timestamp", "metadata", "pipeline", "cost", "feedBack", "latency", "sanitized_response", "system_prompt", "traceUri"]


    def create_from_trace(self, dataset_name, filter_list):
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

        def request_trace_creation():
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

        response = request_trace_creation()
        response_checker(response, "Dataset.create_dataset")
        if response.status_code == 401:
            get_token()  # Fetch a new token and set it in the environment
            response = request_trace_creation()  # Retry the request
        if response.status_code != 200:
            return response.json()["message"]
        message = response.json()["message"]
        return f"{message} {dataset_name}"
    


###################### CSV Upload APIs ###################

    def get_csv_schema(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": self.project_name,
        }
        response = requests.get(
                f"{Dataset.BASE_URL}/v1/llm/schema-elements",
                headers=headers,
                timeout=Dataset.TIMEOUT,
            )
        
        response_data = response.json()
        if not response_data['success']:
            raise ValueError('Unable to fetch Schema Elements for the CSV')
        
        # chema_elements = response['data']['schemaElements']
        return response_data


    def create_from_csv(self, csv_path, dataset_name, schema_mapping):
        #### get presigned URL
        def get_presignedUrl():
            headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            response = requests.get(
                f"{Dataset.BASE_URL}/v2/llm/dataset/csv/presigned-url",
                headers=headers,
                timeout=Dataset.TIMEOUT,
            )
            return response.json()
        
        presignedUrl = get_presignedUrl()
        if presignedUrl['success']:
            url = presignedUrl['data']['presignedUrl']
            filename = presignedUrl['data']['fileName']
            # print('-- PresignedUrl fetched Succussfuly --')
            # print('filename: ', filename)
        else:
            raise ValueError('Unable to fetch presignedUrl')
        


        #### put csv to presigned URL
        def put_csv_to_presignedUrl(url):
            headers = {
                'Content-Type': 'text/csv',
                'x-ms-blob-type': 'BlockBlob',
            }
            with open(csv_path, 'rb') as file:
                response = requests.put(
                    url,
                    headers=headers,
                    data=file,  
                    timeout=Dataset.TIMEOUT,
                )
            return response
        
        
        
        put_csv_response = put_csv_to_presignedUrl(url)
        if put_csv_response.status_code == 200:
            # print('-- csv put to presignedUrl Succussfuly --')
            pass
        else:
            raise ValueError('Unable to put csv to the presignedUrl')
        


        ## Upload csv to elastic
        def upload_csv_to_elastic(data):
            header = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id)
            }
            response = requests.post(
                f"{Dataset.BASE_URL}/v2/llm/dataset/csv",
                headers=header,
                json=data,
                timeout=Dataset.TIMEOUT,
            )

            return response.json()
        
        def generate_schema(mapping):
            result = {}
            for column, schema_element in mapping.items():
                result[column] = {"columnType": schema_element}
            return result
        
        schema_mapping = generate_schema(schema_mapping)
        
        data = {
            "projectId": str(self.project_id),
            "datasetName": dataset_name,
            "fileName": filename,
            "schemaMapping": schema_mapping,
            "opType": "insert",
            "description": ""
        }
        # print(data)

        upload_csv_response = upload_csv_to_elastic(data)
        if not upload_csv_response['success']:
            raise ValueError('Unable to upload csv')
        else:
            print(upload_csv_response['message'])
       
