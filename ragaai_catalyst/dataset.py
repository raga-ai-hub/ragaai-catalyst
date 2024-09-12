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
            else "https://llm-platform.prod5.ragaai.ai/api"
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
                "X-Project-Name": self.project_name,
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

        ## check the validity of schema_mapping
        df = pd.read_csv(csv_path)
        keys = list(df.columns)
        values = self.get_csv_schema()['data']['schemaElements']
        print(type(values), values)
        for k in schema_mapping.keys():
            if k not in keys:
                raise ValueError(f'--{k}-- column is not present in csv column but present in schema_mapping. Plase provide the right schema_mapping.')
        for k in schema_mapping.values():
            if k not in values:
                raise ValueError(f'--{k}-- is not present in the schema_elements but present in schema_mapping. Plase provide the right schema_mapping.')
        

        #### get presigned URL
        def get_presignedUrl():
            headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }
            response = requests.get(
                f"{Dataset.BASE_URL}/v1/llm/presignedUrl/test-url",
                headers=headers,
                timeout=Dataset.TIMEOUT,
            )
            return response.json()
        
        presignedUrl = get_presignedUrl()
        if presignedUrl['success']:
            url = presignedUrl['data']['presignedUrl']
            filename = presignedUrl['data']['fileName']
            print('-- PresignedUrl fetched Succussfuly --')
            print('filename: ', filename)
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
        if put_csv_response.status_code != 201:
            raise ValueError('Unable to put csv to the presignedUrl')
        else:
            print('-- csv put to presignedUrl Succussfuly --')
        


        ## Upload csv to elastic
        def upload_csv_to_elastic(data):
            header = {
                'Authorization': f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                'X-Project-Name': self.project_name
            }
            response = requests.post(
                f"{Dataset.BASE_URL}/v1/llm/csv-dataset",
                headers=header,
                json=data,
                timeout=Dataset.TIMEOUT,
            )

            return response.json()
        
        data = {
            "datasetName": dataset_name,
            "fileName": filename,
            "schemaMapping": schema_mapping
        }
        print(data)

        upload_csv_response = upload_csv_to_elastic(data)
        print(type(upload_csv_response), upload_csv_response)
        if not upload_csv_response['success']:
            raise ValueError('Unable to upload csv')
        else:
            print(upload_csv_response['message'])
       
