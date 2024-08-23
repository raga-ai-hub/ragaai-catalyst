import os
import requests
import logging
from .utils import response_checker
from .ragaai_catalyst import RagaAICatalyst

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

get_token = RagaAICatalyst.get_token


class Experiment:
    BASE_URL = None
    TIMEOUT = 10

    def __init__(
        self, project_name, experiment_name, experiment_description, dataset_name
    ):
        """
        Initializes the Experiment object with the provided project details and initializes various attributes.

        Parameters:
            project_name (str): The name of the project.
            experiment_name (str): The name of the experiment.
            experiment_description (str): The description of the experiment.
            dataset_name (str): The name of the dataset.

        Returns:
            None
        """
        Experiment.BASE_URL = (
            os.getenv("RAGAAI_CATALYST_BASE_URL")
            if os.getenv("RAGAAI_CATALYST_BASE_URL")
            else "https://llm-platform.dev4.ragaai.ai/api"
        )
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.dataset_name = dataset_name
        self.experiment_id = None
        self.job_id = None

        self.access_key = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
        self.secret_key = os.getenv("RAGAAI_CATALYST_SECRET_KEY")

        self.token = (
            os.getenv("RAGAAI_CATALYST_TOKEN")
            if os.getenv("RAGAAI_CATALYST_TOKEN") is not None
            else get_token()
        )
        self.metrics = []

    def list_experiments(self):
        """
        Retrieves a list of experiments associated with the current project.

        Returns:
            list: A list of experiment names.

        Raises:
            requests.exceptions.RequestException: If the request fails.

        """

        def make_request():
            headers = {
                "authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name
            }
            params = {
                "name": self.project_name,
            }
            response = requests.get(
                f"{Experiment.BASE_URL}/project",
                headers=headers,
                params=params,
                timeout=Experiment.TIMEOUT,
            )
            return response

        response = make_request()
        response_checker(response, "Experiment.list_experiments")
        if response.status_code == 401:
            get_token()  # Fetch a new token and set it in the environment
            response = make_request()  # Retry the request
        if response.status_code != 200:
            return {
                "status_code": response.status_code,
                "message": response.json(),
            }
        experiments = response.json()["data"]["experiments"]
        return [experiment["name"] for experiment in experiments]

    def add_metrics(self, metrics):
        """
        Adds metrics to the experiment and handles different status codes in the response.

        Parameters:
            metrics: The metrics to be added to the experiment. It can be a single metric or a list of metrics.

        Returns:
            If the status code is 200, returns a success message with the added metric names.
            If the status code is 401, retries the request, updates the job and experiment IDs, and returns the test response.
            If the status code is not 200 or 401, logs an error, and returns an error message with the response check.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
            "X-Project-Name": self.project_name,
        }

        if not isinstance(metrics, list):
            metrics = [metrics]
        else:
            metrics_list = metrics

        json_data = {
            "projectName": self.project_name,
            "datasetName": self.dataset_name,
            "experimentName": self.experiment_name,
            "metrics": metrics_list,
        }
        logger.debug(
            f"Preparing to add metrics for '{self.experiment_name}': {metrics}"
        )
        response = requests.post(
            f"{Experiment.BASE_URL}/v1/llm/experiment",
            headers=headers,
            json=json_data,
            timeout=Experiment.TIMEOUT,
        )

        status_code = response.status_code
        if status_code == 200:
            test_response = response.json()
            self.job_id = test_response.get("data").get("jobId")
            self.experiment_id = test_response.get("data").get("experiment").get("id")
            self.project_id = (
                test_response.get("data").get("experiment").get("projectId")
            )
            print(f"Metrics added successfully. Job ID: {self.job_id}")
            metric_names = [
                execution["metricName"]
                for execution in test_response["data"]["experiment"]["executions"]
            ]
            return f"Metrics {metric_names} added successfully"
        elif status_code == 401:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Name": self.project_name,
            }
            response = requests.post(
                f"{Experiment.BASE_URL}/v1/llm/experiment",
                headers=headers,
                json=json_data,
                timeout=Experiment.TIMEOUT,
            )
            status_code = response.status_code
            if status_code == 200:
                test_response = response.json()
                self.job_id = test_response.get("data").get("jobId")
                self.experiment_id = (
                    test_response.get("data").get("experiment").get("id")
                )
                self.project_id = (
                    test_response.get("data").get("experiment").get("projectId")
                )

                return test_response
            else:
                logger.error("Endpoint not responsive after retry attempts.")
                return response_checker(response, "Experiment.run_tests")
        else:
            logger.error(f"Failed to add metrics: HTTP {status_code}")
            return (
                "Error in running tests",
                response_checker(response, "Experiment.run_tests"),
            )

    def get_status(self):
        """
        Retrieves the status of a job based on the provided job ID.

        Returns the status of the job if the status code is 200, otherwise handles different status codes.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
            "X-Project-Name": self.project_name,
        }
        if self.job_id is None:
            logger.warning("Attempt to fetch status without a valid job ID.")
            return "Please run an experiment test first"
        json_data = {
            "jobId": self.job_id,
        }
        logger.debug(f"Fetching status for Job ID: {self.job_id}")
        response = requests.get(
            f"{Experiment.BASE_URL}/job/status",
            headers=headers,
            json=json_data,
            timeout=Experiment.TIMEOUT,
        )
        status_code = response_checker(response, "Experiment.get_status")
        if status_code == 200:
            # print(f"Status retrieved: Job ID {self.job_id} is active.")
            test_response = response.json()
            jobs = test_response["data"]["content"]
            for job in jobs:
                if job["id"] == self.job_id:
                    return job["status"]
        elif status_code == 401:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Name": self.project_name,
            }
            response = requests.post(
                f"{Experiment.BASE_URL}/job/status",
                headers=headers,
                json=json_data,
                timeout=Experiment.TIMEOUT,
            )
            status_code = response_checker(response, "Experiment.get_status")
            if status_code == 200:
                test_response = response.json()
                self.experiment_id = (
                    test_response.get("data").get("experiment").get("id")
                )
                return test_response
            else:
                logger.error("Endpoint not responsive after retry attempts.")
                return response_checker(response, "Experiment.get_status")
        else:
            return (
                "Error in running tests",
                response_checker(response, "Experiment.get_status"),
            )

    def get_results(self):
        """
        A function that retrieves results based on the experiment ID.
        It makes a POST request to the BASE_URL to fetch results using the provided JSON data.
        If the request is successful (status code 200), it returns the retrieved test response.
        If the status code is 401, it retries the request and returns the test response if successful.
        If the status is neither 200 nor 401, it logs an error and returns the response checker result.
        """

        if self.job_id is None:
            logger.warning("Results fetch attempted without prior job execution.")
            return "Please run an experiment test first"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
            "X-Project-Id": self.project_id
        }

        json_data = {
            "fields": [],
            "experimentId": self.experiment_id,
            "numRecords": 4,
            "projectId": self.project_id,
            "filterList": [],
        }

        status_json = self.get_status()
        if status_json == "Failed":
            return print("Job failed. No results to fetch.")
        elif status_json == "In Progress":
            return print("Job in progress. Please wait while the job completes.")
        elif status_json == "Completed":
            print("Job completed. fetching results")

        response = requests.post(
            f"{Experiment.BASE_URL}/v1/llm/docs",
            headers=headers,
            json=json_data,
            timeout=Experiment.TIMEOUT,
        )
        # status_code = response_checker(response, "Experiment.get_test_results")
        if response.status_code == 200:
            print(f"Results successfully retrieved.")
            test_response = response.json()
            return test_response
        elif response.status_code == 401:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Id": self.project_id
            }
            response = requests.post(
                f"{Experiment.BASE_URL}/v1/llm/docs",
                headers=headers,
                json=json_data,
                timeout=Experiment.TIMEOUT,
            )
            # status_code = response_checker(response, "Experiment.get_test_results")
            if response.status_code == 200:
                test_response = response.json()
                return test_response
            else:
                logger.error("Endpoint not responsive after retry attempts.")
                return response_checker(response, "Experiment.get_test_results")
        else:

            return (
                "Error in running tests",
                response_checker(response, "Experiment.get_test_results"),
            )
