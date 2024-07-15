import tempfile
import json
import os
import uuid
import logging
import aiohttp
import tempfile
import asyncio

from opentelemetry.sdk.trace.export import SpanExporter
from ..utils import get_unique_key
from .raga_exporter import RagaExporter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileSpanExporter(SpanExporter):
    def __init__(
        self,
        project_name=None,
        session_id=None,
        metadata=None,
        pipeline=None,
    ):
        """
        Initializes the FileSpanExporter.

        Args:
            project_name (str, optional): The name of the project. Defaults to None.
            session_id (str, optional): The session ID. Defaults to None.
            metadata (dict, optional): Metadata information. Defaults to None.
            pipeline (dict, optional): The pipeline configuration. Defaults to None.

        Returns:
            None
        """
        self.project_name = project_name
        self.session_id = session_id if session_id is not None else str(uuid.uuid4())
        self.metadata = metadata
        self.pipeline = pipeline
        self.sync_files = []
        # Set the temp directory to be output dir
        self.dir_name = tempfile.gettempdir()
        # self.raga_client = RagaExporter()

    def export(self, spans):
        """
        Export spans to a JSON file with additional metadata and pipeline information.

        Args:
            spans (list): List of spans to be exported.

        Returns:
            None
        """
        traces_list = [json.loads(span.to_json()) for span in spans]
        trace_id = traces_list[0]["context"]["trace_id"]

        self.filename = os.path.join(self.dir_name, trace_id + ".jsonl")

        # add the ids
        self.metadata["id"] = get_unique_key(self.metadata)
        self.pipeline["id"] = get_unique_key(self.pipeline)

        # add prompt id to each trace in trace_list
        for t in traces_list:
            t["prompt_id"] = get_unique_key(t)

        export_data = {
            "project_name": self.project_name,
            "trace_id": trace_id,
            "session_id": self.session_id,
            "traces": traces_list,
            "metadata": self.metadata,
            "pipeline": self.pipeline,
        }

        json_file_path = os.path.join(self.dir_name, trace_id + ".json")
        with open(self.filename, "a", encoding="utf-8") as f:
            logger.debug(f"Writing jsonl file: {self.filename}")
            f.write(json.dumps(export_data) + "\n")

        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as f:
                data = json.load(f)
                data.append(export_data)
            with open(json_file_path, "w") as f:
                logger.debug(f"Appending to json file: {json_file_path}")
                json.dump(data, f)
        else:
            with open(json_file_path, "w") as f:
                logger.debug(f"Writing json  file: {json_file_path}")
                json_data = [export_data]
                json.dump(json_data, f)
                self.sync_files.append(json_file_path)

        # asyncio.run(self.server_upload(json_file_path))

    def shutdown(self):
        pass

    # async def server_upload(self, file_name):
    #     async with aiohttp.ClientSession() as session:
    #         if os.getenv("RAGAAI_CATALYST_TOKEN"):
    #             logger.info("Token obtained successfully.")

    #             await self.raga_client.check_and_upload_files(
    #                 project_name=self.project_name, session=session, file_paths=[file_name]
    #             )
    #         else:
    #             logger.error("Failed to obtain token.")
