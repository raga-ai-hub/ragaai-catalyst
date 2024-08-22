import tempfile
import json
import os
import uuid
import logging
import aiohttp
import asyncio
import threading
from queue import Queue, Empty
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


class FileSpanExporter(SpanExporter):
    def __init__(
            self,
            project_name=None,
            session_id=None,
            metadata=None,
            pipeline=None,
            raga_client=None,
    ):
        self.project_name = project_name
        self.session_id = session_id if session_id is not None else str(uuid.uuid4())
        self.metadata = metadata
        self.pipeline = pipeline
        self.raga_client = raga_client
        self.upload_timeout = 30  # Default timeout of 30 seconds
        self.successful_uploads = 0
        self.failed_uploads = 0

        # Set up directory for trace files
        self.dir_name = os.path.join(tempfile.gettempdir(), "raga_temp")
        self.backup_dir = os.path.join(self.dir_name, "backup")
        os.makedirs(self.backup_dir, exist_ok=True)

        # Create the sync file after dir_name is set
        self.sync_file = self._create_sync_file()
        self.file_lock = threading.Lock()
        self.upload_queue = Queue()
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()

    def _create_sync_file(self):
        file_name = f"trace_{self.project_name}_{uuid.uuid4()}.json"
        return os.path.join(self.dir_name, file_name)

    def export(self, spans):
        traces_list = [json.loads(span.to_json()) for span in spans]
        trace_id = traces_list[0]["context"]["trace_id"]

        export_data = {
            "project_name": self.project_name,
            "trace_id": trace_id,
            "session_id": self.session_id,
            "traces": traces_list,
            "metadata": self.metadata,
            "pipeline": self.pipeline,
        }

        try:
            with self.file_lock:
                with open(self.sync_file, "a", encoding="utf-8") as f:
                    json.dump(export_data, f)
                    f.write("\n")
            logger.debug(f"Exported trace data to {self.sync_file}")

            # Queue the file for upload
            self.upload_queue.put(self.sync_file)

        except Exception as e:
            logger.error(f"Failed to export trace data: {str(e)}", exc_info=True)
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    def _upload_worker(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()

        async def upload_task():
            async with aiohttp.ClientSession() as session:
                while True:
                    try:
                        file_to_upload = self.upload_queue.get_nowait()
                        await self._upload_traces(session, file_to_upload)
                        self.upload_queue.task_done()
                    except Empty:
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Error in upload worker: {str(e)}", exc_info=True)

        loop.run_until_complete(upload_task())

    async def _upload_traces(self, session, file_path):
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist, cannot upload: {file_path}")
            return

        if not os.getenv("RAGAAI_CATALYST_TOKEN"):
            logger.error("RAGAAI_CATALYST_TOKEN not found. Cannot upload traces.")
            self.failed_uploads += 1
            return

        try:
            upload_stat = await asyncio.wait_for(
                self.raga_client.check_and_upload_files(
                    session=session,
                    file_paths=[file_path],
                ),
                timeout=self.upload_timeout
            )
            logger.info(f"Upload status: {upload_stat}")

            if upload_stat:
                self.successful_uploads += 1
                logger.info(f"Successfully uploaded file: {file_path}")
                self._backup_file(file_path)
            else:
                self.failed_uploads += 1
                logger.error(f"Failed to upload file: {file_path}")
        except asyncio.TimeoutError:
            logger.error(f"Upload timed out after {self.upload_timeout} seconds")
            self.failed_uploads += 1
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}", exc_info=True)
            self.failed_uploads += 1

    def _backup_file(self, file_path):
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist, cannot backup: {file_path}")
            return

        backup_file = os.path.join(self.backup_dir, os.path.basename(file_path))
        try:
            os.rename(file_path, backup_file)
            logger.info(f"Moved file to backup: {backup_file}")
        except Exception as e:
            logger.error(f"Failed to backup file {file_path}: {str(e)}", exc_info=True)

    def get_upload_counts(self):
        return (self.successful_uploads, self.failed_uploads)

    def shutdown(self):
        # Wait for the upload queue to be empty
        self.upload_queue.join()
        logger.info("FileSpanExporter shutdown complete")

    def force_flush(self, timeout_millis: float = 30000) -> bool:
        # Implement force flush if needed
        return True