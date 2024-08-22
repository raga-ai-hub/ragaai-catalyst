import os
import datetime
import logging
import asyncio
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from .exporters.file_span_exporter import FileSpanExporter
from .exporters.raga_exporter import RagaExporter
from .instrumentators import (
    LangchainInstrumentor,
    OpenAIInstrumentor,
    LlamaIndexInstrumentor,
)
from .utils import get_unique_key

logger = logging.getLogger(__name__)

class Tracer:
    def __init__(
        self,
        project_name,
        tracer_type=None,
        pipeline=None,
        metadata=None,
        description=None,
    ):
        self.project_name = project_name
        self.tracer_type = tracer_type
        self.metadata = self._improve_metadata(metadata, tracer_type)
        self.pipeline = pipeline
        self.description = description

        self.raga_client = RagaExporter(project_name=self.project_name)
        self.filespanx = self._setup_exporter()
        self._tracer_provider = self._setup_provider()
        self._instrumentor = self._setup_instrumentor(tracer_type)
        self.is_instrumented = False

    def _setup_exporter(self):
        return FileSpanExporter(
            project_name=self.project_name,
            metadata=self.metadata,
            pipeline=self.pipeline,
            raga_client=self.raga_client,
        )

    def _setup_provider(self):
        tracer_provider = trace_sdk.TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(self.filespanx))
        return tracer_provider

    def _improve_metadata(self, metadata, tracer_type):
        if metadata is None:
            metadata = {}
        metadata.setdefault("log_source", f"{tracer_type}_tracer")
        metadata.setdefault("recorded_on", str(datetime.datetime.now()))
        return metadata

    def _add_unique_key(self, data, key_name):
        data[key_name] = get_unique_key(data)
        return data


    def _setup_instrumentor(self, tracer_type):
        instrumentors = {
            "langchain": LangchainInstrumentor,
            "openai": OpenAIInstrumentor,
            "llama_index": LlamaIndexInstrumentor,
        }
        if tracer_type not in instrumentors:
            raise ValueError(f"Invalid tracer type: {tracer_type}")
        return instrumentors[tracer_type]().get()

    @contextmanager
    def trace(self):
        """
        Synchronous context manager for tracing.
        Usage:
            with tracer.trace():
                # Your code here
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def start(self):
        """Start the tracer."""
        if not self.is_instrumented:
            self._setup_provider()
            self._instrumentor().instrument(tracer_provider=self._tracer_provider)
            self.is_instrumented = True
        logger.info(f"Tracer started for project: {self.project_name}")
        return self

    def stop(self):
        """Stop the tracer."""
        if not self.is_instrumented:
            logger.warning("Tracer was not started. No traces to stop.")
            return "No traces to stop"

        logger.info("Stopping tracer...")
        self._cleanup()
        self.filespanx.shutdown()
        logger.info("Tracer stopped successfully")
        return "Tracer stopped successfully"

    def _cleanup(self):
        """
        Cleans up the tracer by uninstrumenting the instrumentor and shutting down the tracer provider.
        """
        if self.is_instrumented:
            try:
                self._instrumentor().uninstrument()
                self._tracer_provider.shutdown()
                logger.info("Tracer provider shut down successfully")
            except Exception as e:
                logger.error(f"Error during tracer shutdown: {str(e)}")

        # Reset instrumentation flag
        self.is_instrumented = False

    def get_upload_status(self):
        """
        Get a simple summary of trace upload statuses.

        Returns:
            str: A summary of successful and failed trace uploads.
        """
        if not self.filespanx:
            return "No upload information available."

        successful, failed = self.filespanx.get_upload_counts()
        return f"Trace upload summary: {successful} successful, {failed} failed"