# RagaAI Catalyst

RagaAI Catalyst is a powerful tool for managing and optimizing LLM projects. It provides functionalities for project management, trace recording, and experiment management, allowing you to fine-tune and evaluate your LLM applications effectively.

## Table of Contents

- [RagaAI Catalyst](#ragaai-catalyst)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Project Management](#project-management)
    - [Trace Management](#trace-management)
    - [Experiment Management](#experiment-management)
    - [Dataset Management](#dataset-management)
    - [Prompt Management](#prompt-management)

## Installation

To install RagaAI Catalyst, you can use pip:

```bash
pip install ragaai-catalyst
```

## Configuration

Before using RagaAI Catalyst, you need to set up your credentials. You can do this by setting environment variables or passing them directly to the `RagaAICatalyst` class:

```python
from ragaai_catalyst import RagaAICatalyst

catalyst = RagaAICatalyst(
    access_key="YOUR_ACCESS_KEY",
    secret_key="YOUR_SECRET_KEY",
    base_url="BASE_URL"
)
```
**Note**: Authetication to RagaAICatalyst is necessary to perform any operations below 


## Usage

### Project Management

Create and manage projects using RagaAI Catalyst:

```python
# Create a project
project = catalyst.create_project(
    project_name="Test-RAG-App-1",
    description="Description of the project"
)

# List projects
projects = catalyst.list_projects()
print(projects)
```

### Trace Management

Record and analyze traces of your RAG application:

```python
from ragaai_catalyst import Tracer

# Start a trace recording
tracer = Tracer(
    project_name="Test-RAG-App-1",
    metadata={"key1": "value1", "key2": "value2"},
    tracer_type="langchain",
    pipeline={
        "llm_model": "gpt-3.5-turbo",
        "vector_store": "faiss",
        "embed_model": "text-embedding-ada-002",
    }
).start()

# Your code here

# Stop the trace recording
tracer.stop()

# Alternatively, use a context manager
with tracer.trace():
    # Your code here
```

### Experiment Management

Create and manage experiments to evaluate your RAG application:

```python
from ragaai_catalyst import Experiment

# Create an experiment
experiment_manager = Experiment(
    project_name="Test-RAG-App-1",
    experiment_name="Exp-01",
    experiment_description="Experiment Description",
    dataset_name="Dataset Created from UI",
)

# Add metrics to the experiment
experiment_manager.add_metrics(
    metrics=[
      {"name": "hallucination", "config": {"model": "gpt-4o", "provider":"OpenAI"}}
    ]
)

# Add multiple metrics
experiment_manager.add_metrics(
    metrics=[
        {"name": "hallucination", "config": {"model": "gpt-4o", "provider":"OpenAI"}},
        {"name": "hallucination", "config": {"model": "gpt-4", "provider":"OpenAI"}},
        {"name": "hallucination", "config": {"model": "gpt-3.5-turbo", "provider":"OpenAI"}}
    ]
)

# Get the status of the experiment
status = experiment_manager.get_status()
print("Experiment Status:", status)

# Get the results of the experiment
results = experiment_manager.get_results()
print("Experiment Results:", results)
```



## Dataset Management
Manage datasets efficiently for your projects:

```py
from ragaai_catalyst import Dataset

# Initialize Dataset management for a specific project
dataset_manager = Dataset(project_name="project_name")

# List existing datasets
datasets = dataset_manager.list_datasets()
print("Existing Datasets:", datasets)

# Create a dataset from trace
dataset_manager.create_from_trace(
    dataset_name='Test-dataset-1',
    filter_list=[
        {"name": "llm_model", "values": ["gpt-3.5-turbo", "gpt-4"]},
        {"name": "prompt_length", "lte": 27, "gte": 23}
    ]
)

# Create a dataset from CSV
dataset_manager.create_from_csv(
    csv_path='path/to/your.csv',
    dataset_name='MyDataset',
    schema_mapping={'column1': 'schema_element1', 'column2': 'schema_element2'}
)
```

For more detailed information on Dataset Management, including CSV schema handling and advanced usage, please refer to the [Dataset Management documentation](docs/dataset_management.md).

## Prompt Management

Manage and use prompts efficiently in your projects:

```py
from ragaai_catalyst.prompt_manager import PromptManager

# Initialize PromptManager
prompt_manager = PromptManager("your-project-name")

# List available prompts
prompts = prompt_manager.list_prompts()
print("Available prompts:", prompts)

# Get a specific prompt
prompt_name = "your_prompt_name"
prompt = prompt_manager.get_prompt(prompt_name)

# Compile a prompt with variables
compiled_prompt = prompt.compile(query="What's the weather?", context="sunny", llm_response="It's sunny today")
print("Compiled prompt:", compiled_prompt)

# Get prompt parameters
parameters = prompt.get_parameters()
print("Prompt parameters:", parameters)
```

For more detailed information on Prompt Management, please refer to the [Prompt Management documentation](docs/prompt_management.md).



