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
    api_keys={"OPENAI_API_KEY": "YOUR_OPENAI_API_KEY"}
)
```

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
).start_trace()

# Your code here

# Stop the trace recording
tracer.stop_trace()

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

### Create and manage datasets for your projects.

```python
from ragaai_catalyst import Dataset

# Initialize Dataset management for a specific project
dataset_manager = Dataset(project_name="project_name")

# List existing datasets
datasets = dataset_manager.list_datasets()
print("Exisiting Datasets:", datasets)
```

### 1. Create a new dataset from trace
```python
dataset_manager.create_from_trace(
    dataset_name='Test-dataset-1',
    filter_list=[
        {
            "name": "llm_model",
            "values": ["gpt-3.5-turbo", "gpt-4"]
        },
        {
            "name": "prompt_length",
            "lte": 27,
            "gte": 23
        }
    ]
)
```

### 2. Create a new dataset from csv

#### a. `get_csv_schema()`
Retrieves the valid schema elements that the CSV column names must map to.

##### Returns
- A dictionary with the schema elements.
  - `success`: Boolean indicating whether the schema elements were fetched successfully.
  - `data['schemaElements']`: List of valid schema column names.
    
```python
schemaElements = dataset_manager.get_csv_schema()['data']['schemaElements']
print('Supported column names: ', schemaElements)
```

#### b. `create_from_csv(csv_path, dataset_name, schema_mapping)`
Uploads the CSV file to the server, performs schema mapping, and stores the dataset.

##### Parameters
- `csv_path` (str): Path to the CSV file.
- `dataset_name` (str): The name for the dataset you want to create from this CSV.
- `schema_mapping` (dict): Dictionary mapping CSV columns to schema elements. Must be in the format `{csv_column: schema_element}`.

```python
dataset_manager.create_from_csv(
    csv_path='path/to/your.csv',
    dataset_name='MyDataset',
    schema_mapping={'column1': 'schema_element1', 'column2': 'schema_element2'}
)
```




