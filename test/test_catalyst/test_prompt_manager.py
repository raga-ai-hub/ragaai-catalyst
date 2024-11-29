import os
import pytest
import copy
from ragaai_catalyst import PromptManager, RagaAICatalyst
# from ragaai_catalyst.prompt_manager import PromptManager,PromptObject
@pytest.fixture
def base_url():
    return "https://catalyst.raga.ai/api"

@pytest.fixture
def access_keys():
    return {
        "access_key": os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        "secret_key": os.getenv("RAGAAI_CATALYST_SECRET_KEY")}


@pytest.fixture
def PromptManager(base_url, access_keys):
    """Create evaluation instance with specific project and dataset"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    catalyst = RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )
    return PromptManager(project_name="prompt_metric_dataset")

def test_prompt_initialistaion(PromptManager):
    prompt_manager = PromptManager(project_name="prompt_metric_dataset")
    prompt_list= prompt_manager.list_prompts()
    assert prompt_list ==['Sample', 'Hallu']



def test_prompt_object_compile():
    """
    Test compiling a prompt with variable substitution
    """
    # Arrange
    prompt_text = [{"content": "Hello {{name}}, you are {{age}} years old."}]
    parameters = [{"name": "temperature", "value": "0.7", "type": "float"}]
    model = "claude-3"

    prompt_obj = PromptObject(prompt_text, parameters, model)

    # Act
    compiled_text = prompt_obj.compile(name="John", age="30")

    # Assert
    assert compiled_text[0]["content"] == "Hello John, you are 30 years old."

def test_prompt_object_get_variables():
    """
    Test extracting variables from a prompt
    """
    # Arrange
    prompt_text = [
        {"content": "Hello {{name}}, you are {{age}} years old."},
        {"content": "From {{location}}"}
    ]
    parameters = [{"name": "temperature", "value": "0.7", "type": "float"}]
    model = "claude-3"

    prompt_obj = PromptObject(prompt_text, parameters, model)

    # Act
    variables = prompt_obj.get_variables()

    # Assert
    assert set(variables) == {"name", "age", "location"}

def test_prompt_object_get_model_parameters():
    """
    Test retrieving model parameters
    """
    # Arrange
    prompt_text = [{"content": "Test prompt"}]
    parameters = [
        {"name": "temperature", "value": "0.7", "type": "float"},
        {"name": "max_tokens", "value": "100", "type": "int"}
    ]
    model = "claude-3"

    prompt_obj = PromptObject(prompt_text, parameters, model)

    # Act
    model_params = prompt_obj.get_model_parameters()

    # Assert
    assert model_params == {
        "temperature": 0.7,
        "max_tokens": 100,
        "model": "claude-3"
    }

def test_prompt_object_compile_missing_variables():
    """
    Test error handling for missing variables
    """
    # Arrange
    prompt_text = [{"content": "Hello {{name}}, you are {{age}} years old."}]
    parameters = [{"name": "temperature", "value": "0.7", "type": "float"}]
    model = "claude-3"

    prompt_obj = PromptObject(prompt_text, parameters, model)

    # Act & Assert
    with pytest.raises(ValueError, match="Missing variable"):
        prompt_obj.compile(name="John")

def test_prompt_object_compile_extra_variables():
    """
    Test error handling for extra variables
    """
    # Arrange
    prompt_text = [{"content": "Hello {{name}} years old."}]
    parameters = [{"name": "temperature", "value": "0.7", "type": "float"}]
    model = "claude-3"

    prompt_obj = PromptObject(prompt_text, parameters, model)

    # Act & Assert
    with pytest.raises(ValueError, match="Extra variable"):
        prompt_obj.compile(name="John", age="30", location="New York")

def test_prompt_object_variable_extraction():
    """
    Test internal method for extracting variables
    """
    # Arrange
    prompt_obj = PromptObject([], [], "")
    content = "Hello {{name}}, you are {{age}} years old. Working at {{company}}"

    # Act
    variables = prompt_obj._extract_variable_from_content(content)

    # Assert
    assert set(variables) == {"name", "age", "company"}

def test_prompt_object_add_variable_value():
    """
    Test adding variable values to content
    """
    # Arrange
    prompt_obj = PromptObject([], [], "")
    content = "Hello {{name}}, you are {{age}} years old."
    user_variables = {"name": "John", "age": "30"}

    # Act
    updated_content = prompt_obj._add_variable_value_to_content(content, user_variables)

    # Assert
    assert updated_content == "Hello John, you are 30 years old."

def test_prompt_object_invalid_variable_type():
    """
    Test error handling for non-string variable values
    """
    # Arrange
    prompt_text = [{"content": "Hello {{name}}"}]
    parameters = []
    model = "claude-3"

    prompt_obj = PromptObject(prompt_text, parameters, model)

    # Act & Assert
    with pytest.raises(ValueError, match="Value for variable 'name' must be a string"):
        prompt_obj.compile(name=42)

def test_prompt_object_get_prompt_content():
    """
    Test retrieving original prompt content
    """
    # Arrange
    prompt_text = [{"content": "Test prompt {{variable}}"}]
    parameters = []
    model = "claude-3"

    prompt_obj = PromptObject(prompt_text, parameters, model)

    # Act
    content = prompt_obj.get_prompt_content()

    # Assert
    assert content == prompt_text