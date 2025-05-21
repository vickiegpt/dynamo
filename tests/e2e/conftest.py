import pytest
from tests.utils import find_free_port
from tests.e2e.dynamo_client import DynamoRunProcess
# pytest fixture for DynamoRunProcess
@pytest.fixture()
def dynamo_run(backend, model, input_type, timeout):
    """
    Create and start a DynamoRunProcess for testing.
    """
    port = find_free_port()
    with DynamoRunProcess(
        model=model, backend=backend, port=port, input_type=input_type, timeout=timeout
    ) as process:
        yield process 
