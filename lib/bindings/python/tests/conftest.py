import pytest
import uvloop


@pytest.fixture(scope="session")
def event_loop_policy():
    return uvloop.EventLoopPolicy()
