from farm_system.hello_world import root_agent


def test_hello_world_exports_a_standard_adk_root_agent() -> None:
    assert root_agent.name == "hello_world"
    assert root_agent.description == "Responds with a friendly greeting from the farm system."
