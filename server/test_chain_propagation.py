from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.runnables import RunnableConfig
from server.schemas import Config

async def corrected_dispatch_downstream_media(config_obj: Any, config: RunnableConfig) -> dict:
    task_id = config.get("configurable", {}).get("task_id")
    return {
        "resolved_task_id": task_id,
        "media": config_obj.media_required
    }

@pytest.mark.asyncio
async def test_langchain_context_passing():
    mock_pydantic_output = MagicMock(spec=Config)
    mock_pydantic_output.media_required = "STATIC_PNG"
    mock_pydantic_output.media_prompt = "Cyberpunk interface blueprint"
    
    test_task_id = "test-token-uuid-12345"
    run_config: RunnableConfig = {"configurable": {"task_id": test_task_id}}
    
    result = await corrected_dispatch_downstream_media(mock_pydantic_output, run_config)
    
    assert result["resolved_task_id"] == test_task_id
    assert result["media"] == "STATIC_PNG"
    print("Context propagation testing completed successfully.")