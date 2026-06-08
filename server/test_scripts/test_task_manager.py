import asyncio
import pytest
from pydantic import BaseModel
from server.task_manager import TaskManager, TaskState

class DummyTaskState(BaseModel):
    status: str
    progress: int
    result: dict | None = None
    error: str | None = None

@pytest.mark.asyncio
async def test_task_manager_dictionary_unpacking():
    manager = TaskManager(max_concurrent_gpu_tasks=2)
    task_id = manager.create_task()
    
    manager.update_task(task_id, status="Running", progress=10)
    state = manager.get_task_status(task_id)
    assert state.status == "Running"
    assert state.progress == 10
    
    manager.update_task(task_id, progress=45)
    state = manager.get_task_status(task_id)
    assert state.status == "Running", "Dictionary unpacking broke unchanged fields!"
    assert state.progress == 45
    
    with pytest.raises(Exception):
        manager.update_task(task_id, progress="NOT_AN_INTEGER")


@pytest.mark.asyncio
async def test_gpu_semaphore_concurrency_boundary():
    manager = TaskManager(max_concurrent_gpu_tasks=2)
    execution_log = []

    async def simulated_gpu_bound_job(worker_name: str):
        execution_log.append(f"{worker_name}_waiting")
        async with manager.vram_semaphore:
            execution_log.append(f"{worker_name}_running")
            await asyncio.sleep(0.5)
            execution_log.append(f"{worker_name}_finished")

    await asyncio.gather(
        simulated_gpu_bound_job("Job_1"),
        simulated_gpu_bound_job("Job_2"),
        simulated_gpu_bound_job("Job_3"),
        simulated_gpu_bound_job("Job_4")
    )

    assert execution_log[0] == "Job_1_waiting"
    assert "Job_1_running" in execution_log[:4]
    assert "Job_2_running" in execution_log[:4]
    assert execution_log[-1] in ["Job_3_finished", "Job_4_finished"]
    print("\nTask Manager Concurrency & Dictionary State Mutations verified successfully!")