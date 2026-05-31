import asyncio
import uuid
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class TaskState(BaseModel):
    task_id: str
    status: str = "PENDING"
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TaskManager:
    def __init__(self, max_concurrent_gpu_tasks: int = 1):
        self._tasks: Dict[str, TaskState] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self.vram_semaphore = asyncio.Semaphore(max_concurrent_gpu_tasks)

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        state = TaskState(task_id=task_id)
        self._tasks[task_id] = state
        return task_id

    def get_task_status(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        if task_id in self._tasks:
            current_state = self._tasks[task_id]
            updated_data = current_state.model_dump()
            updated_data.update(kwargs)
            self._tasks[task_id] = TaskState(**updated_data)

    async def enqueue_job(self, task_id: str, payload: Dict[str, Any]):
        await self._queue.put({"task_id": task_id, "payload": payload})
        self.update_task(task_id, status="QUEUED", progress=5)

    async def dequeue_job(self) -> Dict[str, Any]:
        return await self._queue.get()

    def task_done(self):
        self._queue.task_done()

global_task_manager = TaskManager(max_concurrent_gpu_tasks=5)