from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

app = FastAPI(title="Todo API")
todos: dict[int, "Todo"] = {}
_next_id = 1


class TodoCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    done: bool = False


class Todo(TodoCreate):
    id: int


def reset_store() -> None:
    global _next_id
    todos.clear()
    _next_id = 1


@app.get("/todos", response_model=list[Todo])
def list_todos() -> list[Todo]:
    return list(todos.values())


@app.post("/todos", response_model=Todo, status_code=status.HTTP_201_CREATED)
def create_todo(todo: TodoCreate) -> Todo:
    global _next_id
    created = Todo(id=_next_id, title=todo.title, done=todo.done)
    todos[created.id] = created
    _next_id += 1
    return created


@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int) -> Todo:
    todo = todos.get(todo_id)
    if todo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Todo not found")
    return todo


@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, todo: TodoCreate) -> Todo:
    if todo_id not in todos:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Todo not found")
    updated = Todo(id=todo_id, title=todo.title, done=todo.done)
    todos[todo_id] = updated
    return updated


@app.delete("/todos/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_todo(todo_id: int) -> Response:
    if todos.pop(todo_id, None) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Todo not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
