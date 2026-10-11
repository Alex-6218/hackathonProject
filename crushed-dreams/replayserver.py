"""Tiny API between the Godot game and Tiger Cloud.

Godot never talks to the database directly: the connection string (with its
password) stays in this process's .env, not inside the game.

Setup:
    pip install fastapi uvicorn "psycopg[binary]" psycopg_pool python-dotenv
    # .env  ->  TIGER_URL=<connection string from your Tiger Cloud service>
    python replay_server.py
"""

import os
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

load_dotenv()

pool = ConnectionPool(os.environ["TIGER_URL"], min_size=1, max_size=4, open=True)
app = FastAPI()


class NewRun(BaseModel):
    level: int
    started: float  # unix seconds


class FinishRun(BaseModel):
    duration: float
    completed: bool


@app.post("/runs")
def create_run(run: NewRun):
    with pool.connection() as conn:
        row = conn.execute(
            "INSERT INTO runs (level, started_at) VALUES (%s, to_timestamp(%s)) RETURNING id",
            (run.level, run.started),
        ).fetchone()
    return {"id": str(row[0])}


@app.post("/runs/{run_id}/frames")
def add_frames(run_id: str, frames: list[list[Any]]):
    # Frame layout: [unix_time, entity, kind, px, py, pz, rx, ry, rz, extra]
    rows = [
        (f[0], run_id, f[1], f[2], *f[3:9], Jsonb(f[9]) if f[9] else None)
        for f in frames
    ]
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO replay_frames "
                "(time, run_id, entity, kind, px, py, pz, rx, ry, rz, extra) "
                "VALUES (to_timestamp(%s), %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                rows,
            )
    return {"saved": len(rows)}


@app.post("/runs/{run_id}/finish")
def finish_run(run_id: str, body: FinishRun):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE runs SET duration_s = %s, completed = %s WHERE id = %s::uuid",
            (body.duration, body.completed, run_id),
        )
    return {"ok": True}


@app.get("/runs")
def list_runs(limit: int = 10):
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, level, started_at, duration_s, completed "
            "FROM runs ORDER BY started_at DESC LIMIT %s",
            (limit,),
        ).fetchall()
    return [
        {
            "id": str(r[0]),
            "level": r[1],
            "started": r[2].isoformat(),
            "duration": r[3],
            "completed": r[4],
        }
        for r in rows
    ]


@app.get("/runs/{run_id}/frames")
def get_frames(run_id: str):
    # Same layout as the upload, but with time as seconds since the run started.
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT EXTRACT(EPOCH FROM f.time - r.started_at)::float8,
                   f.entity, f.kind, f.px, f.py, f.pz, f.rx, f.ry, f.rz, f.extra
            FROM replay_frames f
            JOIN runs r ON r.id = f.run_id
            WHERE f.run_id = %s::uuid
            ORDER BY f.time
            """,
            (run_id,),
        ).fetchall()
    return [list(r) for r in rows]


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)