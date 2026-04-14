from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import PersonProfile


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FriendRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    def _init_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT NOT NULL,
                face_signature TEXT NOT NULL,
                focal_points TEXT NOT NULL,
                affinity INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_seen_at TEXT
            )
            """
        )
        self.connection.commit()

    def _row_to_person(self, row: sqlite3.Row) -> PersonProfile:
        focal_points_data = json.loads(row["focal_points"])
        return PersonProfile(
            id=row["id"],
            display_name=row["display_name"],
            face_signature=json.loads(row["face_signature"]),
            focal_points=[(float(x), float(y)) for x, y in focal_points_data],
            affinity=row["affinity"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
        )

    def list_people(self) -> list[PersonProfile]:
        rows = self.connection.execute(
            "SELECT * FROM people ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_person(row) for row in rows]

    def get_person(self, person_id: int) -> PersonProfile | None:
        row = self.connection.execute(
            "SELECT * FROM people WHERE id = ?",
            (person_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_person(row)

    def create_person(
        self,
        signature: list[float],
        focal_points: list[tuple[float, float]],
        display_name: str | None = None,
    ) -> PersonProfile:
        created_at = utc_now_iso()
        cursor = self.connection.execute(
            """
            INSERT INTO people (display_name, face_signature, focal_points, affinity, created_at)
            VALUES (?, ?, ?, 0, ?)
            """,
            (
                display_name or "pending",
                json.dumps(signature),
                json.dumps(focal_points),
                created_at,
            ),
        )
        person_id = int(cursor.lastrowid)
        final_name = display_name or f"Person {person_id}"
        self.connection.execute(
            "UPDATE people SET display_name = ? WHERE id = ?",
            (final_name, person_id),
        )
        self.connection.commit()
        person = self.get_person(person_id)
        if person is None:
            raise RuntimeError("Failed to load newly created person")
        return person

    def update_affinity(self, person_id: int, delta: int) -> PersonProfile:
        self.connection.execute(
            """
            UPDATE people
            SET affinity = MAX(0, MIN(100, affinity + ?))
            WHERE id = ?
            """,
            (delta, person_id),
        )
        self.connection.commit()
        person = self.get_person(person_id)
        if person is None:
            raise KeyError(f"Unknown person id {person_id}")
        return person

    def record_seen(self, person_id: int, timestamp: str | None = None) -> PersonProfile:
        seen_at = timestamp or utc_now_iso()
        self.connection.execute(
            "UPDATE people SET last_seen_at = ? WHERE id = ?",
            (seen_at, person_id),
        )
        self.connection.commit()
        person = self.get_person(person_id)
        if person is None:
            raise KeyError(f"Unknown person id {person_id}")
        return person

    def find_best_match(
        self, signature: list[float], threshold: float
    ) -> tuple[PersonProfile | None, float]:
        best_person: PersonProfile | None = None
        best_distance = float("inf")

        for person in self.list_people():
            distance = _signature_distance(signature, person.face_signature)
            if distance < best_distance:
                best_distance = distance
                best_person = person

        if best_person is None or best_distance > threshold:
            return None, 0.0

        confidence = max(0.0, 1.0 - (best_distance / max(threshold, 1e-6)))
        return best_person, confidence


def _signature_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    if not left:
        return float("inf")
    total = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        total += abs(left_value - right_value)
    return total / len(left)
