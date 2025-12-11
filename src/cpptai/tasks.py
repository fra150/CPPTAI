"""Informatics task generator.

Produces structured tasks across typical CS/IT domains: algorithms, networks,
security, databases, and devops. Each task includes a title, description, and
difficulty.
"""

from __future__ import annotations

from typing import Dict, List


def generate_informatics_tasks(n: int = 10) -> List[Dict]:
    catalog = [
        {"category": "Algorithms", "title": "Implement Dijkstra", "desc": "Shortest paths on weighted graphs", "difficulty": "medium"},
        {"category": "Security", "title": "Add input validation", "desc": "Sanitize and validate user inputs", "difficulty": "easy"},
        {"category": "Networks", "title": "HTTP client retry policy", "desc": "Exponential backoff and jitter", "difficulty": "medium"},
        {"category": "Databases", "title": "Design normalized schema", "desc": "3NF for user/projects/tasks", "difficulty": "hard"},
        {"category": "DevOps", "title": "Add CI unit tests", "desc": "Run Python unittest on push", "difficulty": "easy"},
        {"category": "Algorithms", "title": "Topological sort", "desc": "Order DAG nodes respecting dependencies", "difficulty": "easy"},
        {"category": "Security", "title": "Secret management", "desc": "Load env vars via .env and vault", "difficulty": "medium"},
        {"category": "Networks", "title": "Rate limiting", "desc": "Protect endpoints from abuse", "difficulty": "hard"},
        {"category": "Databases", "title": "Query optimization", "desc": "Add indexes and analyze plans", "difficulty": "medium"},
        {"category": "DevOps", "title": "Containerize app", "desc": "Create Dockerfile and compose", "difficulty": "medium"},
    ]
    return catalog[: max(1, n)]

