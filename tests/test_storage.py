from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_friend.storage import FriendRepository


class FriendRepositoryTests(unittest.TestCase):
    def test_create_update_and_match_person(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = FriendRepository(Path(temp_dir) / "friend.db")

            created = repository.create_person(
                signature=[0.1, 0.2, 0.3],
                focal_points=[(0.1, 0.2), (0.3, 0.4)],
            )
            self.assertEqual(created.display_name, "Person 1")

            updated = repository.update_affinity(created.id, 15)
            self.assertEqual(updated.affinity, 15)

            matched, confidence = repository.find_best_match(
                [0.1, 0.2, 0.31],
                threshold=0.1,
            )
            self.assertIsNotNone(matched)
            self.assertEqual(matched.id, created.id)
            self.assertGreater(confidence, 0.0)

            repository.close()


if __name__ == "__main__":
    unittest.main()
