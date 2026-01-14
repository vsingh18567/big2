"""In-memory game store with thread-safe operations."""

from threading import Lock

from big2.game_server.game_logic import Game


class GameStore:
    """Thread-safe in-memory store for games."""

    def __init__(self):
        self._games: dict[str, Game] = {}
        self._lock = Lock()

    def create_game(self, game_id: str) -> Game:
        """
        Create a new game.

        Args:
            game_id: Unique game identifier

        Returns:
            The created game

        Raises:
            ValueError: If game_id already exists
        """
        with self._lock:
            if game_id in self._games:
                raise ValueError(f"Game {game_id} already exists")
            game = Game(game_id)
            self._games[game_id] = game
            return game

    def get_game(self, game_id: str) -> Game | None:
        """Get a game by ID."""
        with self._lock:
            return self._games.get(game_id)

    def list_games(self) -> list[str]:
        """List all game IDs."""
        with self._lock:
            return list(self._games.keys())

    def delete_game(self, game_id: str) -> bool:
        """
        Delete a game.

        Args:
            game_id: Game to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if game_id in self._games:
                del self._games[game_id]
                return True
            return False


# Global game store instance
game_store = GameStore()
