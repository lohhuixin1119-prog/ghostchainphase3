from datetime import datetime, timedelta
from typing import List, Dict, Set
from models import TransactionIn

class AMLGraph:
    def __init__(self):
        self.reset()

    def reset(self):
        # Maps txId -> TransactionIn
        self.transactions: Dict[str, TransactionIn] = {}
        # Tracks temporal order for 24h eviction: list of (timestamp, txId)
        self.time_index: List[tuple[datetime, str]] = []
        
        # Adjacency lists for structural traversal
        self.out_edges: Dict[str, List[str]] = {} # userId -> list of txIds originating here
        self.in_edges: Dict[str, List[str]] = {}  # userId -> list of txIds arriving here
        
        # Idempotency cache: txId -> calculated score
        self.score_cache: Dict[str, float] = {}

    def parse_time(self, time_str: str) -> datetime:
        """Parse ISO 8601 string to a timezone-aware datetime object."""
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))

    def evict_old_transactions(self, current_time: datetime):
        """Removes transactions older than 24 hours relative to the current transaction."""
        cutoff_time = current_time - timedelta(hours=24)
        
        while self.time_index and self.time_index[0][0] < cutoff_time:
            _, tx_id = self.time_index.pop(0)
            if tx_id in self.transactions:
                tx = self.transactions.pop(tx_id)
                
                # Remove from graph edges
                if tx.fromUserId in self.out_edges:
                    self.out_edges[tx.fromUserId] = [tid for tid in self.out_edges[tx.fromUserId] if tid != tx_id]
                if tx.toUserId in self.in_edges:
                    self.in_edges[tx.toUserId] = [tid for tid in self.in_edges[tx.toUserId] if tid != tx_id]

    def add_transaction(self, tx: TransactionIn) -> bool:
        """Adds a transaction to the graph. Returns False if it was already processed."""
        if tx.txId in self.score_cache:
            return False
            
        tx_time = self.parse_time(tx.createdAt)
        self.evict_old_transactions(tx_time)
        
        self.transactions[tx.txId] = tx
        self.time_index.append((tx_time, tx.txId))
        
        self.out_edges.setdefault(tx.fromUserId, []).append(tx.txId)
        self.in_edges.setdefault(tx.toUserId, []).append(tx.txId)
        
        # Ensure time index remains sorted chronologically
        self.time_index.sort(key=lambda x: x[0]) 
        return True

    def get_predecessors(self, user_id: str, current_time: datetime) -> List[TransactionIn]:
        """Gets transactions arriving at the user strictly prior to the current time."""
        edge_ids = self.in_edges.get(user_id, [])
        preds = []
        for tid in edge_ids:
            ptx = self.transactions[tid]
            if self.parse_time(ptx.createdAt) <= current_time:
                preds.append(ptx)
        return preds

    def has_path(self, start_node: str, target_node: str, max_depth=5) -> bool:
        """Phase 1: Basic Depth-First Search to detect structural cycles/paths."""
        visited: Set[str] = set()
        
        def dfs(current: str, depth: int) -> bool:
            if current == target_node:
                return True
            if depth >= max_depth:
                return False
            
            visited.add(current)
            for edge_id in self.out_edges.get(current, []):
                next_node = self.transactions[edge_id].toUserId
                if next_node not in visited:
                    if dfs(next_node, depth + 1):
                        return True
            return False
            
        return dfs(start_node, 0)
