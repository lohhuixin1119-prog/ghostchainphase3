from datetime import datetime, timedelta, timezone
from typing import List, Dict, Set, Optional
from collections import deque
from models import TransactionIn

class AMLGraph:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transactions: Dict[str, TransactionIn] = {}
        self.time_index: List[tuple[datetime, str]] = []
        self.out_edges: Dict[str, List[str]] = {} 
        self.in_edges: Dict[str, List[str]] = {}  
        self.score_cache: Dict[str, float] = {}
        self.latest_time: Optional[datetime] = None  # Tracks global clock

    def parse_time(self, time_str: str) -> datetime:
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))

    def evict_old_transactions(self):
        """Removes transactions older than 24 hours relative to the global latest time."""
        if not self.latest_time:
            return
            
        cutoff_time = self.latest_time - timedelta(hours=24)
        
        while self.time_index and self.time_index[0][0] < cutoff_time:
            _, tx_id = self.time_index.pop(0)
            if tx_id in self.transactions:
                tx = self.transactions.pop(tx_id)
                
                if tx.fromUserId in self.out_edges:
                    self.out_edges[tx.fromUserId] = [tid for tid in self.out_edges[tx.fromUserId] if tid != tx_id]
                if tx.toUserId in self.in_edges:
                    self.in_edges[tx.toUserId] = [tid for tid in self.in_edges[tx.toUserId] if tid != tx_id]

    def add_transaction(self, tx: TransactionIn) -> bool:
        if tx.txId in self.score_cache:
            return False
            
        tx_time = self.parse_time(tx.createdAt)
        
        # Advance global clock if this transaction is the newest
        if self.latest_time is None or tx_time > self.latest_time:
            self.latest_time = tx_time
            
        self.evict_old_transactions()
        
        self.transactions[tx.txId] = tx
        self.time_index.append((tx_time, tx.txId))
        
        self.out_edges.setdefault(tx.fromUserId, []).append(tx.txId)
        self.in_edges.setdefault(tx.toUserId, []).append(tx.txId)
        
        self.time_index.sort(key=lambda x: x[0]) 
        return True

    def get_immediate_predecessors(self, user_id: str, current_time: datetime) -> List[TransactionIn]:
        """Gets transactions arriving at the user strictly prior to or exactly at the current time."""
        edge_ids = self.in_edges.get(user_id, [])
        preds = []
        for tid in edge_ids:
            ptx = self.transactions[tid]
            if self.parse_time(ptx.createdAt) <= current_time:
                preds.append(ptx)
        return preds

    def has_temporal_path(self, start_node: str, target_node: str, max_time: datetime, max_depth=10) -> bool:
        """Phase 1: Time-Aware BFS to detect cycles ensuring chronological flow."""
        # Queue stores (current_node, time_of_arrival, depth)
        queue = deque([(start_node, datetime.min.replace(tzinfo=timezone.utc), 0)])
        visited = set()
        
        while queue:
            curr_node, curr_time, depth = queue.popleft()
            
            if curr_node == target_node and depth > 0:
                return True
            if depth >= max_depth:
                continue
                
            for edge_id in self.out_edges.get(curr_node, []):
                tx = self.transactions[edge_id]
                tx_time = self.parse_time(tx.createdAt)
                
                # A valid path MUST flow forward in time
                if curr_time <= tx_time <= max_time:
                    state_key = (tx.toUserId, tx_time)
                    if state_key not in visited:
                        visited.add(state_key)
                        queue.append((tx.toUserId, tx_time, depth + 1))
        return False
