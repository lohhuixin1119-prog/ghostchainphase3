from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Set, Tuple
from datetime import datetime, timedelta, timezone
import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ghost Chains - AML Risk Scoring (Phase 3)")

# ---------- Data Models ----------

class TransactionRequest(BaseModel):
    txId: str
    fromUserId: str
    toUserId: str
    amount: float = Field(..., gt=0)
    createdAt: str
    ipAddress: Optional[str] = None
    deviceId: Optional[str] = None

    @validator("createdAt")
    def validate_iso8601(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except:
            raise ValueError("Invalid ISO 8601 timestamp")
        return v

class TransactionsRequest(BaseModel):
    transactions: List[TransactionRequest]

class TransactionResult(BaseModel):
    txId: str
    riskScore: float

class TransactionsResponse(BaseModel):
    transactions: List[TransactionResult]

# ---------- Graph State ----------

class GraphState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tx_store: Dict[str, dict] = {}
        self.adj: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adj: Dict[str, Set[str]] = defaultdict(set)
        self.timestamps: Dict[str, datetime] = {}
        self.from_to_latest: Dict[Tuple[str, str], str] = {}
        self.identity_store: Dict[str, Dict] = {}
        self.score_cache: Dict[str, float] = {}

    def add_transaction(self, tx: TransactionRequest, score: float):
        tx_id = tx.txId
        self.tx_store[tx_id] = tx.dict()
        dt = datetime.fromisoformat(tx.createdAt.replace('Z', '+00:00'))
        self.timestamps[tx_id] = dt
        self.adj[tx.fromUserId].add(tx.toUserId)
        self.reverse_adj[tx.toUserId].add(tx.fromUserId)
        self.from_to_latest[(tx.fromUserId, tx.toUserId)] = tx_id
        if tx.ipAddress or tx.deviceId:
            self.identity_store[tx_id] = {
                'ip': tx.ipAddress,
                'device': tx.deviceId
            }
        self.score_cache[tx_id] = score

    def get_score(self, tx_id: str) -> Optional[float]:
        return self.score_cache.get(tx_id)

    def get_transaction(self, tx_id: str) -> Optional[dict]:
        return self.tx_store.get(tx_id)

    def cleanup(self, cutoff: datetime):
        """Remove all transactions with timestamp < cutoff."""
        to_remove = [tid for tid, ts in self.timestamps.items() if ts < cutoff]
        for tid in to_remove:
            tx_data = self.tx_store.get(tid)
            if tx_data:
                f = tx_data['fromUserId']
                t = tx_data['toUserId']
                # Remove from adjacency if no other active edge
                other = [tid2 for tid2, ts2 in self.timestamps.items()
                         if ts2 >= cutoff and self.tx_store.get(tid2, {}).get('fromUserId') == f
                         and self.tx_store[tid2].get('toUserId') == t]
                if not other:
                    if t in self.adj.get(f, set()):
                        self.adj[f].remove(t)
                        if not self.adj[f]:
                            del self.adj[f]
                    if f in self.reverse_adj.get(t, set()):
                        self.reverse_adj[t].remove(f)
                        if not self.reverse_adj[t]:
                            del self.reverse_adj[t]
                    if self.from_to_latest.get((f, t)) == tid:
                        del self.from_to_latest[(f, t)]
                del self.tx_store[tid]
                del self.timestamps[tid]
                if tid in self.identity_store:
                    del self.identity_store[tid]
                if tid in self.score_cache:
                    del self.score_cache[tid]

# ---------- Global State ----------
state = GraphState()

# ---------- Helper: Get the longest path ending at a node ----------

def get_longest_path_to_node(node: str, max_depth: int = 6) -> List[str]:
    """Return the longest simple path from any source to 'node' (inclusive)."""
    paths = []
    queue = deque([(node, [node])])
    while queue:
        current, path = queue.popleft()
        if len(path) > max_depth:
            continue
        for pred in state.reverse_adj.get(current, set()):
            if pred in path:
                continue
            new_path = [pred] + path
            paths.append(new_path)
            queue.append((pred, new_path))
    if not paths:
        return [node]  # only itself
    paths.sort(key=len, reverse=True)
    return paths[0]

# ---------- Scoring Functions ----------

def structural_score(tx: TransactionRequest) -> float:
    """Global structural risk (cycles, branching, path length)."""
    score = 0.0
    # Cycle detection: BFS from toUserId to fromUserId
    visited = set()
    queue = deque([tx.toUserId])
    found_cycle = False
    while queue:
        node = queue.popleft()
        if node == tx.fromUserId:
            found_cycle = True
            break
        if node in visited:
            continue
        visited.add(node)
        for nxt in state.adj.get(node, set()):
            if nxt not in visited:
                queue.append(nxt)
    if found_cycle:
        score += 0.3

    # Path length (longest chain ending at fromUserId)
    path = get_longest_path_to_node(tx.fromUserId)
    if len(path) >= 4:
        score += 0.1 * (len(path) - 3)

    # Convergence (multiple incoming to toUserId)
    if len(state.reverse_adj.get(tx.toUserId, set())) > 1:
        score += 0.15
    # Divergence (multiple outgoing from fromUserId)
    if len(state.adj.get(tx.fromUserId, set())) > 1:
        score += 0.1

    return min(score, 0.6)

def identity_score_on_path(path: List[str]) -> float:
    """
    Evaluate identity changes along a single path.
    path includes nodes from source to the current fromUserId.
    We compare IP/device between consecutive edges on that path.
    """
    if len(path) < 2:
        return 0.0
    score = 0.0
    for i in range(1, len(path)):
        f = path[i-1]
        t = path[i]
        edge_tx_id = state.from_to_latest.get((f, t))
        if not edge_tx_id:
            continue
        tx_data = state.tx_store.get(edge_tx_id)
        if not tx_data:
            continue
        if i > 1:
            prev_f = path[i-2]
            prev_t = path[i-1]
            prev_edge_id = state.from_to_latest.get((prev_f, prev_t))
            if prev_edge_id:
                prev_data = state.tx_store.get(prev_edge_id)
                if prev_data:
                    # IP change/disappearance
                    prev_ip = prev_data.get('ipAddress')
                    curr_ip = tx_data.get('ipAddress')
                    if prev_ip and curr_ip and prev_ip != curr_ip:
                        score += 0.15
                    elif prev_ip and not curr_ip:
                        score += 0.2
                    # Device change/disappearance
                    prev_dev = prev_data.get('deviceId')
                    curr_dev = tx_data.get('deviceId')
                    if prev_dev and curr_dev and prev_dev != curr_dev:
                        score += 0.15
                    elif prev_dev and not curr_dev:
                        score += 0.2
    return min(score, 0.5)

def value_score_on_path(path: List[str], new_amount: float) -> float:
    """
    Evaluate value progression along a path.
    path includes nodes up to the current fromUserId (excludes new edge).
    We append new_amount as the final step.
    """
    if len(path) < 2:
        return 0.0
    amounts = []
    for i in range(len(path)-1):
        f = path[i]
        t = path[i+1]
        edge_tx_id = state.from_to_latest.get((f, t))
        if not edge_tx_id:
            return 0.0
        tx_data = state.tx_store.get(edge_tx_id)
        if not tx_data:
            return 0.0
        amounts.append(tx_data['amount'])
    amounts.append(new_amount)

    ratios = []
    for i in range(len(amounts)-1):
        if amounts[i] == 0:
            continue
        ratios.append(amounts[i+1] / amounts[i])
    if not ratios:
        return 0.0

    # Reversal: any ratio > 1.0 -> high risk
    if any(r > 1.0 for r in ratios):
        return 0.4

    # No reversal: check consistency (variance of ratios)
    mean = sum(ratios) / len(ratios)
    variance = sum((r - mean)**2 for r in ratios) / len(ratios) if len(ratios) > 1 else 0.0
    if variance > 0.01:
        return 0.15
    else:
        return 0.05  # normal layering

def combine_scores(structural, identity, value) -> float:
    """Weighted combination, emphasising value in Phase 3."""
    raw = structural * 0.3 + identity * 0.3 + value * 0.4
    return min(raw, 1.0)

# ---------- Endpoints ----------

@app.get("/ghost-chains/health")
async def health():
    return {"status": "ok"}

@app.post("/ghost-chains/reset")
async def reset(clear: dict = None):
    state.reset()
    return {"clearTransactions": True}

@app.post("/ghost-chains/transactions", response_model=TransactionsResponse)
async def process_transactions(req: TransactionsRequest):
    results = []
    for tx in req.transactions:
        # Idempotency
        cached = state.get_score(tx.txId)
        if cached is not None:
            results.append(TransactionResult(txId=tx.txId, riskScore=cached))
            continue

        # Expire old transactions based on this transaction's timestamp
        tx_time = datetime.fromisoformat(tx.createdAt.replace('Z', '+00:00'))
        cutoff = tx_time - timedelta(hours=24)
        state.cleanup(cutoff)

        # 1. Structural signal (global graph)
        struct_score = structural_score(tx)

        # 2. Find the longest path ending at tx.fromUserId
        path = get_longest_path_to_node(tx.fromUserId)

        # 3. Identity and Value signals on that path
        id_score = identity_score_on_path(path)
        val_score = value_score_on_path(path, tx.amount)

        # 4. Combine
        final_score = combine_scores(struct_score, id_score, val_score)

        # 5. Add transaction
        state.add_transaction(tx, final_score)

        results.append(TransactionResult(txId=tx.txId, riskScore=final_score))

    return TransactionsResponse(transactions=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
