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
    createdAt: str  # ISO 8601
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
        self.tx_store: Dict[str, dict] = {}          # txId -> full tx dict
        self.adj: Dict[str, Set[str]] = defaultdict(set)     # from -> set of to
        self.reverse_adj: Dict[str, Set[str]] = defaultdict(set) # to -> set of from
        self.timestamps: Dict[str, datetime] = {}    # txId -> datetime
        self.from_to_latest: Dict[Tuple[str, str], str] = {}  # (from, to) -> latest txId
        self.identity_store: Dict[str, Dict] = {}    # txId -> {ip, device}
        self.score_cache: Dict[str, float] = {}      # txId -> riskScore

    def add_transaction(self, tx: TransactionRequest, score: float):
        tx_id = tx.txId
        self.tx_store[tx_id] = tx.dict()
        self.timestamps[tx_id] = datetime.fromisoformat(tx.createdAt.replace('Z', '+00:00'))
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

    def cleanup_old(self, reference_time: datetime):
        """Remove transactions older than (reference_time - 24h)."""
        cutoff = reference_time - timedelta(hours=24)
        to_remove = [tid for tid, ts in self.timestamps.items() if ts < cutoff]
        for tid in to_remove:
            tx_data = self.tx_store.get(tid)
            if tx_data:
                f = tx_data['fromUserId']
                t = tx_data['toUserId']
                # Remove from adjacency if no other active transaction with same edge
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

# ---------- Helper: Find longest path ending at a node ----------

def find_longest_paths_to_node(node: str, max_depth: int = 6) -> List[List[str]]:
    """
    Find all simple paths from any source to 'node' (reverse traversal).
    Returns paths sorted by length (longest first).
    """
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
    # Sort by length descending (longest first)
    paths.sort(key=len, reverse=True)
    return paths

def get_longest_path(node: str) -> List[str]:
    """Return the longest path ending at 'node' (including node)."""
    all_paths = find_longest_paths_to_node(node)
    if not all_paths:
        return [node]  # just itself
    # Return the longest one (first)
    return all_paths[0]

# ---------- Scoring Functions ----------

def structural_signals(tx: TransactionRequest) -> float:
    """Global structural signals (cycles, length, convergence/divergence)."""
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

    # Path length: longest chain ending at fromUserId
    path = get_longest_path(tx.fromUserId)
    if len(path) >= 4:
        score += 0.1 * (len(path) - 3)

    # Convergence: many incoming to toUserId
    if len(state.reverse_adj.get(tx.toUserId, set())) > 1:
        score += 0.15
    # Divergence: many outgoing from fromUserId
    if len(state.adj.get(tx.fromUserId, set())) > 1:
        score += 0.1

    return min(score, 0.6)

def identity_signals(tx: TransactionRequest, path: List[str]) -> float:
    """
    Evaluate identity changes along the given path.
    path includes all nodes from source to tx.fromUserId.
    We check IP/device changes between consecutive edges.
    Also check for disappearance (present earlier, absent later).
    """
    if len(path) < 2:
        return 0.0
    score = 0.0
    # For each edge in the path, get the transaction ID
    for i in range(len(path)-1):
        f = path[i]
        t = path[i+1]
        edge_tx_id = state.from_to_latest.get((f, t))
        if not edge_tx_id:
            continue
        tx_data = state.tx_store.get(edge_tx_id)
        if not tx_data:
            continue
        # Compare identity with previous edge (if any)
        if i > 0:
            prev_f = path[i-1]
            prev_t = path[i]
            prev_edge_id = state.from_to_latest.get((prev_f, prev_t))
            if prev_edge_id:
                prev_data = state.tx_store.get(prev_edge_id)
                if prev_data:
                    # IP change
                    if tx_data.get('ipAddress') and prev_data.get('ipAddress'):
                        if tx_data['ipAddress'] != prev_data['ipAddress']:
                            score += 0.15
                    # IP disappearance: present earlier, absent now
                    elif prev_data.get('ipAddress') and not tx_data.get('ipAddress'):
                        score += 0.2
                    # Device change
                    if tx_data.get('deviceId') and prev_data.get('deviceId'):
                        if tx_data['deviceId'] != prev_data['deviceId']:
                            score += 0.15
                    elif prev_data.get('deviceId') and not tx_data.get('deviceId'):
                        score += 0.2
    return min(score, 0.5)

def value_signals(tx: TransactionRequest, path: List[str]) -> float:
    """
    Evaluate amount progression along the path.
    path includes all nodes from source to tx.fromUserId.
    We append the new transaction amount as the last step.
    """
    if len(path) < 2:
        return 0.0
    # Get amounts along the edges of the path
    amounts = []
    for i in range(len(path)-1):
        f = path[i]
        t = path[i+1]
        edge_tx_id = state.from_to_latest.get((f, t))
        if not edge_tx_id:
            return 0.0  # should not happen
        tx_data = state.tx_store.get(edge_tx_id)
        if not tx_data:
            return 0.0
        amounts.append(tx_data['amount'])
    # Append the new transaction amount
    amounts.append(tx.amount)

    # Compute ratios (step i+1 / step i)
    ratios = []
    for i in range(len(amounts)-1):
        if amounts[i] == 0:
            continue
        ratios.append(amounts[i+1] / amounts[i])

    if not ratios:
        return 0.0

    # Check for reversal (any ratio > 1.0)
    if any(r > 1.0 for r in ratios):
        return 0.4  # high penalty for reversal

    # Check for consistent decay (all ratios < 1 and close to 0.99)
    # Calculate mean and variance
    mean = sum(ratios) / len(ratios)
    variance = sum((r - mean)**2 for r in ratios) / len(ratios) if len(ratios) > 1 else 0.0
    # If all ratios < 1, this is typical layering. Low risk.
    if all(r < 1.0 for r in ratios):
        # But if variance is high, it might be less consistent -> medium risk
        if variance > 0.01:
            return 0.15
        else:
            return 0.05  # very low
    else:
        # Mixed – some below, some above? Actually we already handled >1.
        # If some ratios exactly 1.0? Rare.
        return 0.2

def combine_scores(structural, identity, value) -> float:
    """Weighted combination to produce final risk score."""
    # Weights: structural 0.3, identity 0.3, value 0.4
    raw = structural * 0.3 + identity * 0.3 + value * 0.4
    # Cap at 1.0
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
        # Check idempotency
        cached = state.get_score(tx.txId)
        if cached is not None:
            results.append(TransactionResult(txId=tx.txId, riskScore=cached))
            continue

        # 1. Cleanup old transactions based on this tx's createdAt
        tx_time = datetime.fromisoformat(tx.createdAt.replace('Z', '+00:00'))
        state.cleanup_old(tx_time)

        # 2. Find the longest path ending at tx.fromUserId
        path = get_longest_path(tx.fromUserId)
        # The path includes tx.fromUserId at the end; we'll use it for identity and value

        # 3. Compute signals
        struct_score = structural_signals(tx)
        id_score = identity_signals(tx, path)
        value_score = value_signals(tx, path)

        # 4. Combine
        final_score = combine_scores(struct_score, id_score, value_score)

        # 5. Add transaction to state
        state.add_transaction(tx, final_score)

        # 6. Append result
        results.append(TransactionResult(txId=tx.txId, riskScore=final_score))

    return TransactionsResponse(transactions=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
