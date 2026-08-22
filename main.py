from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Set, Tuple
from datetime import datetime, timedelta, timezone
import logging
import uuid
from collections import defaultdict, deque
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ghost Chains - AML Risk Scoring", version="3.0")

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

# ---------- State Management ----------

class GraphState:
    def __init__(self):
        self.reset()

    def reset(self):
        # Core storage
        self.tx_store: Dict[str, dict] = {}          # txId -> full tx data + processed score
        self.adj: Dict[str, Set[str]] = defaultdict(set)  # fromUserId -> set of toUserId
        self.reverse_adj: Dict[str, Set[str]] = defaultdict(set)  # toUserId -> set of fromUserId
        self.timestamps: Dict[str, datetime] = {}    # txId -> datetime
        self.from_to_latest: Dict[Tuple[str, str], str] = {}  # (from, to) -> latest txId
        self.identity_store: Dict[str, Dict] = {}    # txId -> ipAddress, deviceId
        self.score_cache: Dict[str, float] = {}      # txId -> riskScore (for idempotency)

    def add_transaction(self, tx: TransactionRequest, score: float):
        # Store everything
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

    def get_transaction(self, tx_id: str) -> Optional[dict]:
        return self.tx_store.get(tx_id)

    def get_score(self, tx_id: str) -> Optional[float]:
        return self.score_cache.get(tx_id)

    def cleanup_old(self, cutoff: datetime):
        """Remove transactions older than cutoff."""
        to_remove = [tid for tid, ts in self.timestamps.items() if ts < cutoff]
        for tid in to_remove:
            # Remove from stores
            tx_data = self.tx_store.get(tid)
            if tx_data:
                f = tx_data['fromUserId']
                t = tx_data['toUserId']
                # Remove adjacency if no other edge from f to t
                if self.from_to_latest.get((f,t)) == tid:
                    # Need to check if there are other transactions with same edge? 
                    # We'll rebuild later or keep only latest? For simplicity, we'll remove the edge from adjacency sets.
                    # But we need to ensure we don't remove if there are other tx still active with same edge.
                    # We'll check if any other active tx with same from-to exists.
                    other = [tid2 for tid2, ts2 in self.timestamps.items() if ts2 >= cutoff and self.tx_store.get(tid2, {}).get('fromUserId') == f and self.tx_store[tid2].get('toUserId') == t]
                    if not other:
                        # Remove from adjacency sets
                        if t in self.adj.get(f, set()):
                            self.adj[f].remove(t)
                            if not self.adj[f]:
                                del self.adj[f]
                        if f in self.reverse_adj.get(t, set()):
                            self.reverse_adj[t].remove(f)
                            if not self.reverse_adj[t]:
                                del self.reverse_adj[t]
                        # remove from from_to_latest
                        if self.from_to_latest.get((f,t)) == tid:
                            del self.from_to_latest[(f,t)]
                # remove from other stores
                del self.tx_store[tid]
                del self.timestamps[tid]
                if tid in self.identity_store:
                    del self.identity_store[tid]
                if tid in self.score_cache:
                    del self.score_cache[tid]

# ---------- Global state instance ----------
state = GraphState()

# ---------- Helper Functions for Scoring ----------

def get_paths_to_node(node: str, max_depth: int = 5) -> List[List[str]]:
    """Simple BFS to find paths ending at 'node' (reverse direction)."""
    # We'll find all simple paths from sources to node up to max_depth.
    # Not efficient but fine for moderate graphs.
    paths = []
    # Use reverse adjacency to traverse upstream
    queue = deque([(node, [node])])
    while queue:
        current, path = queue.popleft()
        if len(path) > max_depth:
            continue
        # Get predecessors
        for pred in state.reverse_adj.get(current, set()):
            if pred in path:
                continue  # avoid cycles
            new_path = [pred] + path
            paths.append(new_path)
            queue.append((pred, new_path))
    # Return paths sorted by length (shortest first)
    paths.sort(key=len)
    return paths

def structural_signals(tx: TransactionRequest) -> float:
    """Evaluate structural graph patterns for the new transaction."""
    score = 0.0
    # 1. Cycle detection: check if there is a path from tx.toUserId back to tx.fromUserId
    #    (forming a cycle when this edge is added)
    # Simple: BFS from tx.toUserId to tx.fromUserId (following outgoing edges)
    visited = set()
    queue = deque([tx.toUserId])
    found_cycle = False
    while queue and not found_cycle:
        node = queue.popleft()
        if node == tx.fromUserId:
            found_cycle = True
            break
        if node in visited:
            continue
        visited.add(node)
        for neighbor in state.adj.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)
    if found_cycle:
        score += 0.3  # cycles are suspicious
    # 2. Path length: if the transaction extends a long chain, increase score
    #    We'll find max path length ending at tx.fromUserId (incoming paths)
    paths = get_paths_to_node(tx.fromUserId, max_depth=6)
    max_len = max((len(p) for p in paths), default=1)
    if max_len >= 4:
        score += 0.1 * (max_len - 3)  # incremental
    # 3. Convergence/Divergence: check if toUserId has multiple incoming or fromUserId multiple outgoing
    if len(state.reverse_adj.get(tx.toUserId, set())) > 1:
        score += 0.15  # convergence
    if len(state.adj.get(tx.fromUserId, set())) > 1:
        score += 0.1  # divergence
    return min(score, 0.6)  # cap

def identity_signals(tx: TransactionRequest) -> float:
    """Evaluate identity attributes (IP, device) changes and sharing."""
    score = 0.0
    # We need to look at previous transactions along the same path or connected nodes.
    # Simple: check if there are other transactions from tx.fromUserId or to tx.toUserId
    # with different IP/device.
    # For brevity, we'll implement a basic check:
    # - If the transaction has identity info, check if the same identity appears in multiple unrelated chains.
    # - If identity is present but changes along a path, increase.
    # For now, we'll score based on:
    # 1. If IP is present, check if same IP appears in transactions that are not directly connected.
    # 2. If device present, similar.
    # We'll keep it simple: if the transaction has IP or device, and it's different from the immediate predecessor along the same edge,
    # we raise score.
    if tx.ipAddress or tx.deviceId:
        # Check previous transaction from same fromUserId to same toUserId? (latest edge)
        prev_tx_id = state.from_to_latest.get((tx.fromUserId, tx.toUserId))
        if prev_tx_id:
            prev_tx = state.tx_store.get(prev_tx_id)
            if prev_tx:
                if tx.ipAddress and prev_tx.get('ipAddress') and tx.ipAddress != prev_tx['ipAddress']:
                    score += 0.15
                if tx.deviceId and prev_tx.get('deviceId') and tx.deviceId != prev_tx['deviceId']:
                    score += 0.15
        # Additionally, if same IP appears across disconnected components, raise risk.
        # We'll check all transactions with same IP but not connected.
        if tx.ipAddress:
            same_ip_txs = [tid for tid, idata in state.identity_store.items() if idata.get('ip') == tx.ipAddress]
            # Check if any of those are not reachable from current transaction's graph
            # This is expensive; we'll skip for now.
            pass
    return min(score, 0.4)

def value_signals(tx: TransactionRequest) -> float:
    """Evaluate value progression along structural paths."""
    score = 0.0
    # We need to find the incoming path(s) that lead to tx.fromUserId.
    # Then check the amount progression along that path.
    # We'll get all paths to tx.fromUserId, and for each, compute the ratio of amounts.
    paths = get_paths_to_node(tx.fromUserId, max_depth=5)
    # For each path, compute the amounts along the edges.
    # We need the amount for each edge in the path.
    # Since we have edge->latest tx, we can get the amount from that tx.
    for path in paths:
        if len(path) < 2:
            continue
        amounts = []
        valid = True
        for i in range(len(path)-1):
            f = path[i]
            t = path[i+1]
            edge_tx_id = state.from_to_latest.get((f, t))
            if not edge_tx_id:
                valid = False
                break
            tx_data = state.tx_store.get(edge_tx_id)
            if not tx_data:
                valid = False
                break
            amounts.append(tx_data['amount'])
        if not valid or len(amounts) < 1:
            continue
        # Now we have amounts along the path leading to tx.fromUserId.
        # Add the current tx amount as the next step.
        amounts.append(tx.amount)
        # Check for consistent decay: we expect amounts to generally decrease along a layering path.
        # But layering often has a factor ~0.99, etc. We'll check if there is a reversal (increase).
        # We'll compute ratio of each step to previous.
        # If all ratios < 1.0, that's consistent decay.
        # If any ratio > 1.0, that's a reversal.
        # Also, if there are large jumps or unusual patterns.
        # Let's compute:
        ratios = []
        for i in range(1, len(amounts)):
            if amounts[i-1] == 0:
                continue
            ratios.append(amounts[i] / amounts[i-1])
        # If all ratios < 1.0 (strict decay), that's low risk (but layering is expected)
        # If any ratio > 1.0, increase risk.
        # Also, if there is a big variation (std dev), increase.
        if ratios:
            if any(r > 1.0 for r in ratios):
                # reversal
                score += 0.3
            # Also check for very small decay (e.g., ratio close to 1) might be more suspicious?
            # We'll add a small amount for high consistency? No, consistent decay is normal.
            # But if ratios are all < 1 but some are far from 0.99, could be abnormal.
            # We'll add a little for variance.
            mean_ratio = sum(ratios) / len(ratios)
            variance = sum((r - mean_ratio)**2 for r in ratios) / len(ratios) if len(ratios)>1 else 0
            if variance > 0.01:  # high variance
                score += 0.1
    # Cap
    return min(score, 0.5)

def combine_scores(structural, identity, value) -> float:
    """Combine component scores into final risk score 0-1."""
    # We'll weight: structural 0.5, identity 0.2, value 0.3
    # But we'll also consider maximum to avoid double counting.
    # Use a simple weighted sum, capped at 1.0.
    # Ensure not to exceed 1.
    weighted = structural * 0.5 + identity * 0.2 + value * 0.3
    # Add a small base to indicate that all transactions have some baseline risk?
    # For now, we'll keep it as is.
    return min(weighted, 1.0)

# ---------- Main Endpoints ----------

@app.get("/ghost-chains/health")
async def health():
    return {"status": "ok"}

@app.post("/ghost-chains/reset")
async def reset(clear: dict = None):
    state.reset()
    return {"clearTransactions": True}

@app.post("/ghost-chains/transactions", response_model=TransactionsResponse)
async def process_transactions(req: TransactionsRequest):
    # 1. Cleanup old transactions (W = 24 hours)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    state.cleanup_old(cutoff)

    results = []
    for tx in req.transactions:
        # Check idempotency
        cached_score = state.get_score(tx.txId)
        if cached_score is not None:
            # If duplicate, return stored score, no state change
            results.append(TransactionResult(txId=tx.txId, riskScore=cached_score))
            continue

        # 2. Compute signals
        struct_score = structural_signals(tx)
        id_score = identity_signals(tx)
        value_score = value_signals(tx)

        # 3. Combine
        final_score = combine_scores(struct_score, id_score, value_score)

        # 4. Store transaction with score
        state.add_transaction(tx, final_score)

        # 5. Append result
        results.append(TransactionResult(txId=tx.txId, riskScore=final_score))

    return TransactionsResponse(transactions=results)

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
