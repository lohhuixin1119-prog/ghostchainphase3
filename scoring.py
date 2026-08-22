from models import TransactionIn
from graph import AMLGraph

def calculate_risk_score(tx: TransactionIn, graph: AMLGraph) -> float:
    """
    Evaluates risk based on Structural, Identity, and Value signals.
    Returns a float between 0.0 and 1.0.
    """
    score = 0.0
    tx_time = graph.parse_time(tx.createdAt)
    
    # 1. STRUCTURAL SIGNALS (Phase 1)
    # If the destination of this transaction has a path back to the sender, it's a cycle.
    if graph.has_path(start_node=tx.toUserId, target_node=tx.fromUserId):
        score += 0.35 
    
    # 2. VALUE & IDENTITY SIGNALS (Phases 2 & 3)
    # Look at money that arrived at the sender *before* they sent this current transaction
    predecessors = graph.get_predecessors(tx.fromUserId, tx_time)
    
    if predecessors:
        for prev_tx in predecessors:
            # --- Phase 3: Value Signal ---
            if tx.amount > prev_tx.amount:
                # Value Trajectory Reversal: The amount increased instead of decayed.
                # High risk: contradicts the expected layering pattern.
                score += 0.40
            else:
                # Consistent Decay: Standard layering behavior.
                # Minor risk increase, but fundamentally less suspicious than a reversal.
                score += 0.05
                
            # --- Phase 2: Identity Signal ---
            # Identity vanished mid-flow (evasion)
            if prev_tx.deviceId and not tx.deviceId:
                score += 0.15
            if prev_tx.ipAddress and not tx.ipAddress:
                score += 0.15
                
            # Identity changed mid-flow (account takeover or proxy)
            if prev_tx.deviceId and tx.deviceId and prev_tx.deviceId != tx.deviceId:
                score += 0.20
            if prev_tx.ipAddress and tx.ipAddress and prev_tx.ipAddress != tx.ipAddress:
                score += 0.20

    # Clamp the final score strictly between 0.0 and 1.0
    return min(max(round(score, 4), 0.0), 1.0)
