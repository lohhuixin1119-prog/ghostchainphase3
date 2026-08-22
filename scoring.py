from models import TransactionIn
from graph import AMLGraph

def calculate_risk_score(tx: TransactionIn, graph: AMLGraph) -> float:
    tx_time = graph.parse_time(tx.createdAt)
    
    # 1. STRUCTURAL SIGNALS (Phase 1)
    # Check if this transaction closes a chronological cycle back to the sender
    is_cycle = graph.has_temporal_path(start_node=tx.toUserId, target_node=tx.fromUserId, max_time=tx_time)
    
    # Base risk is heavily elevated if a loop is detected
    base_score = 0.50 if is_cycle else 0.00
    
    # 2. VALUE & IDENTITY SIGNALS (Phases 2 & 3)
    predecessors = graph.get_immediate_predecessors(tx.fromUserId, tx_time)
    
    if not predecessors:
        return base_score
        
    max_path_risk = 0.0
    
    for prev_tx in predecessors:
        path_risk = base_score
        
        # --- Phase 3: Value Signal ---
        if tx.amount > prev_tx.amount:
            # Value Trajectory Reversal (Very suspicious, breaks layering rules)
            path_risk += 0.40
        else:
            # Consistent Decay (Expected in standard layering)
            path_risk += 0.10
            
        # --- Phase 2: Identity Signal ---
        # Device ID Evasion/Switching
        if prev_tx.deviceId and not tx.deviceId:
            path_risk += 0.20  
        elif prev_tx.deviceId and tx.deviceId and prev_tx.deviceId != tx.deviceId:
            path_risk += 0.25 
            
        # IP Address Evasion/Switching
        if prev_tx.ipAddress and not tx.ipAddress:
            path_risk += 0.20  
        elif prev_tx.ipAddress and tx.ipAddress and prev_tx.ipAddress != tx.ipAddress:
            path_risk += 0.25 
            
        if path_risk > max_path_risk:
            max_path_risk = path_risk

    # Clamp the final score strictly between 0.0 and 1.0
    return min(max(round(max_path_risk, 4), 0.0), 1.0)
