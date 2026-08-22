from fastapi import FastAPI
import uvicorn

from models import TransactionsRequest, TransactionsResponse, TransactionOut, ResetRequest
from graph import AMLGraph
from scoring import calculate_risk_score

app = FastAPI(title="Ghost Chains AML API")

# Global in-memory state
aml_state = AMLGraph()

@app.get("/ghost-chains/health")
async def health_check():
    """Phase 1: Verify service availability."""
    return {"status": "ok"}

@app.post("/ghost-chains/reset")
async def reset_state(payload: ResetRequest):
    """Phase 1: Clears all internal state, graph caches, and derived structures."""
    if payload.clearTransactions:
        aml_state.reset()
        return {"clearTransactions": True}
    return {"error": "Invalid payload"}

@app.post("/ghost-chains/transactions", response_model=TransactionsResponse)
async def process_transactions(request: TransactionsRequest):
    """Phase 1-3: Processes streaming transactions sequentially."""
    results = []
    
    for tx in request.transactions:
        # 1. Idempotency Check: Return cached score if already processed
        if tx.txId in aml_state.score_cache:
            results.append(TransactionOut(txId=tx.txId, riskScore=aml_state.score_cache[tx.txId]))
            continue
            
        # 2. Evaluate Score BEFORE updating state
        # (This ensures the transaction is scored against historical context only)
        risk_score = calculate_risk_score(tx, aml_state)
        
        # 3. Commit to graph state
        aml_state.add_transaction(tx)
        
        # 4. Cache and append to response
        aml_state.score_cache[tx.txId] = risk_score
        results.append(TransactionOut(txId=tx.txId, riskScore=risk_score))
        
    return TransactionsResponse(transactions=results)

if __name__ == "__main__":
    # Start the server on port 8080 as requested in the briefing
    uvicorn.run(app, host="0.0.0.0", port=8080)
