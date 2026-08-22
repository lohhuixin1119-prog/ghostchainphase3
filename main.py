from fastapi import FastAPI
import uvicorn

from models import TransactionsRequest, TransactionsResponse, TransactionOut, ResetRequest
from graph import AMLGraph
from scoring import calculate_risk_score

app = FastAPI(title="Ghost Chains AML API")

# Global in-memory state
aml_state = AMLGraph()

@app.get("/")
@app.head("/")
async def root():
    """Satisfy Render's automatic load-balancer health checks to prevent 404 logs."""
    return {"status": "ok"}

@app.get("/ghost-chains/health")
async def health_check():
    """Verify service availability."""
    return {"status": "ok"}

@app.post("/ghost-chains/reset")
async def reset_state(payload: ResetRequest):
    """Clears all internal state, graph caches, and derived structures."""
    if payload.clearTransactions:
        aml_state.reset()
        return {"clearTransactions": True}
    return {"error": "Invalid payload"}

@app.post("/ghost-chains/transactions", response_model=TransactionsResponse)
async def process_transactions(request: TransactionsRequest):
    """Processes streaming transactions sequentially."""
    results = []
    
    for tx in request.transactions:
        if tx.txId in aml_state.score_cache:
            results.append(TransactionOut(txId=tx.txId, riskScore=aml_state.score_cache[tx.txId]))
            continue
            
        # Score the transaction BEFORE adding it to the state
        risk_score = calculate_risk_score(tx, aml_state)
        
        # Commit to graph state
        aml_state.add_transaction(tx)
        
        aml_state.score_cache[tx.txId] = risk_score
        results.append(TransactionOut(txId=tx.txId, riskScore=risk_score))
        
    return TransactionsResponse(transactions=results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
