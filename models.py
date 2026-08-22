from pydantic import BaseModel
from typing import List, Optional

class TransactionIn(BaseModel):
    txId: str
    fromUserId: str
    toUserId: str
    amount: float
    createdAt: str
    ipAddress: Optional[str] = None
    deviceId: Optional[str] = None

class TransactionOut(BaseModel):
    txId: str
    riskScore: float

class TransactionsRequest(BaseModel):
    transactions: List[TransactionIn]

class TransactionsResponse(BaseModel):
    transactions: List[TransactionOut]

class ResetRequest(BaseModel):
    clearTransactions: bool
