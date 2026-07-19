from fastapi import FastAPI
from pydantic import BaseModel

# from predictor import predict
from app.predictor import predict

app = FastAPI(
    title="Product Attribute Extraction API",
    description="Extract structured attributes from product descriptions",
    version="1.0.0",
)


class ProductRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Product Attribute Extraction API is running."
    }


@app.post("/extract")
def extract_attributes(request: ProductRequest):

    result = predict(request.text)

    return {
        "description": request.text,
        "attributes": result,
    }