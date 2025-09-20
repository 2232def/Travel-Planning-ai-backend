from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import uvicorn

app = FastAPI(title="Travel Planning AI")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Welcome to the Travel Planning AI API."}


if __name__ == "__main__":

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
