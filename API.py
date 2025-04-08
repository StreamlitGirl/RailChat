from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import similarity5  # Importer ton module chatbot (sans Streamlit)

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")
    
    if not question:
        return JSONResponse(content={"error": "Field 'question' is required"}, status_code=400)
    
    try:
        
        response = similarity5.predict_answer(question)  

        return JSONResponse(content={"answer": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
if __name__ == "__main__":
    chat();
