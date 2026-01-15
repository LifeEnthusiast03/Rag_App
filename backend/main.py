from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path

app = FastAPI()

# Enable CORS for your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "PDF Upload API is running"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file is a PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save the file
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "message": "PDF uploaded successfully",
            "filename": file.filename,
            "size": file_path.stat().st_size,
            "path": str(file_path)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    
    finally:
        file.file.close()


@app.get("/uploaded-files")
def list_uploaded_files():
    files = list(UPLOAD_DIR.glob("*.pdf"))
    return {
        "files": [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "path": str(f)
            }
            for f in files
        ]
    }