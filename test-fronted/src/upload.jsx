import { useState } from 'react';

export default function PDFUploadTest() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    const pdfFiles = selectedFiles.filter(file => file.type === 'application/pdf');
    
    if (pdfFiles.length === 0) {
      setError('Please select valid PDF files');
      setFiles([]);
    } else if (pdfFiles.length !== selectedFiles.length) {
      setError(`Only ${pdfFiles.length} PDF file(s) selected. Non-PDF files were ignored.`);
      setFiles(pdfFiles);
    } else {
      setFiles(pdfFiles);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select at least one file first');
      return;
    }

    setUploading(true);
    setError(null);
    setResponse(null);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const res = await fetch('http://localhost:8000/upload-pdfs', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await res.json();
      setResponse({ 
        message: data.message, 
        files: data.files 
      });
      setFiles([]);
      document.getElementById('fileInput').value = '';
      fetchUploadedFiles();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const fetchUploadedFiles = async () => {
    try {
      const res = await fetch('http://localhost:8000/uploaded-files');
      const data = await res.json();
      setUploadedFiles(data.files);
    } catch (err) {
      console.error('Error fetching files:', err);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            PDF Upload Test
          </h1>
          <p className="text-gray-600 mb-6">
            Test your FastAPI backend PDF upload functionality
          </p>

          {/* Upload Section */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 mb-6 hover:border-indigo-400 transition-colors">
            <div className="text-center">
              <svg
                className="mx-auto h-12 w-12 text-gray-400 mb-4"
                stroke="currentColor"
                fill="none"
                viewBox="0 0 48 48"
              >
                <path
                  d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <label
                htmlFor="fileInput"
                className="cursor-pointer text-indigo-600 hover:text-indigo-500 font-medium"
              >
                Choose a PDF file
              </label>
              <input
                id="fileInput"
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="hidden"
                multiple
              />
              <p className="text-sm text-gray-500 mt-2">or drag and drop</p>
            </div>
          </div>

          {files.length > 0 && (
            <div className="mb-6 p-4 bg-blue-50 rounded-lg">
              <p className="text-sm font-medium text-gray-700 mb-2">
                Selected {files.length} file(s):
              </p>
              <div className="space-y-2">
                {files.map((file, idx) => (
                  <div key={idx} className="flex justify-between items-center text-xs">
                    <span className="text-indigo-600 font-medium truncate">{file.name}</span>
                    <span className="text-gray-500 ml-2">{formatBytes(file.size)}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Total: {formatBytes(files.reduce((acc, f) => acc + f.size, 0))}
              </p>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={files.length === 0 || uploading}
            className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-colors ${
              files.length === 0 || uploading
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
          >
            {uploading ? `Uploading ${files.length} file(s)...` : `Upload ${files.length > 0 ? files.length : ''} PDF${files.length !== 1 ? 's' : ''}`}
          </button>

          {/* Response Messages */}
          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-800 text-sm font-medium">Error: {error}</p>
            </div>
          )}

          {response && (
            <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-green-800 font-medium mb-2">
                ✓ {response.message}
              </p>
              {response.files && (
                <div className="text-sm text-green-700 space-y-2 mt-2">
                  {response.files.map((file, idx) => (
                    <div key={idx} className="border-t border-green-200 pt-2">
                      <p className="font-medium">File {idx + 1}: {file.filename}</p>
                      <p>Size: {formatBytes(file.size)}</p>
                      <p className="text-xs text-green-600">Path: {file.path}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* List Uploaded Files */}
          <div className="mt-8">
            <button
              onClick={fetchUploadedFiles}
              className="mb-4 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium text-gray-700 transition-colors"
            >
              Refresh File List
            </button>

            {uploadedFiles.length > 0 && (
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                  <h3 className="font-medium text-gray-700">
                    Uploaded Files ({uploadedFiles.length})
                  </h3>
                </div>
                <div className="divide-y divide-gray-200">
                  {uploadedFiles.map((file, idx) => (
                    <div
                      key={idx}
                      className="px-4 py-3 hover:bg-gray-50 transition-colors"
                    >
                      <p className="text-sm font-medium text-gray-800">
                        {file.name}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {formatBytes(file.size)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}