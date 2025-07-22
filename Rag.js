import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { YoutubeLoader } from "@langchain/community/document_loaders/web/youtube";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenAI } from "@google/genai"
import dotenv from 'dotenv'
dotenv.config()

const nike10kPdfPath = "./example/Arrays(1D).pdf";

const loader1 = new PDFLoader(nike10kPdfPath);
const loader2 = YoutubeLoader.createFromUrl("https://www.youtube.com/watch?v=MOeGnamlUP4&t=3s", {
  language: "en",
  addVideoInfo: true,
});
const docs1 = await loader1.load();
const docs2 = await loader2.load()

const output = []
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 10,
});
output.push( await splitter.splitDocuments(docs1))
output.push( await splitter.splitDocuments(docs2))
const content = []
output[0].forEach((doc)=>{content.push(doc.pageContent)});
output[1].forEach((doc)=>{content.push(String(doc.pageContent))});
const contents= content.slice(0,99)


async function main() {

    const ai = new GoogleGenAI({
         apiKey: process.env.GOOGLE_API_KEY
    });

    const response = await ai.models.embedContent({
        model: 'gemini-embedding-001',
        contents
    });

    console.log(response.embeddings);
}

main()




