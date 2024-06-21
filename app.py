import chainlit as cl
from datasets import load_dataset
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load dataset
try:
    dataset = load_dataset("HC-85/open-food-facts", split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = None

if dataset:
    # Convert dataset to dictionary format
    documents = [{"content": item["text"]} for item in dataset]

    # Initialize document store and write documents
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(documents)

    # Initialize retriever
    retriever = BM25Retriever(document_store=document_store, top_k=3)

    # Define prompt template
    prompt_template = PromptTemplate(
        prompt="""
        Create a recipe that fits the following nutritional values and dietary requirements based solely on the given documents. If the documents do not contain enough information to create a recipe, state that creating the recipe is not possible with the available information. Ensure the recipe is detailed, including ingredients and preparation steps, and your answer should be no longer than 500 words.
        Documents:{join(documents)}
        Question:{query}
        Answer:
        """,
        output_parser=AnswerParser(),
    )

    # Initialize PromptNode with the model
    HF_TOKEN = os.environ.get("HF_TOKEN")

    prompt_node = PromptNode(
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1", 
        api_key=HF_TOKEN, 
        default_prompt_template=prompt_template
    )

    # Create pipeline and add nodes
    gen_pipeline = Pipeline()
    gen_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    gen_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

    # Define chat start handling function
    @cl.on_chat_start
    async def on_chat_start():
        await cl.Message(author="System", content="Welcome to the Recipe Generator! Please enter your nutritional values and dietary requirements.").send()

    # Define message handling function
    @cl.on_message
    async def main(message: str):
        response = await cl.make_async(gen_pipeline.run)(message)
        sentences = response['answers'][0].answer.split('\n')

        # Check if the last sentence doesn't end with '.', '?', or '!'
        if sentences and not sentences[-1].strip().endswith(('.', '?', '!')):
            # Remove the last sentence
            sentences.pop()

        result = '\n'.join(sentences[1:])
        await cl.Message(author="Bot", content=result).send()
    # Define audio chunk handling function (if needed)
    @cl.on_audio_chunk
    async def handle_audio_chunk(chunk):
        # Placeholder for handling audio chunk
        pass
else:
    print("Dataset could not be loaded. Please check the dataset path and try again.")
