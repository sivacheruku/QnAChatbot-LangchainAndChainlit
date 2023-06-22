# QnAChatbot-LangChainAndChainlit
---A Chatbot that uses LangChain to answer questions over private data---

This project is focussed on creating a chatbot from scratch, using _Large Language Models._ leveraging the power of OpenAI.
The steps involved are:
1. Loading Documents: Can be done using _DirectoryLoader_
2. Splitting Documents: Split documents into smaller chunks for processing. This helps to adhere to the token size limits of the Large Language models that we will be using.
3. Embedding documents with OpenAI: Once the documents are split, we need to embed them using OpenAI's language model. We can use the OpenAIEmbeddings class from LangChain to embed the documents.
4. Creating Vector storage for embedded documents: We are creating a new Pinecone vector index using the _Pinecone.from_documents()_ method.
5. Finding similar documents: We define a function to find similar documents based on the given query.
6. Final output: We will create a question-answering system using the OpenAI class from LangChain and a pre-built question-answering chain.



# Setting up the environment
The required modules are in the _requirements.txt_ file. 

--- You need to have CUDA on your system for these requirements to work properly without any dependency issues ---
