A QA BOT for a software company named VeloTriz.\
Uses RAG model (Retrievel-Augemented Generation)
This can handle Questions related to the company and about the products they use.
Since the OpenAI api is paid I have embedded a different model which is similar to OpenAI model.
**STEPS TO USE EXECUTE :**
1)Install the requirements:
    pip install transformers torch pinecone-client sentence-transformers
    pip install pinecone
    pip install -q transformers torch pinecone-client sentence-transformers accelerate bitsandbytes
2)Run the Main.py
**NOTE**
MAKE SURE TO INSTALL THE REQUIRED LIBRARIES
->You can also add your own pinecone API and the your desired model API
-.The model will be further finetuned



Few Questions examples:
what is velotriz?
What features does Velotriz offer?
Who can use Velotriz?
Does Velotriz support multiple languages?
What payment gateways does Velotriz integrate with?
How secure is Velotriz?
What kind of analytics does Velotriz provide?
Can I migrate my existing store to Velotriz?
Is there a free trial available?
How much does Velotriz cost?

The questions can be asked in many formats(lowercase,uppercase,tanglish etc..)
There are 50 questions embedded in the model
You can aslo ask the questions indirectly like"what is the company name",the model can identify it.
The model gives the answer under the keyword circumstances
