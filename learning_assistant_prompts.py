from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
question_prompt_template="""Based on the following summary, generate {nquestions} questions based on the information given in the summary. 
        Return the questions as a JSON list.

        Summary: {summary}

        Questions (in JSON format):"""
question_template = PromptTemplate(input_variables=["summary","nquestions"],template=question_prompt_template)
rag_prompt_template= """Use the given context to answer the question.\
    If you don't know the answer, say you don't know. 
    Use three sentence maximum and keep the answer concise. 
    Context: {context}"""
rag_template = ChatPromptTemplate.from_messages(
    [
        ("system", rag_prompt_template),
        ("human", "{question}"),
    ])

