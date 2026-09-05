import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "My_First_RAG_Project"

llm = ChatOllama(
    # model="gemini-3.6-flash",
    model="gemma3:1b",
    # google_api_key="",
)
llm_response = llm.invoke("Explain the concept of Reinforcement Learning in one line.")
# print(llm_response)


# Parser output

output_parser = StrOutputParser()
parsed_response = output_parser.invoke(llm_response)
# print(parsed_response)

# Simple chain
chain = llm | output_parser
structured_out = chain.invoke("Tell me a short joke about a vector database")
print(structured_out)

# Structured output parser


class MobileReview(BaseModel):
    phone_model: str = Field(description="Name and model of the Phone.")
    rating: int = Field(description="Rating of the phone out of 5.")
    pros: list[str] = Field(description="List of pros of the phone.")
    cons: list[str] = Field(description="List of cons of the phone.")
    summary: str = Field(description="Summary of the review.")


review_text = """
I recently bought the Samsung Galaxy S24 and I am very impressed with it.
The phone has a beautiful 6.2-inch AMOLED display, excellent camera quality,
and very good performance. The Snapdragon processor handles games and
multitasking smoothly.

The battery easily lasts a full day with normal usage, and the phone feels
premium and comfortable to hold. I also like the software experience and
the long-term software update support.

However, the phone is quite expensive compared with some competitors.
The charging speed is also slower than many other phones in this price range,
and the charger is not included in the box.

Overall, the Samsung Galaxy S24 is an excellent flagship phone with great
performance, cameras, display, and software. I would give it a rating of 4
out of 5.
"""

structured_llm = llm.with_structured_output(MobileReview)
output = structured_llm.invoke(review_text)
print(output)
print(output.phone_model)
print(output.rating)
print(output.pros)
print(output.cons)


# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [("human", "Tell me a short joke about {topic}")]
)
prompt_template.invoke({"topic": "Programming"})

chain = prompt_template | llm | output_parser
chain.invoke({"topic": "Programer"})


# All in one

# Define the prompt
prompt = ChatPromptTemplate.from_messages([("human", "Tell me a short joke about {topic}")])

# Initialize the LLM
# llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key="")

# define the output parser
output_parser = StrOutputParser()

# compose the chain

chain = prompt | llm | output_parser

# Use the chain to generate a joke about a specific topic
result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)


# LLM Message
system_msg = SystemMessage(
    content="You are a helpful assistant that provides information about programming languages."
)
human_msg = HumanMessage(content="Can you explain the difference between Python and Java?")
llm_response = llm.invoke([system_msg, human_msg])
print(llm_response)

template = ChatPromptTemplate.from_messages(
    {
        "system": "You are a helpful assistant that provides information about programming languages.",
        "human": "Tell me about {topic}.",
    }
)

prompt_response = template.invoke({"topic": "Programming"})
print(prompt_response)
