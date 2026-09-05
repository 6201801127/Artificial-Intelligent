from google import genai

GEMINI_API_KEY = ""
client = genai.Client(api_key=GEMINI_API_KEY)

interation = client.interactions.create(
    model="gemini-3.6-flash",
    input="Write a poem about the beauty of nature",
    generation_config={"temprature": 1.0, "max_output_tokens": 100},
)
# print(interation)

print(interation.output_text)
