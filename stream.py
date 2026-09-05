from google import genai

GEMINI_API_KEY = " "
client = genai.Client(api_key=GEMINI_API_KEY)

interation = client.interactions.create(
    model="gemini-3.6-flash",
    input="Write a poem about the beauty of nature in 10 paragraph",
<<<<<<< HEAD
    stream=True,  #
=======
    stream=True # 
)
