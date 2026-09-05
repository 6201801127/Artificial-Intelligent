from google import genai
from google.genai import errors


GEMINI_API_KEY = ""
client = genai.Client(api_key=GEMINI_API_KEY)

try:
    interation = client.interactions.create(
        model="gemini-3.6-flash",
        input="Write a poem about the beauty of nature",
        generation_config={
            "temprature": 1.0,
            "max_output_tokens": 100},
    )
    print(interation.output_text)
except errors.BadRequestError as e:
    print(f"Bad request error: {e}")    
    print(f"Error details: {e.details}")
except errors.UnauthorizedError as e:
    print(f"Unauthorized error: {e}")
    print(f"Error details: {e.details}")
except errors.ForbiddenError as e:
    print(f"Forbidden error: {e}")
    print(f"Error details: {e.details}")
    
