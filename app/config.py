from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
