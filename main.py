import os
import ssl
import httpx
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Monkey patch requests to disable SSL verification for LangSmith
from langchain_openai import ChatOpenAI
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.poolmanager import PoolManager


class NoSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = ssl.create_default_context()
        kwargs["ssl_context"].check_hostname = False
        kwargs["ssl_context"].verify_mode = ssl.CERT_NONE
        return super().init_poolmanager(*args, **kwargs)


# Apply to all requests sessions
original_session_init = requests.Session.__init__


def patched_session_init(self, *args, **kwargs):
    original_session_init(self, *args, **kwargs)
    adapter = NoSSLAdapter()
    self.mount("https://", adapter)
    self.mount("http://", adapter)
    self.verify = False


requests.Session.__init__ = patched_session_init

# Disable SSL warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create httpx client with SSL disabled
httpx_client = httpx.Client(verify=False)


def main():
    print("Hello from langchain-course!")
    information = """
    Elon Musk (prononcé en anglais : /ˈiːlɒn ˈmʌsk/), né le 28 juin 1971 à Pretoria (Afrique du Sud), est un entrepreneur, homme d'affaires et homme politique de nationalité sud-africaine, canadienne et américaine. Il est depuis 2024 la personne la plus riche du monde et depuis 2026, la première personne de l'histoire à posséder une fortune de plus de 1000 milliards de dollars (billionnaire)[1].
    Elon Musk commence sa carrière en affaires comme cofondateur de la société de logiciels Zip2 avec son frère, Kimbal Musk. La start-up est acquise par Compaq pour 307 millions de dollars en 1999. La même année, Musk cofonde la banque en ligne X.com, qui fusionne avec Confinity en 2000 pour former PayPal. eBay rachète PayPal en 2002 pour 1,5 milliard de dollars.
    """

    summary_template = """
    given the information {information} about a person, I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # llm = ChatGroq(
    #     model="llama-3.3-70b-versatile", temperature=0.7, http_client=httpx_client
    # )

    # llm = ChatOpenAI(
    #     model="gpt-5", temperature=0, http_client=httpx_client
    # )

    llm = ChatOllama(model="gemma3:270m", temperature=0, http_client=httpx_client)

    chain = summary_prompt_template | llm
    result = chain.invoke(input={"information": information})

    print("\n--- Result ---")
    print(result.content)


if __name__ == "__main__":
    main()
