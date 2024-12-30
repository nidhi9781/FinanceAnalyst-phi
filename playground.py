from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
import openai
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
Groq.api_key = os.getenv("GROQ_API_KEY")

web_search_ag = Agent(
    name = "web search agent",
    role = "Search web for India for information",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources in bullets"],
    show_tools_calls = True,
    markdown=True

)

sentiment_agent = Agent(
     name = "Sentiment agent",
    role = "Search and interpret news articles ",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[GoogleSearch()],
    show_tool_calls=True,
    instructions=("Find relevant news aarticles for each company and analyze the sentiment",
    "Provide the sentiments score from 1(negative) to 10(positive) with reasoning sources.",
    "Cite your sources. Be specific and provide links"
    )
)

fin_search_ag = Agent(
    name = "financial agent",
    role = "Get financial data and interpret trends",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info = True)],
    instructions=["Retreive stock prices,analyst recommendations, and key financial data",
                  "Focus on trends and present  data in tables with key insights"],
    show_tools_calls = True,
    markdown=True

)

analyst_agent = Agent(
     name = "Analyst Agent",
    role = "Ensure thoroughness and draw conclusions",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info = True)],
    instructions=["Check output for accuracy and completeness",
                  "Synthesize data to provide final settlemnet score (1-10) with justification"],
    show_tools_calls = True,
    markdown=True
)

app = Playground(agents = [sentiment_agent, fin_search_ag, analyst_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload = True)
