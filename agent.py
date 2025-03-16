import time
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

#  Wikipedia Setup
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=3, lang="en")
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

#  Local LLM using Ollama (Mistral)
llm = OllamaLLM(model="mistral")

#  Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#  Initialize Agent
agent = initialize_agent(
    tools=[wiki_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=5,
)

#  User Input Loop
while True:
    user_input = input("\nEnter your query (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            response = agent.invoke(user_input)

            # Ensure proper response formatting
            if isinstance(response, dict) and 'output' in response:
                print(f"\n🔹 Answer: {response['output']}")
            else:
                print(f"\n🔹 Answer: {response}")

            break  # Exit loop if successful

        except Exception as e:
            retry_count += 1
            print(f" Error (attempt {retry_count}/{max_retries}): {e}")
            time.sleep(1)

            if retry_count == max_retries:
                print(" Max retries reached. Could not complete query.")
