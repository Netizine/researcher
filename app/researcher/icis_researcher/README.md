# 🔎 ICIS Researcher

**ICIS Researcher is an autonomous agent designed for comprehensive online research on a variety of tasks.** 

The agent can produce detailed, factual and unbiased research reports, with customization options for focusing on relevant resources, outlines, and lessons. Inspired by the recent [Plan-and-Solve](https://arxiv.org/abs/2305.04091) and [RAG](https://arxiv.org/abs/2005.11401) papers, ICIS Researcher addresses issues of speed, determinism and reliability, offering a more stable performance and increased speed through parallelized agent work, as opposed to synchronous operations.

**Our mission is to empower individuals and organizations with accurate, unbiased, and factual information by leveraging the power of AI.**

#### PIP Package
```bash
$ pip install icis-researcher
```
> **Step 2** - Create .env file with your OpenAI Key and Tavily API key or simply export it
```bash
$ export OPENAI_API_KEY={Your OpenAI API Key here}
```
```bash
$ export TAVILY_API_KEY={Your Tavily API Key here}
```
> **Step 3** - Start Coding using ICIS Researcher in your own code, example:
```python
from icis_researcher import ICISResearcher
import asyncio


async def get_report(query: str, report_type: str) -> str:
    researcher = GPTResearcher(query, report_type)
    report = await researcher.run()
    return report

if __name__ == "__main__":
    query = "what team may win the NBA finals?"
    report_type = "research_report"

    report = asyncio.run(get_report(query, report_type))
    print(report)

```


#### Using a Custom JSON Configuration

If you want to modify the default configuration of ICIS Researcher, you can create a custom JSON configuration file. This allows you to tailor the researcher's behavior to your specific needs. Here's how to do it:

a. Create a JSON file (e.g., `your_config.json`) with your desired settings:

```json
{
  "retrievers": ["google"],
  "fast_llm": "cohere:command",
  "smart_llm": "cohere:command-nightly",
  "max_iterations": 3,
  "max_subtopics": 1
}
```

b. When initializing the GPTResearcher, pass the path to your custom configuration file:

```python
researcher = GPTResearcher(query, report_type, config_path="your_config.json")
```

#### Using Environment Variables

Alternatively, you can set up the same configuration using environment variables instead of a JSON file. Here's how the example from Part 1 would look in your `.env` file:

```
RETRIEVERS=google
FAST_LLM=cohere:command
SMART_LLM=cohere:command-nightly
MAX_ITERATIONS=3
MAX_SUBTOPICS=1
```

Simply add these lines to your `.env` file, and ICIS Researcher will use the environment variables to configure its behavior. This approach provides flexibility when deploying in different environments.